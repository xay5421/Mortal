# Mortal 训练指南

## 环境搭建

```bash
# 1. 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 2. 创建 Python 环境（conda 或 venv）
conda create -n mortal python=3.12
conda activate mortal

# 3. 安装 PyTorch（根据你的 CUDA 版本选择）
# RTX 5070 等 Blackwell 架构需要 cu128+
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install tqdm toml tensorboard maturin requests

# 4. 编译 libriichi
cd libriichi
maturin develop --release
cd ..

# 5. 验证
python -c "from libriichi.consts import ACTION_SPACE; print('OK')"
python -c "import torch; print(torch.zeros(1).cuda())"
```

## 训练流程概览

```
1. 准备权重  →  2. 下载数据  →  3. 训练 GRP  →  4. 离线训练 Mortal  →  5. 自对弈 RL (可选)  →  6. 评估  →  7. 部署
```

## 1. 准备预训练权重

```bash
cd mortal

# 复制 majsoul-bot 里的推理权重
cp /path/to/mortal_inference.pth original.pth

# 转换为训练 checkpoint（不依赖 libriichi 的简化版）
python convert_weights_simple.py original.pth mortal.pth

# 修复 optimizer 兼容性（首次必须）
python -c "import torch;s=torch.load('mortal.pth',weights_only=True,map_location='cpu');s['config']['control']['online']=True;torch.save(s,'mortal.pth');print('Done')"

# 准备 baseline 和配置
cp config.train.toml config.toml
python setup_training.py
```

## 2. 下载训练数据

训练数据为天凤鳳凰卓（最高段位）的牌谱，转换为 mjai JSON 格式。

### 方式 A：使用 houou-logs（推荐，快）

```bash
# 需要 Python 3.12+，可以用单独的 conda 环境
conda create -n houou python=3.12 -y
conda activate houou
pip install git+https://github.com/Apricot-S/houou-logs.git

# 先手动下载年度索引包: https://tenhou.net/sc/raw/scraw2024.zip
houou-logs import db/2024.db scraw2024.zip
houou-logs download db/2024.db --players 4 --length h

# 回到训练环境转换
conda activate mortal
python download_tenhou.py --from-db db/2024.db --output data
```

### 方式 B：使用内置脚本（慢但简单）

```bash
# 下载最近 7 天
python download_tenhou.py

# 下载指定年份（需要先下载 scraw20XX.zip 到 mortal/ 目录）
python download_tenhou.py --year 2024

# 限制数量（调试用）
python download_tenhou.py --limit 100
```

### 方式 C：本地 mjlog 文件转换

```bash
python download_tenhou.py --convert-dir /path/to/mjlog/files --output data
```

### 数据量参考

| 数据量 | 效果 |
|--------|------|
| 100 局 | 跑通流程 |
| 1,000 局 | 初步可用 |
| 10,000 局 | 有明显提升 |
| 100,000+ 局 | 接近原版 Mortal 水平 |

## 3. 训练 GRP（顺位预测模型）

GRP 用于计算训练 reward，**必须在训练 Mortal 之前完成**。

```bash
rm -f grp.pth   # 删掉 setup_training.py 生成的随机权重
python train_grp.py
```

GRP 很小（~10 万参数），几分钟就能训好。

## 4. 训练 Mortal

```bash
python train.py
```

### ⚠️ 重要注意事项

- **添加新数据后**必须删除文件索引缓存，否则新数据不会被使用：
  ```bash
  rm -f file_index.pth
  python train.py
  ```
- 训练会从最近的 checkpoint 自动恢复（读取 `mortal.pth`）
- 每 `save_every=400` 步自动保存到 `mortal.pth`
- 每 `test_every=20000` 步自动跑 1v3 评估
- 如果评估结果是历史最好，会额外保存到 `best.pth`
- `num_epochs=1` 时数据跑完一遍就停止，改大可多跑几遍

### 监控训练

```bash
# 在 mortal/ 目录下
tensorboard --logdir runs --bind_all
# 浏览器打开 http://localhost:6006
```

关键指标：
| 指标 | 含义 | 好的趋势 |
|------|------|---------|
| `dqn_loss` | Q 值预测误差（核心） | 稳定下降 |
| `cql_loss` | 防 Q 值过高估计 | 稳定 |
| `next_rank_loss` | 顺位预测（辅助） | 下降 |
| `test_play/avg_ranking` | 1v3 平均顺位 | <2.50 就超过多数人 |
| `test_play/avg_pt` | 平均得分 | >0 就是正收益 |

### 顺位奖励配置（pts）

config.toml 中的 `pts` 应匹配你打的段位：

```toml
# 雀魂金之间四人南 (55/15/0/-70)
pts = [5.5, 1.5, 0.0, -7.0]

# 雀魂玉之间四人南 (75/25/0/-100)
pts = [7.5, 2.5, 0.0, -10.0]

# 雀魂王座之间四人南 (105/35/0/-140)
pts = [10.5, 3.5, 0.0, -14.0]

# 天凤鳳凰卓（原版 Mortal）
pts = [6.0, 4.0, 2.0, 0.0]
```

## 5. 自对弈在线 RL（可选，进阶）

离线训练是从固定的人类牌谱学习。自对弈（online RL）让模型自己打牌产生训练数据，不断迭代提升——类似 AlphaZero 的思路。

### 架构

```
┌─────────────┐     TCP :5000     ┌─────────────┐     TCP :5000     ┌─────────────┐
│  client.py  │ ←── get_param ──→ │  server.py  │ ←── drain ──────→ │  train.py   │
│  (对弈器)   │ ──→ submit_replay │  (中继站)   │ ←── submit_param  │  (训练器)   │
│             │                   │             │                   │ online=true │
│ 1v3 自对弈  │                   │ buffer/     │                   │ DQN 训练    │
│ 产生牌谱    │                   │ drain/      │                   │ 消费牌谱    │
└─────────────┘                   └─────────────┘                   └─────────────┘
```

- **server.py**：中继服务器，管理参数分发和牌谱缓冲
- **client.py**：对弈器，用最新权重跑 1v3 自对弈（trainee vs baseline），产生牌谱
- **train.py** (`online=true`)：训练器，消费牌谱做 DQN 更新，把新权重推回 server

### 一键启动

```bash
cd mortal

# 1. 准备配置（首次使用复制模板）
cp config.selfplay.toml config.toml
# 按需修改 pts（顺位分）、device 等

# 2. 确保 mortal.pth / baseline.pth / grp.pth 存在
python setup_training.py

# 3. 一键启动（server + trainer + client）
python self_play.py
```

### 更多用法

```bash
# 指定配置文件
MORTAL_CFG=config.selfplay.toml python self_play.py

# Dry-run：只检查配置，不启动
python self_play.py --dry-run

# 多 client 并行（有多 GPU 时加速数据生产）
python self_play.py --num-clients 2

# 单独启动各组件（适合多机分布式）
python self_play.py --server-only         # 机器 A
python self_play.py --client-only         # 机器 B（可多开）
python train.py                           # 机器 A（需另开终端）
```

### 关键参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `train_play.default.games` | 每轮自对弈局数 | 400~800 |
| `train_play.default.boltzmann_epsilon` | 探索率 | 0.005~0.01 |
| `train_play.default.boltzmann_temp` | 探索温度 | 0.05 |
| `online.server.capacity` | buffer 容量 | 800 |
| `freeze_bn.mortal` | 冻结 BN | online 必须 `true` |
| `optim.scheduler.peak` | 学习率 | 1e-5（比离线小） |

### 工作流程

1. trainer 启动后把当前模型参数推送给 server
2. client 从 server 拉取最新权重
3. client 用权重跑 1v3 对弈（trainee vs baseline），有小概率随机探索
4. client 把产生的牌谱提交给 server 的 buffer
5. trainer 从 server drain 牌谱，做一步训练更新
6. trainer 把更新后的权重推回 server
7. 循环 2~6

每 `test_every` 步，trainer 会自动跑一次 1v3 评估，如果是历史最好成绩会保存到 `best.pth`。

### 监控

```bash
# TensorBoard
tensorboard --logdir runs --bind_all

# 日志
tail -f self_play_logs/self_play_*.log
```

### 硬件要求

自对弈需要 GPU 同时做推理（client）和训练（trainer）：
- **最低**：1 张 GPU，分时共享（慢但可行）
- **建议**：2+ 张 GPU，1 个 trainer + 多个 client 并行
- **多机**：server 监听 0.0.0.0，client 可以在其他机器上连接

### 离线 vs 自对弈

| | 离线训练 (`online=false`) | 自对弈 (`online=true`) |
|---|---|---|
| 数据来源 | 天凤人类牌谱 | 模型自己打牌产生 |
| CQL 正则化 | 有（防过估计） | 无（数据是当前策略） |
| BN | 可不冻结 | 必须冻结 |
| 学习率 | 1e-5 ~ 3e-5 | 5e-6 ~ 1e-5 |
| 适合 | 有大量牌谱时 | 离线训练收敛后进一步提升 |

建议先用离线训练打底，再切换到自对弈微调。

## 6. 评估

```bash
# 1v3 自对弈：训练后的模型 vs baseline
python one_vs_three.py
```

输出示例：`challenger rankings: [140 130 120 110] (2.40, +5.2pt)`
- avg_rank < 2.50 → 比 baseline 强
- avg_rank < 2.45 → 明显更强

## 7. 部署到 majsoul-bot

训练好的权重直接复制替换即可：

```bash
cp mortal.pth /path/to/majsoul-bot/Mortal/mortal/mortal.pth
# 或者用最佳版本
cp best.pth /path/to/majsoul-bot/Mortal/mortal/mortal.pth
```

格式完全兼容，不需要额外转换。

## 显存不够？

依次尝试：
1. `enable_compile = false`（关闭编译，省启动时间）
2. 减小 `batch_size`（512→256→128），相应增大 `opt_step_every`
3. 确认 `enable_amp = true`
4. `num_workers = 0`

## 模型结构

| 配置 | 参数量 | 说明 |
|------|--------|------|
| 256ch / 54blocks | ~24M | 当前配置（原版 Mortal 最强） |
| 192ch / 40blocks | ~18M | 原版默认 |
| 128ch / 30blocks | ~8M | 轻量版 |

模型大小由 config 的 `[resnet]` 决定，但必须与权重文件匹配。
不能随意改大小，除非从零训练。
