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
1. 准备权重  →  2. 下载数据  →  3. 训练 GRP  →  4. 训练 Mortal  →  5. 评估  →  6. 部署
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

## 5. 评估

```bash
# 1v3 自对弈：训练后的模型 vs baseline
python one_vs_three.py
```

输出示例：`challenger rankings: [140 130 120 110] (2.40, +5.2pt)`
- avg_rank < 2.50 → 比 baseline 强
- avg_rank < 2.45 → 明显更强

## 6. 部署到 majsoul-bot

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
