# Mortal 训练指南（RTX 5070 8GB）

## 环境搭建

```bash
# 1. 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 2. 创建 Python 环境
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# Linux:
# source .venv/bin/activate

# 3. 安装依赖
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install tqdm toml tensorboard maturin

# 4. 编译 libriichi
cd libriichi
maturin develop --release
cd ..
```

## 准备预训练权重

将 majsoul-bot 里的推理权重转换为训练格式：

```bash
cd mortal

# 复制权重文件过来（从 majsoul-bot 或其他来源）
cp /path/to/mortal_inference.pth original.pth

# 转换为完整 checkpoint
python convert_weights.py original.pth mortal.pth
```

## 准备训练数据

训练数据为 mjai 格式的 `.json.gz` 文件，放在 `mortal/data/` 目录下：

```
mortal/data/
├── 2024/
│   ├── game001.json.gz
│   ├── game002.json.gz
│   └── ...
└── 2025/
    └── ...
```

天凤牌谱可以从 https://tenhou.net/sc/raw/ 下载，用 `mortal/data/mjlog2mjai.py` 转换。

## 开始训练

```bash
cd mortal

# 复制训练配置（config.toml 被 gitignore，需要从模板复制）
cp config.train.toml config.toml

python train.py
```

训练参数在 `config.train.toml` 中，已针对 RTX 5070 8GB 显存配置好：
- batch_size=64 + 梯度累积 8 步 = 等效 batch 512
- AMP 混合精度 (省一半显存)
- 学习率 3e-5（fine-tune 用小学习率）

## 监控训练

```bash
tensorboard --logdir mortal/runs
```

关键指标：
- `loss/dqn_loss`: Q 值损失，应该稳定下降
- `test_play/avg_ranking`: 平均顺位，<2.50 就超过多数人类
- `test_play/avg_pt`: 平均 pt，>0 就是天凤特上级水平

## 评估（1v3 自对弈）

```bash
cd mortal
python one_vs_three.py
```

## 部署到 majsoul-bot

训练好的 `mortal.pth` 可以直接给 majsoul-bot 使用（格式兼容）。

## 显存不够？

如果 OOM，依次尝试：
1. 减小 batch_size（64→32），同时增大 opt_step_every（8→16）
2. 确认 enable_amp = true
3. 减少 num_workers（2→0）
