"""
一站式训练准备脚本。

在 convert_weights.py 之后运行，会：
1. 用 mortal.pth 创建 baseline.pth（测试对比用）
2. 创建一个简单的 GRP 模型（训练 reward 计算用）
3. 检查训练数据目录

用法:
    python setup_training.py
"""

import os
import sys
import shutil
import torch
from pathlib import Path

def main():
    # 1. 检查 mortal.pth
    if not os.path.exists('mortal.pth'):
        print("❌ mortal.pth 不存在！先运行 convert_weights.py")
        sys.exit(1)
    print("✓ mortal.pth 存在")

    # 2. 创建 baseline.pth（用当前模型作为对比基准）
    if not os.path.exists('baseline.pth'):
        print("📋 复制 mortal.pth → baseline.pth（作为测试对比基准）")
        shutil.copy('mortal.pth', 'baseline.pth')
    print("✓ baseline.pth 存在")

    # 3. 创建 GRP 模型
    if not os.path.exists('grp.pth'):
        print("🔧 创建初始 GRP 模型...")
        from model import GRP
        grp = GRP(hidden_size=64, num_layers=2)
        grp_state = {'model': grp.state_dict()}
        torch.save(grp_state, 'grp.pth')
        print("  注意：GRP 是随机初始化的，reward 估算不精确但可以先用")
        print("  如果有天凤牌谱，之后可以用 train_grp.py 单独训练 GRP")
    print("✓ grp.pth 存在")

    # 4. 检查训练数据
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    gz_files = list(data_dir.rglob('*.json.gz'))
    if len(gz_files) == 0:
        print(f"\n⚠️  data/ 目录为空！需要放入训练数据")
        print(f"   天凤牌谱下载: https://tenhou.net/sc/raw/")
        print(f"   格式: mjai JSON, gzip 压缩 (.json.gz)")
        print(f"   放到 data/ 目录下即可（支持子目录）")
    else:
        print(f"✓ 找到 {len(gz_files)} 个训练数据文件")

    # 5. 创建输出目录
    for d in ['runs', 'test_play_logs', '1v3_logs']:
        os.makedirs(d, exist_ok=True)

    # 6. 检查 config.toml
    if not os.path.exists('config.toml'):
        if os.path.exists('config.train.toml'):
            print("\n📋 复制 config.train.toml → config.toml")
            shutil.copy('config.train.toml', 'config.toml')
        else:
            print("\n❌ config.toml 不存在！")
            sys.exit(1)
    print("✓ config.toml 存在")

    print("\n" + "="*50)
    if len(gz_files) > 0:
        print("✅ 一切就绪！运行 python train.py 开始训练")
    else:
        print("⏳ 还差训练数据，放入后运行 python train.py")
    print("="*50)

    print("""
📖 训练相关命令:
  python train.py              # 开始训练（离线监督学习）
  python one_vs_three.py       # 1v3 评估（训练后的模型 vs baseline）
  tensorboard --logdir runs    # 查看训练曲线

📊 关键指标:
  avg_rank < 2.50  →  超过多数人类
  avg_rank < 2.45  →  天凤特上级
  avg_pt > 0       →  稳定正收益
""")

if __name__ == '__main__':
    main()
