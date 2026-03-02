"""
将推理用的 mortal.pth（只含 mortal + dqn 权重）转换为
train.py 能加载的完整 checkpoint 格式。

用法:
    python convert_weights.py <input.pth> [output.pth]

如果不指定 output，默认输出到 mortal.pth
"""

import sys
import torch
from datetime import datetime
from model import Brain, DQN, AuxNet

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.pth> [output.pth]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'mortal.pth'

    print(f"Loading {input_path} ...")
    state = torch.load(input_path, weights_only=True, map_location='cpu')

    # 检查是否已经是完整 checkpoint
    if all(k in state for k in ['mortal', 'current_dqn', 'optimizer', 'aux_net']):
        print("Already a full checkpoint, nothing to do.")
        return

    # 提取已有信息
    cfg = state.get('config', {})
    version = cfg.get('control', {}).get('version', 4)
    conv_channels = cfg.get('resnet', {}).get('conv_channels', 256)
    num_blocks = cfg.get('resnet', {}).get('num_blocks', 54)

    print(f"Model: version={version}, conv_channels={conv_channels}, num_blocks={num_blocks}")

    # 验证权重能加载
    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks)
    dqn = DQN(version=version)
    aux_net = AuxNet((4,))

    mortal.load_state_dict(state['mortal'])
    print(f"  mortal: {sum(p.numel() for p in mortal.parameters()):,} params ✓")

    dqn.load_state_dict(state['current_dqn'])
    print(f"  dqn: {sum(p.numel() for p in dqn.parameters()):,} params ✓")

    # aux_net 如果存在就加载，不存在就随机初始化
    if 'aux_net' in state:
        aux_net.load_state_dict(state['aux_net'])
        print(f"  aux_net: loaded from checkpoint ✓")
    else:
        print(f"  aux_net: randomly initialized (not in source checkpoint)")

    # 构建完整 checkpoint
    from torch import optim
    from torch.amp import GradScaler

    # 创建 dummy optimizer/scheduler 状态
    all_params = list(mortal.parameters()) + list(dqn.parameters()) + list(aux_net.parameters())
    optimizer = optim.AdamW([{'params': all_params}], lr=3e-5)
    scaler = GradScaler('cpu', enabled=False)

    full_state = {
        'mortal': mortal.state_dict(),
        'current_dqn': dqn.state_dict(),
        'aux_net': aux_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': {'last_epoch': 0, '_step_count': 1, '_last_lr': [3e-5]},
        'scaler': scaler.state_dict(),
        'steps': state.get('steps', 0),
        'timestamp': state.get('timestamp', datetime.now().timestamp()),
        'best_perf': state.get('best_perf', {'avg_rank': 4.0, 'avg_pt': -135.0}),
        'config': {
            'control': {
                'version': version,
                'online': False,
            },
            'resnet': {
                'conv_channels': conv_channels,
                'num_blocks': num_blocks,
            },
        },
    }

    print(f"\nSaving full checkpoint to {output_path} ...")
    torch.save(full_state, output_path)
    print("Done! ✓")
    print(f"\nYou can now run: python train.py")

if __name__ == '__main__':
    main()
