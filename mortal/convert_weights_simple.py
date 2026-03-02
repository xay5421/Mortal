"""
简化版权重转换：不依赖 libriichi，直接操作 state_dict。

用法:
    python convert_weights_simple.py <input.pth> [output.pth]
"""

import sys
import torch
from datetime import datetime

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.pth> [output.pth]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'mortal.pth'

    print(f"Loading {input_path} ...")
    state = torch.load(input_path, weights_only=True, map_location='cpu')

    print(f"Keys: {list(state.keys())}")

    # 已经是完整 checkpoint
    if all(k in state for k in ['mortal', 'current_dqn', 'optimizer', 'aux_net']):
        print("Already a full checkpoint, nothing to do.")
        return

    cfg = state.get('config', {})
    version = cfg.get('control', {}).get('version', 4)
    conv_channels = cfg.get('resnet', {}).get('conv_channels', 256)
    num_blocks = cfg.get('resnet', {}).get('num_blocks', 54)
    print(f"Model: version={version}, conv_channels={conv_channels}, num_blocks={num_blocks}")

    mortal_sd = state['mortal']
    dqn_sd = state['current_dqn']
    mortal_params = sum(p.numel() for p in mortal_sd.values())
    dqn_params = sum(p.numel() for p in dqn_sd.values())
    print(f"  mortal: {mortal_params:,} params")
    print(f"  dqn: {dqn_params:,} params")

    # aux_net: 如果没有就创建随机初始化的
    if 'aux_net' in state:
        aux_sd = state['aux_net']
        print(f"  aux_net: loaded from checkpoint")
    else:
        # AuxNet 就是一个 Linear(1024, 4, bias=False)
        aux_sd = {
            'net.weight': torch.randn(4, 1024) * 0.01
        }
        print(f"  aux_net: randomly initialized")

    full_state = {
        'mortal': mortal_sd,
        'current_dqn': dqn_sd,
        'aux_net': aux_sd,
        'optimizer': {'state': {}, 'param_groups': [{'lr': 3e-5, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.1, 'params': []}]},
        'scheduler': {'last_epoch': 0, '_step_count': 1, '_last_lr': [3e-5]},
        'scaler': {'scale': 65536.0, 'growth_factor': 2.0, 'backoff_factor': 0.5, 'growth_interval': 2000, '_growth_tracker': 0},
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

    print(f"\nSaving to {output_path} ...")
    torch.save(full_state, output_path)
    print("Done! ✓")

if __name__ == '__main__':
    main()
