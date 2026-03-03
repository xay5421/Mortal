"""
自对弈在线 RL 训练一键启动脚本。

架构:
  ┌─────────┐  TCP :5000  ┌─────────┐  TCP :5000  ┌─────────┐
  │ client  │ ←─────────→ │ server  │ ←─────────→ │ trainer │
  │ (对弈)  │  get_param  │ (中继)  │  drain      │ (训练)  │
  │         │  submit_log │         │  submit_prm │ online  │
  └─────────┘             └─────────┘             └─────────┘

用法:
    # 一键启动（server + trainer + 1 client）
    python self_play.py

    # 指定配置文件
    MORTAL_CFG=config.selfplay.toml python self_play.py

    # 只启动 client（连接远程 server，可多开）
    python self_play.py --client-only

    # 只启动 server
    python self_play.py --server-only

    # 自定义 client 数量
    python self_play.py --num-clients 2

    # dry-run: 检查配置但不启动
    python self_play.py --dry-run

前提:
    1. mortal.pth（训练 checkpoint，可用 convert_weights_simple.py 转换）
    2. baseline.pth（对弈基准，可用 setup_training.py 生成）
    3. grp.pth（reward 模型，可用 train_grp.py 训练或 setup_training.py 生成随机版）
    4. config.toml 中 online = true
    5. CUDA GPU（CPU 可跑但极慢）

详见 README_TRAIN.md。
"""

import os
import sys
import time
import signal
import argparse
import logging
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path

# ─── Logging ───

LOG_FMT = '%(asctime)s %(levelname)-5s [%(name)s] %(message)s'
LOG_DATEFMT = '%H:%M:%S'

def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # console
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATEFMT))
    root.addHandler(ch)

    # file
    log_dir = Path('self_play_logs')
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(log_dir / f'self_play_{ts}.log', encoding='utf-8')
    fh.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATEFMT))
    root.addHandler(fh)

    return logging.getLogger('self_play')

# ─── Config validation ───

def load_and_validate_config(logger):
    """加载 config.toml 并验证自对弈所需的配置。"""
    import toml

    config_file = os.environ.get('MORTAL_CFG', 'config.toml')
    if not os.path.exists(config_file):
        logger.error(f'❌ 配置文件不存在: {config_file}')
        logger.info(f'   提示: 复制 config.train.toml 并修改 online = true')
        sys.exit(1)

    with open(config_file, encoding='utf-8') as f:
        config = toml.load(f)

    errors = []
    warnings = []

    # online 必须开启
    if not config.get('control', {}).get('online', False):
        errors.append('control.online 必须为 true')

    # 检查文件
    for key, desc in [
        ('control.state_file', '训练 checkpoint (mortal.pth)'),
        ('baseline.train.state_file', '对弈基准 (baseline.pth)'),
        ('baseline.test.state_file', '测试基准 (baseline.pth)'),
        ('grp.state_file', 'GRP 模型 (grp.pth)'),
    ]:
        parts = key.split('.')
        val = config
        for p in parts:
            val = val.get(p, {})
        if isinstance(val, str) and not os.path.exists(val):
            errors.append(f'{key} = {val!r} 文件不存在 ({desc})')

    # 检查 train_play 配置
    if 'train_play' not in config:
        errors.append('缺少 [train_play.default] 配置段（在线模式必需）')

    # 检查 online.server 配置
    server_cfg = config.get('online', {}).get('server', {})
    if not server_cfg.get('buffer_dir'):
        errors.append('缺少 online.server.buffer_dir')
    if not server_cfg.get('drain_dir'):
        errors.append('缺少 online.server.drain_dir')

    # 检查 pts
    pts = config.get('env', {}).get('pts', [])
    if len(pts) != 4:
        errors.append(f'env.pts 应为 4 个值，当前: {pts}')

    # 检查 freeze_bn（online 建议冻结）
    if not config.get('freeze_bn', {}).get('mortal', False):
        warnings.append('freeze_bn.mortal = false，在线训练建议设为 true')

    # 检查 device
    import torch
    device = config.get('control', {}).get('device', 'cpu')
    if device.startswith('cuda') and not torch.cuda.is_available():
        errors.append(f'device = {device!r} 但 CUDA 不可用')
    if device == 'cpu':
        warnings.append('device = cpu，训练会非常慢')

    return config, config_file, errors, warnings


def print_config_summary(config, logger):
    """打印关键配置摘要。"""
    ctrl = config.get('control', {})
    env = config.get('env', {})
    resnet = config.get('resnet', {})
    server = config.get('online', {}).get('server', {})
    remote = config.get('online', {}).get('remote', {})
    tp = config.get('train_play', {}).get('default', {})
    optim_cfg = config.get('optim', {})
    sched = optim_cfg.get('scheduler', {})

    logger.info('=' * 60)
    logger.info('自对弈训练配置摘要')
    logger.info('=' * 60)
    logger.info(f'  模型版本:    v{ctrl.get("version", "?")}')
    logger.info(f'  模型结构:    {resnet.get("conv_channels", "?")}ch / {resnet.get("num_blocks", "?")}blocks')
    logger.info(f'  设备:        {ctrl.get("device", "?")}')
    logger.info(f'  AMP:         {ctrl.get("enable_amp", False)}')
    logger.info(f'  Compile:     {ctrl.get("enable_compile", False)}')
    logger.info(f'  Batch size:  {ctrl.get("batch_size", "?")} × {ctrl.get("opt_step_every", 1)} = {ctrl.get("batch_size", 0) * ctrl.get("opt_step_every", 1)} effective')
    logger.info(f'  学习率:      {sched.get("peak", "?")} → {sched.get("final", "?")}')
    logger.info(f'  顺位分 pts:  {env.get("pts", "?")}')
    logger.info(f'  Freeze BN:   {config.get("freeze_bn", {}).get("mortal", False)}')
    logger.info(f'  ─── Server ───')
    logger.info(f'  地址:        {remote.get("host", "?")}:{remote.get("port", "?")}')
    logger.info(f'  Buffer容量:  {server.get("capacity", "?")}')
    logger.info(f'  Sequential:  {server.get("force_sequential", False)}')
    logger.info(f'  ─── Client (对弈) ───')
    logger.info(f'  每轮局数:    {tp.get("games", "?")}')
    logger.info(f'  探索率 ε:    {tp.get("boltzmann_epsilon", "?")}')
    logger.info(f'  探索温度:    {tp.get("boltzmann_temp", "?")}')
    logger.info(f'  Top-p:       {tp.get("top_p", "?")}')
    logger.info(f'  ─── Trainer ───')
    logger.info(f'  Save every:  {ctrl.get("save_every", "?")} steps')
    logger.info(f'  Test every:  {ctrl.get("test_every", "?")} steps')
    logger.info(f'  Checkpoint:  {ctrl.get("state_file", "?")}')
    logger.info('=' * 60)


# ─── Process management ───

class ProcessManager:
    """管理 server / trainer / client 子进程的生命周期。"""

    def __init__(self, logger):
        self.logger = logger
        self.processes = {}  # name → Popen
        self._lock = threading.Lock()
        self._shutting_down = False

    def start(self, name, cmd, env=None, prefix_color=None):
        """启动一个子进程，stdout/stderr 加前缀转发。"""
        full_env = {**os.environ, **(env or {})}
        self.logger.info(f'启动 [{name}]: {" ".join(cmd)}')

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=full_env,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        with self._lock:
            self.processes[name] = proc

        # 日志转发线程
        t = threading.Thread(
            target=self._pipe_output,
            args=(name, proc, prefix_color),
            daemon=True,
        )
        t.start()
        return proc

    def _pipe_output(self, name, proc, color):
        """将子进程输出加前缀打到 stderr。"""
        COLORS = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
        }
        RESET = '\033[0m'
        c = COLORS.get(color, '')
        r = RESET if c else ''
        prefix = f'{c}[{name}]{r} '

        for raw_line in proc.stdout:
            line = raw_line.decode('utf-8', errors='replace').rstrip('\n')
            sys.stderr.write(f'{prefix}{line}\n')
            sys.stderr.flush()

    def wait_any(self):
        """等待任一子进程退出，返回 (name, returncode)。"""
        while True:
            with self._lock:
                for name, proc in list(self.processes.items()):
                    ret = proc.poll()
                    if ret is not None:
                        return name, ret
            time.sleep(0.5)

    def shutdown(self, timeout=10):
        """优雅关闭所有子进程。"""
        if self._shutting_down:
            return
        self._shutting_down = True
        self.logger.info('正在关闭所有子进程...')

        with self._lock:
            procs = list(self.processes.items())

        # 先 SIGTERM
        for name, proc in procs:
            if proc.poll() is None:
                self.logger.info(f'  发送 SIGTERM → [{name}] (PID {proc.pid})')
                try:
                    proc.terminate()
                except OSError:
                    pass

        # 等待
        deadline = time.time() + timeout
        for name, proc in procs:
            remaining = max(0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
                self.logger.info(f'  [{name}] 已退出 (code={proc.returncode})')
            except subprocess.TimeoutExpired:
                self.logger.warning(f'  [{name}] 超时，SIGKILL')
                proc.kill()
                proc.wait()

    def is_alive(self, name):
        with self._lock:
            proc = self.processes.get(name)
            return proc is not None and proc.poll() is None


# ─── Main ───

def wait_for_server(host, port, timeout=30, logger=None):
    """等待 server 端口就绪。"""
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket() as s:
                s.settimeout(2)
                s.connect((host, port))
                if logger:
                    logger.info(f'Server 已就绪 ({host}:{port})')
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Mortal 自对弈在线 RL 训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--server-only', action='store_true',
                        help='只启动 server（中继服务器）')
    parser.add_argument('--client-only', action='store_true',
                        help='只启动 client（对弈器，可多开）')
    parser.add_argument('--num-clients', type=int, default=1,
                        help='启动几个 client 进程 (默认: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='只检查配置，不启动进程')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f'Mortal 自对弈训练 — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # ── 加载配置 ──
    config, config_file, errors, warnings = load_and_validate_config(logger)

    logger.info(f'配置文件: {config_file}')
    print_config_summary(config, logger)

    for w in warnings:
        logger.warning(f'⚠️  {w}')
    if errors:
        for e in errors:
            logger.error(f'❌ {e}')
        logger.error('配置校验失败，请修复后重试')
        sys.exit(1)
    else:
        logger.info('✅ 配置校验通过')

    if args.dry_run:
        logger.info('Dry-run 完成，退出')
        return

    # ── 准备 ──
    python = sys.executable
    mortal_cfg = os.environ.get('MORTAL_CFG', 'config.toml')
    env_base = {'MORTAL_CFG': mortal_cfg}
    remote = config['online']['remote']
    host, port = remote['host'], remote['port']

    pm = ProcessManager(logger)

    # 信号处理
    def handle_signal(sig, frame):
        logger.info(f'收到信号 {signal.Signals(sig).name}，准备退出...')
        pm.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── 确保所需目录存在 ──
    mortal_dir = os.path.dirname(os.path.abspath(__file__))
    for d in ['runs', 'test_play_logs', 'train_play_logs', 'self_play_logs',
              '1v3_logs', 'grp_runs', 'buffer', 'drain']:
        os.makedirs(os.path.join(mortal_dir, d), exist_ok=True)
    logger.info(f'工作目录: {mortal_dir}')

    try:
        # ── 启动 server ──
        if not args.client_only:
            pm.start('server', [python, 'server.py'], env=env_base, prefix_color='cyan')
            if not wait_for_server(host, port, timeout=15, logger=logger):
                logger.error(f'❌ Server 启动超时 ({host}:{port})')
                pm.shutdown()
                sys.exit(1)
        else:
            logger.info(f'Client-only 模式，连接 {host}:{port}')
            if not wait_for_server(host, port, timeout=5, logger=logger):
                logger.error(f'❌ 无法连接 server ({host}:{port})')
                sys.exit(1)

        # ── 启动 trainer ──
        if not args.client_only and not args.server_only:
            pm.start('trainer', [python, 'train.py'], env=env_base, prefix_color='green')
            # 等 trainer 提交初始参数
            time.sleep(3)

        # ── 启动 client(s) ──
        if not args.server_only:
            for i in range(args.num_clients):
                name = f'client-{i}' if args.num_clients > 1 else 'client'
                colors = ['yellow', 'magenta', 'blue', 'red']
                color = colors[i % len(colors)]
                pm.start(name, [python, 'client.py'], env=env_base, prefix_color=color)

        # ── 监控 ──
        logger.info('')
        logger.info('🎮 自对弈训练已启动！')
        logger.info(f'   Server:  PID {pm.processes.get("server", type("",(),{"pid":"N/A"})).pid}')
        if not args.client_only and not args.server_only:
            logger.info(f'   Trainer: PID {pm.processes.get("trainer", type("",(),{"pid":"N/A"})).pid}')
        for name in pm.processes:
            if name.startswith('client'):
                logger.info(f'   {name.capitalize()}: PID {pm.processes[name].pid}')
        logger.info('')
        logger.info('   Ctrl+C 优雅退出')
        logger.info('   TensorBoard: tensorboard --logdir runs --bind_all')
        logger.info('   日志目录: self_play_logs/')
        logger.info('')

        # 等任一进程退出
        trainer_restarts = 0
        max_trainer_restarts = 5
        while True:
            name, code = pm.wait_any()
            if name == 'trainer':
                if code == 0:
                    # train.py online 模式在 test_play 后会 sys.exit(0)
                    logger.info(f'[trainer] 正常退出 (test_play 完成，自动重启)')
                    trainer_restarts = 0  # 正常退出重置计数
                else:
                    trainer_restarts += 1
                    logger.warning(f'[trainer] 异常退出 (code={code})，第 {trainer_restarts}/{max_trainer_restarts} 次重启')
                    if trainer_restarts > max_trainer_restarts:
                        logger.error(f'[trainer] 连续异常退出超过 {max_trainer_restarts} 次，停止')
                        pm.shutdown()
                        sys.exit(1)
                time.sleep(3)
                # 重新确保目录存在（防止被清理）
                for d in ['runs', 'test_play_logs', 'train_play_logs']:
                    os.makedirs(os.path.join(mortal_dir, d), exist_ok=True)
                pm.start('trainer', [python, 'train.py'], env=env_base, prefix_color='green')
            else:
                logger.error(f'[{name}] 意外退出 (code={code})')
                logger.info('关闭其余进程...')
                pm.shutdown()
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt')
        pm.shutdown()
    except Exception:
        logger.exception('未捕获异常')
        pm.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()
