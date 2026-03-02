"""
天凤牌谱下载+转换工具

从天凤鳳凰卓(scc)下载牌谱，转换为 mjai JSON 格式供 Mortal 训练。

用法:
    # 下载最近 7 天的鳳凰卓四人南牌谱
    python download_tenhou.py

    # 下载指定年份的历史牌谱
    python download_tenhou.py --year 2024

    # 指定输出目录
    python download_tenhou.py --output data/tenhou

    # 限制下载数量（调试用）
    python download_tenhou.py --limit 10

依赖: pip install requests tqdm
"""

import argparse
import gzip
import io
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("需要安装依赖: pip install requests tqdm")
    sys.exit(1)


# ============================================================
# 天凤 mjlog XML → mjai JSON 转换器
# ============================================================

# 牌 ID 映射: 天凤用 0-135 (136张牌), mjai 用字符串
TILE_NAMES = []
for suit, name in [('m', '万'), ('p', '筒'), ('s', '索')]:
    for num in range(1, 10):
        TILE_NAMES.append(f'{num}{suit}')
# 字牌: 東南西北白發中
for wind in ['E', 'S', 'W', 'N', 'P', 'F', 'C']:
    TILE_NAMES.append(f'{wind}')


def tenhou_tile_to_mjai(tile_id: int) -> str:
    """天凤牌ID (0-135) → mjai 牌名"""
    idx = tile_id // 4  # 0-33
    mod = tile_id % 4
    if idx < 27:
        suit = idx // 9  # 0=m, 1=p, 2=s
        num = idx % 9 + 1
        suit_char = 'mps'[suit]
        # 赤宝牌: 每种花色的第 5 张的第 0 号是赤 (id=16,52,88)
        if num == 5 and mod == 0:
            return f'5{suit_char}r'
        return f'{num}{suit_char}'
    else:
        # 字牌: 27=東,28=南,29=西,30=北,31=白,32=發,33=中
        wind_map = ['E', 'S', 'W', 'N', 'P', 'F', 'C']
        return wind_map[idx - 27]


def parse_seed(seed_str):
    """解析 INIT 标签的 seed 属性: 'round,honba,riichi_sticks,dice0,dice1,dora_indicator'"""
    parts = list(map(int, seed_str.split(',')))
    return {
        'round': parts[0],       # 0=東1, 1=東2, ..., 4=南1, ...
        'honba': parts[1],
        'kyotaku': parts[2],
        'dice0': parts[3],
        'dice1': parts[4],
        'dora_marker': parts[5],
    }


def parse_hai_list(hai_str):
    """解析手牌列表: '1,2,3,...' → [tile_id, ...]"""
    if not hai_str:
        return []
    return list(map(int, hai_str.split(',')))


def bakaze_from_round(round_num):
    """round_num → 場風牌名"""
    wind_idx = round_num // 4
    winds = ['E', 'S', 'W', 'N']
    return winds[min(wind_idx, 3)]


def parse_mentsu(m_val):
    """
    解析副露編碼。
    天凤的副露用一个 16 bit 整数编码，格式很复杂。
    返回 (type, tiles, trigger_tile, from_who_offset)
    """
    m = int(m_val)

    from_who = m & 3  # 0=自家,1=下家,2=对家,3=上家 (相对偏移)

    if m & (1 << 2):  # bit2: 吃
        t = (m >> 10) & 63
        r = t % 3
        t = t // 3
        base_kind = t // 7
        base_num = t % 7
        base = base_kind * 9 + base_num
        tiles = []
        for i in range(3):
            tile_offset = (m >> (3 + 2*i)) & 3
            tiles.append((base + i) * 4 + tile_offset)
        trigger = tiles[r]
        return 'chi', tiles, trigger, from_who

    elif m & (1 << 3):  # bit3: 碰
        t = (m >> 9) & 127
        r = t % 3
        t = t // 3
        base = t
        tiles = []
        unused = (m >> 5) & 3
        idx = 0
        for i in range(4):
            if i == unused:
                continue
            tiles.append(base * 4 + i)
            idx += 1
        trigger = tiles[r]
        return 'pon', tiles, trigger, from_who

    elif m & (1 << 4):  # bit4: 加杠
        t = (m >> 9) & 127
        r = t % 3
        t = t // 3
        base = t
        tiles = []
        for i in range(4):
            tiles.append(base * 4 + i)
        added = (m >> 5) & 3
        return 'kakan', tiles, tiles[added], from_who

    elif from_who == 0:  # 暗杠
        t = (m >> 8) & 255
        base = t // 4
        # r = t % 4  # not used
        tiles = [base * 4 + i for i in range(4)]
        return 'ankan', tiles, None, 0

    else:  # 大明杠
        t = (m >> 8) & 255
        base = t // 4
        r_tile = t % 4
        tiles = [base * 4 + i for i in range(4)]
        trigger = base * 4 + r_tile
        return 'daiminkan', tiles, trigger, from_who

    return None, [], None, 0


def mjlog_to_mjai(xml_content, game_id='unknown'):
    """
    将天凤 mjlog XML 转换为 mjai JSON events 列表。
    每局游戏返回一个 event 列表（每行一个 JSON）。
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return None

    # 提取玩家名
    un_tags = root.findall('.//UN')
    names = ['NoName'] * 4
    if un_tags:
        un = un_tags[0]
        for i in range(4):
            key = f'n{i}'
            if key in un.attrib:
                names[i] = unquote(un.attrib[key])

    events = []
    events.append({'type': 'start_game', 'names': names})

    cur_scores = [25000, 25000, 25000, 25000]
    oya = 0

    for elem in root.iter():
        tag = elem.tag

        if tag == 'GO':
            # 游戏类型信息，跳过
            pass

        elif tag == 'INIT':
            seed = parse_seed(elem.attrib['seed'])
            cur_round = seed['round']
            oya = int(elem.attrib['oya'])

            # 手牌
            tehais = [[], [], [], []]
            for i in range(4):
                key = f'hai{i}'
                if key in elem.attrib:
                    tehais[i] = [tenhou_tile_to_mjai(t) for t in parse_hai_list(elem.attrib[key])]
                    # 补齐到 13 张
                    while len(tehais[i]) < 13:
                        tehais[i].append('?')

            # 得分
            if 'ten' in elem.attrib:
                cur_scores = [int(x) * 100 for x in elem.attrib['ten'].split(',')]

            bakaze = bakaze_from_round(cur_round)
            kyoku = (cur_round % 4) + 1
            dora_marker = tenhou_tile_to_mjai(seed['dora_marker'])

            events.append({
                'type': 'start_kyoku',
                'bakaze': bakaze,
                'kyoku': kyoku,
                'honba': seed['honba'],
                'kyotaku': seed['kyotaku'],
                'oya': oya,
                'scores': cur_scores[:],
                'dora_marker': dora_marker,
                'tehais': tehais,
            })

        # 摸牌: T=0家, U=1家, V=2家, W=3家
        elif tag in ('T', 'U', 'V', 'W') and len(tag) == 1:
            # 这些是大写字母后跟数字的标签，但 XML 解析为属性
            # 实际上天凤用 <T0/> <T1/> ... <D0/> <D1/> 等标签
            pass

        elif tag == 'DORA':
            if 'hai' in elem.attrib:
                dora = tenhou_tile_to_mjai(int(elem.attrib['hai']))
                events.append({'type': 'dora', 'dora_marker': dora})

        elif tag == 'REACH':
            who = int(elem.attrib['who'])
            step = int(elem.attrib.get('step', 1))
            if step == 1:
                events.append({'type': 'reach', 'actor': who})
            elif step == 2:
                events.append({'type': 'reach_accepted', 'actor': who})

        elif tag == 'N':
            who = int(elem.attrib['who'])
            m = int(elem.attrib['m'])
            ntype, tiles, trigger, from_offset = parse_mentsu(m)

            if ntype == 'chi':
                consumed = [tenhou_tile_to_mjai(t) for t in tiles if t != trigger][:2]
                target = (who - from_offset) % 4
                events.append({
                    'type': 'chi',
                    'actor': who,
                    'target': target,
                    'pai': tenhou_tile_to_mjai(trigger),
                    'consumed': consumed,
                })
            elif ntype == 'pon':
                consumed = [tenhou_tile_to_mjai(t) for t in tiles if t != trigger][:2]
                target = (who - from_offset) % 4
                events.append({
                    'type': 'pon',
                    'actor': who,
                    'target': target,
                    'pai': tenhou_tile_to_mjai(trigger),
                    'consumed': consumed,
                })
            elif ntype == 'daiminkan':
                consumed = [tenhou_tile_to_mjai(t) for t in tiles if t != trigger][:3]
                target = (who - from_offset) % 4
                events.append({
                    'type': 'daiminkan',
                    'actor': who,
                    'target': target,
                    'pai': tenhou_tile_to_mjai(trigger),
                    'consumed': consumed,
                })
            elif ntype == 'kakan':
                events.append({
                    'type': 'kakan',
                    'actor': who,
                    'pai': tenhou_tile_to_mjai(trigger),
                    'consumed': [tenhou_tile_to_mjai(t) for t in tiles if t != trigger][:3],
                })
            elif ntype == 'ankan':
                events.append({
                    'type': 'ankan',
                    'actor': who,
                    'consumed': [tenhou_tile_to_mjai(t) for t in tiles],
                })

        elif tag == 'AGARI':
            who = int(elem.attrib['who'])
            from_who = int(elem.attrib['fromWho'])
            deltas = None
            if 'sc' in elem.attrib:
                sc = list(map(int, elem.attrib['sc'].split(',')))
                # sc 交替为 score, delta: s0,d0,s1,d1,...
                deltas = [sc[i*2+1] * 100 for i in range(4)]
                cur_scores = [(sc[i*2] + sc[i*2+1]) * 100 for i in range(4)]

            event = {
                'type': 'hora',
                'actor': who,
                'target': from_who,
            }
            if deltas:
                event['deltas'] = deltas

            events.append(event)
            events.append({'type': 'end_kyoku'})

        elif tag == 'RYUUKYOKU':
            deltas = None
            if 'sc' in elem.attrib:
                sc = list(map(int, elem.attrib['sc'].split(',')))
                deltas = [sc[i*2+1] * 100 for i in range(4)]
                cur_scores = [(sc[i*2] + sc[i*2+1]) * 100 for i in range(4)]

            event = {'type': 'ryukyoku'}
            if deltas:
                event['deltas'] = deltas
            events.append(event)
            events.append({'type': 'end_kyoku'})

    # 处理摸牌和出牌（天凤用特殊标签名 T0-T135, D0-D135 等）
    # 由于 XML 解析会把 <T23/> 当作标签名 "T23"，需要特殊处理
    # 重新用正则解析原始 XML
    events_with_draws = process_draw_discard(xml_content, names)
    if events_with_draws:
        return events_with_draws

    events.append({'type': 'end_game'})
    return events


def process_draw_discard(xml_content, names):
    """
    用正则从原始 XML 中提取所有事件（包括摸牌/出牌标签）。
    天凤的摸牌/出牌用 <T0/>...<T135/> (0家摸) <D0/>...<D135/> (0家打) 等标签。
    """
    events = []
    events.append({'type': 'start_game', 'names': names})

    cur_scores = [25000, 25000, 25000, 25000]

    # 匹配所有 XML 标签
    tag_pattern = re.compile(r'<(\w+)([^/>]*)/?>(?:</\w+>)?')

    for match in tag_pattern.finditer(xml_content):
        tag = match.group(1)
        attrs_str = match.group(2)

        # 解析属性
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', attrs_str))

        # 摸牌: [TUVW]\d+
        draw_match = re.match(r'^([TUVW])(\d+)$', tag)
        if draw_match:
            player = 'TUVW'.index(draw_match.group(1))
            tile_id = int(draw_match.group(2))
            events.append({
                'type': 'tsumo',
                'actor': player,
                'pai': tenhou_tile_to_mjai(tile_id),
            })
            continue

        # 出牌: [defg]\d+ (小写)
        discard_match = re.match(r'^([DEFGdefg])(\d+)$', tag)
        if discard_match:
            letter = discard_match.group(1).upper()
            player = 'DEFG'.index(letter)
            tile_id = int(discard_match.group(2))
            tsumogiri = discard_match.group(1).islower()
            events.append({
                'type': 'dahai',
                'actor': player,
                'pai': tenhou_tile_to_mjai(tile_id),
                'tsumogiri': tsumogiri,
            })
            continue

        # INIT
        if tag == 'INIT':
            seed = parse_seed(attrs['seed'])
            oya = int(attrs['oya'])
            tehais = [[], [], [], []]
            for i in range(4):
                key = f'hai{i}'
                if key in attrs:
                    tehais[i] = [tenhou_tile_to_mjai(t) for t in parse_hai_list(attrs[key])]
                    while len(tehais[i]) < 13:
                        tehais[i].append('?')

            if 'ten' in attrs:
                cur_scores = [int(x) * 100 for x in attrs['ten'].split(',')]

            bakaze = bakaze_from_round(seed['round'])
            kyoku = (seed['round'] % 4) + 1
            dora_marker = tenhou_tile_to_mjai(seed['dora_marker'])

            events.append({
                'type': 'start_kyoku',
                'bakaze': bakaze,
                'kyoku': kyoku,
                'honba': seed['honba'],
                'kyotaku': seed['kyotaku'],
                'oya': oya,
                'scores': cur_scores[:],
                'dora_marker': dora_marker,
                'tehais': tehais,
            })

        elif tag == 'DORA':
            if 'hai' in attrs:
                events.append({
                    'type': 'dora',
                    'dora_marker': tenhou_tile_to_mjai(int(attrs['hai'])),
                })

        elif tag == 'REACH':
            who = int(attrs['who'])
            step = int(attrs.get('step', '1'))
            if step == 1:
                events.append({'type': 'reach', 'actor': who})
            elif step == 2:
                events.append({'type': 'reach_accepted', 'actor': who})

        elif tag == 'N':
            who = int(attrs['who'])
            m = int(attrs['m'])
            ntype, tiles, trigger, from_offset = parse_mentsu(m)

            if ntype == 'chi':
                consumed = [tenhou_tile_to_mjai(t) for t in tiles if t != trigger][:2]
                target = (who - from_offset) % 4
                events.append({
                    'type': 'chi',
                    'actor': who,
                    'target': target,
                    'pai': tenhou_tile_to_mjai(trigger),
                    'consumed': consumed,
                })
            elif ntype == 'pon':
                consumed = [tenhou_tile_to_mjai(t) for t in tiles if t != trigger][:2]
                target = (who - from_offset) % 4
                events.append({
                    'type': 'pon',
                    'actor': who,
                    'target': target,
                    'pai': tenhou_tile_to_mjai(trigger),
                    'consumed': consumed,
                })
            elif ntype == 'daiminkan':
                consumed = [tenhou_tile_to_mjai(t) for t in tiles if t != trigger][:3]
                target = (who - from_offset) % 4
                events.append({
                    'type': 'daiminkan',
                    'actor': who,
                    'target': target,
                    'pai': tenhou_tile_to_mjai(trigger),
                    'consumed': consumed,
                })
            elif ntype == 'kakan':
                events.append({
                    'type': 'kakan',
                    'actor': who,
                    'pai': tenhou_tile_to_mjai(trigger),
                    'consumed': [tenhou_tile_to_mjai(t) for t in tiles if t != trigger][:3],
                })
            elif ntype == 'ankan':
                events.append({
                    'type': 'ankan',
                    'actor': who,
                    'consumed': [tenhou_tile_to_mjai(t) for t in tiles],
                })

        elif tag == 'AGARI':
            who = int(attrs['who'])
            from_who = int(attrs['fromWho'])
            deltas = None
            if 'sc' in attrs:
                sc = list(map(int, attrs['sc'].split(',')))
                deltas = [sc[i*2+1] * 100 for i in range(4)]
                cur_scores = [(sc[i*2] + sc[i*2+1]) * 100 for i in range(4)]
            event = {'type': 'hora', 'actor': who, 'target': from_who}
            if deltas:
                event['deltas'] = deltas
            events.append(event)
            events.append({'type': 'end_kyoku'})

        elif tag == 'RYUUKYOKU':
            deltas = None
            if 'sc' in attrs:
                sc = list(map(int, attrs['sc'].split(',')))
                deltas = [sc[i*2+1] * 100 for i in range(4)]
                cur_scores = [(sc[i*2] + sc[i*2+1]) * 100 for i in range(4)]
            event = {'type': 'ryukyoku'}
            if deltas:
                event['deltas'] = deltas
            events.append(event)
            events.append({'type': 'end_kyoku'})

    events.append({'type': 'end_game'})
    return events


# ============================================================
# 天凤牌谱下载
# ============================================================

BASE_URL = 'https://tenhou.net/sc/raw/dat/'
LOG_URL = 'https://tenhou.net/0/log/?'
LIST_URL = 'https://tenhou.net/sc/raw/list.cgi'

SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': '*/*',
    'Referer': 'https://tenhou.net/',
})


def get_scc_file_list(old=False):
    """获取鳳凰卓牌谱文件列表（最近7天）"""
    url = LIST_URL + ('?old' if old else '')
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    # 解析 JS: list([{file:'scc...',size:1234}, ...])
    matches = re.findall(r"\{file:'(scc[^']+)',size:(\d+)\}", resp.text)
    return [(name, int(size)) for name, size in matches]


def download_scc_index(filename):
    """下载 scc 索引文件，提取牌谱 log ID"""
    url = BASE_URL + filename
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    content = gzip.decompress(resp.content).decode('utf-8', errors='replace')

    # 提取 log ID: log=XXXX
    log_ids = re.findall(r'log=([a-f0-9\-gm]+)', content)

    # 只要四人南/四人东的 (gm-00a9 = 四人南喰赤, gm-00e1 = 四人東喰赤速)
    # 过滤三人(00b9)
    four_player_logs = [lid for lid in log_ids if 'gm-00a9' in lid or 'gm-00e1' in lid or 'gm-0029' in lid or 'gm-0061' in lid]

    return four_player_logs


def download_mjlog(log_id):
    """下载单个牌谱"""
    url = LOG_URL + log_id
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def download_year_archive(year):
    """下载年度打包文件"""
    url = f'https://tenhou.net/sc/raw/scraw{year}.zip'
    print(f"下载 {url} ...")
    resp = SESSION.get(url, timeout=300, stream=True)
    resp.raise_for_status()

    import zipfile
    zip_data = io.BytesIO(resp.content)
    with zipfile.ZipFile(zip_data) as zf:
        scc_files = [f for f in zf.namelist() if f.startswith('scc')]
        print(f"找到 {len(scc_files)} 个鳳凰卓索引文件")

        all_log_ids = []
        for fname in tqdm(scc_files, desc='解析索引'):
            with zf.open(fname) as f:
                content = gzip.decompress(f.read()).decode('utf-8', errors='replace')
                log_ids = re.findall(r'log=([a-f0-9\-gm]+)', content)
                four_player = [lid for lid in log_ids if 'gm-00a9' in lid or 'gm-00e1' in lid]
                all_log_ids.extend(four_player)

        return all_log_ids


def main():
    parser = argparse.ArgumentParser(description='天凤牌谱下载+转换工具')
    parser.add_argument('--year', type=int, help='下载指定年份的历史牌谱')
    parser.add_argument('--output', type=str, default='data', help='输出目录')
    parser.add_argument('--limit', type=int, default=0, help='限制下载数量（0=不限）')
    parser.add_argument('--delay', type=float, default=1.2, help='每次请求间隔秒数（天凤要求≥1秒）')
    parser.add_argument('--skip-existing', action='store_true', default=True, help='跳过已存在的文件')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取牌谱 ID 列表
    print("=" * 50)
    if args.year:
        print(f"模式: 下载 {args.year} 年历史牌谱")
        log_ids = download_year_archive(args.year)
        sub_dir = output_dir / str(args.year)
    else:
        print("模式: 下载最近 7 天鳳凰卓牌谱")
        scc_files = get_scc_file_list()
        print(f"找到 {len(scc_files)} 个索引文件")

        log_ids = []
        for fname, size in tqdm(scc_files, desc='解析索引'):
            ids = download_scc_index(fname)
            log_ids.extend(ids)
            time.sleep(0.5)  # 遵守天凤规则
        sub_dir = output_dir / 'recent'

    sub_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n找到 {len(log_ids)} 个四人牌谱")

    if args.limit > 0:
        log_ids = log_ids[:args.limit]
        print(f"限制为 {args.limit} 个")

    # 过滤已下载的
    if args.skip_existing:
        new_ids = []
        for lid in log_ids:
            safe_name = lid.replace('?', '_')
            out_path = sub_dir / f'{safe_name}.json.gz'
            if not out_path.exists():
                new_ids.append(lid)
        skipped = len(log_ids) - len(new_ids)
        if skipped > 0:
            print(f"跳过 {skipped} 个已存在的文件")
        log_ids = new_ids

    if not log_ids:
        print("没有新的牌谱需要下载")
        return

    # 下载并转换
    print(f"\n开始下载 {len(log_ids)} 个牌谱...")
    success = 0
    errors = 0

    for lid in tqdm(log_ids, desc='下载+转换'):
        try:
            mjlog = download_mjlog(lid)
            if not mjlog or '<mjloggm' not in mjlog:
                errors += 1
                continue

            events = process_draw_discard(
                mjlog,
                names=['P0', 'P1', 'P2', 'P3']  # 默认名字，后面从 UN 标签更新
            )

            # 从 mjlog 提取真实玩家名
            un_match = re.findall(r'n(\d)="([^"]*)"', mjlog)
            real_names = ['P0', 'P1', 'P2', 'P3']
            for idx_str, name in un_match:
                idx = int(idx_str)
                if 0 <= idx <= 3:
                    real_names[idx] = unquote(name)

            # 更新 start_game 的 names
            if events and events[0]['type'] == 'start_game':
                events[0]['names'] = real_names

            # 写入 gzip JSON (每行一个 event)
            safe_name = lid.replace('?', '_')
            out_path = sub_dir / f'{safe_name}.json.gz'
            json_lines = '\n'.join(json.dumps(e, ensure_ascii=False) for e in events)
            with gzip.open(out_path, 'wt', encoding='utf-8') as f:
                f.write(json_lines)

            success += 1

        except Exception as e:
            errors += 1
            tqdm.write(f"  错误 {lid}: {e}")

        time.sleep(args.delay)

    print(f"\n{'=' * 50}")
    print(f"完成! 成功: {success}, 失败: {errors}")
    print(f"输出目录: {sub_dir}")
    print(f"\n训练数据就绪后运行:")
    print(f"  python train.py")


if __name__ == '__main__':
    main()
