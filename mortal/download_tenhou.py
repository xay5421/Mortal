"""
天凤牌谱下载+转换工具

从天凤鳳凰卓(scc)下载牌谱，转换为 mjai JSON 格式供 Mortal 训练。

用法:
    # 下载最近 7 天的鳳凰卓四人牌谱
    python download_tenhou.py

    # 下载指定年份的历史牌谱（先下载索引再逐个下载牌谱）
    python download_tenhou.py --year 2024

    # 指定输出目录
    python download_tenhou.py --output data/tenhou

    # 限制下载数量（调试用）
    python download_tenhou.py --limit 10

    # 本地 mjlog XML 文件批量转换
    python download_tenhou.py --convert-dir /path/to/mjlog/files

    # 从 houou-logs 的 SQLite 数据库导出并转换
    python download_tenhou.py --from-db db/2024.db

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

def tenhou_tile_to_mjai(tile_id: int) -> str:
    """天凤牌ID (0-135) → mjai 牌名"""
    idx = tile_id // 4
    mod = tile_id % 4
    if idx < 27:
        suit = idx // 9
        num = idx % 9 + 1
        suit_char = 'mps'[suit]
        if num == 5 and mod == 0:
            return f'5{suit_char}r'
        return f'{num}{suit_char}'
    else:
        wind_map = ['E', 'S', 'W', 'N', 'P', 'F', 'C']
        return wind_map[idx - 27]


def parse_seed(seed_str):
    parts = list(map(int, seed_str.split(',')))
    return {
        'round': parts[0],
        'honba': parts[1],
        'kyotaku': parts[2],
        'dora_marker': parts[5],
    }


def parse_hai_list(hai_str):
    if not hai_str:
        return []
    return list(map(int, hai_str.split(',')))


def bakaze_from_round(round_num):
    winds = ['E', 'S', 'W', 'N']
    return winds[min(round_num // 4, 3)]


def parse_mentsu(m_val):
    m = int(m_val)
    from_who = m & 3

    if m & (1 << 2):  # 吃
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

    elif m & (1 << 3):  # 碰
        t = (m >> 9) & 127
        r = t % 3
        t = t // 3
        unused = (m >> 5) & 3
        tiles = []
        for i in range(4):
            if i == unused:
                continue
            tiles.append(t * 4 + i)
        trigger = tiles[r]
        return 'pon', tiles, trigger, from_who

    elif m & (1 << 4):  # 加杠
        t = (m >> 9) & 127
        r = t % 3
        t = t // 3
        tiles = [t * 4 + i for i in range(4)]
        added = (m >> 5) & 3
        return 'kakan', tiles, tiles[added], from_who

    elif from_who == 0:  # 暗杠
        t = (m >> 8) & 255
        base = t // 4
        tiles = [base * 4 + i for i in range(4)]
        return 'ankan', tiles, None, 0

    else:  # 大明杠
        t = (m >> 8) & 255
        base = t // 4
        r_tile = t % 4
        tiles = [base * 4 + i for i in range(4)]
        trigger = base * 4 + r_tile
        return 'daiminkan', tiles, trigger, from_who


def mjlog_to_mjai(xml_content):
    """将天凤 mjlog XML 转换为 mjai JSON events 列表"""
    # 提取玩家名
    names = ['NoName'] * 4
    un_matches = re.findall(r'n(\d)="([^"]*)"', xml_content)
    for idx_str, name in un_matches:
        idx = int(idx_str)
        if 0 <= idx <= 3:
            names[idx] = unquote(name)

    events = [{'type': 'start_game', 'names': names}]
    cur_scores = [25000, 25000, 25000, 25000]

    tag_pattern = re.compile(r'<(\w+)([^/>]*)/?>(?:</\w+>)?')

    for match in tag_pattern.finditer(xml_content):
        tag = match.group(1)
        attrs_str = match.group(2)
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', attrs_str))

        # 摸牌
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

        # 出牌
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

            events.append({
                'type': 'start_kyoku',
                'bakaze': bakaze_from_round(seed['round']),
                'kyoku': (seed['round'] % 4) + 1,
                'honba': seed['honba'],
                'kyotaku': seed['kyotaku'],
                'oya': oya,
                'scores': cur_scores[:],
                'dora_marker': tenhou_tile_to_mjai(seed['dora_marker']),
                'tehais': tehais,
            })

        elif tag == 'DORA' and 'hai' in attrs:
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
            ntype, tiles, trigger, from_offset = parse_mentsu(int(attrs['m']))

            if ntype == 'chi':
                events.append({
                    'type': 'chi',
                    'actor': who,
                    'target': (who - from_offset) % 4,
                    'pai': tenhou_tile_to_mjai(trigger),
                    'consumed': [tenhou_tile_to_mjai(t) for t in tiles if t != trigger][:2],
                })
            elif ntype == 'pon':
                events.append({
                    'type': 'pon',
                    'actor': who,
                    'target': (who - from_offset) % 4,
                    'pai': tenhou_tile_to_mjai(trigger),
                    'consumed': [tenhou_tile_to_mjai(t) for t in tiles if t != trigger][:2],
                })
            elif ntype == 'daiminkan':
                events.append({
                    'type': 'daiminkan',
                    'actor': who,
                    'target': (who - from_offset) % 4,
                    'pai': tenhou_tile_to_mjai(trigger),
                    'consumed': [tenhou_tile_to_mjai(t) for t in tiles if t != trigger][:3],
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


def save_mjai(events, output_path):
    """将 events 保存为 gzip JSON"""
    json_lines = '\n'.join(json.dumps(e, ensure_ascii=False) for e in events)
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        f.write(json_lines)


def is_valid_mjai(events):
    """简单检查转换结果是否合理"""
    if len(events) < 10:
        return False
    types = set(e['type'] for e in events)
    return 'start_kyoku' in types and 'tsumo' in types and 'dahai' in types


# ============================================================
# 模式 1: 在线下载
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


def get_scc_file_list():
    """获取鳳凰卓牌谱文件列表（最近7天）"""
    resp = SESSION.get(LIST_URL, timeout=30)
    resp.raise_for_status()
    matches = re.findall(r"\{file:'(scc[^']+)',size:(\d+)\}", resp.text)
    return [(name, int(size)) for name, size in matches]


def download_scc_index(filename):
    """下载 scc 索引文件，提取四人牌谱 log ID"""
    url = BASE_URL + filename
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    content = gzip.decompress(resp.content).decode('utf-8', errors='replace')
    log_ids = re.findall(r'log=([a-f0-9\-gm]+)', content)
    # 四人: gm-00a9(四人南喰赤), gm-00e1(四人東喰赤速), gm-0029(四人南), gm-0061(四人東)
    return [lid for lid in log_ids
            if 'gm-00a9' in lid or 'gm-00e1' in lid
            or 'gm-0029' in lid or 'gm-0061' in lid]


def download_year_index(year):
    """从年度压缩包提取四人牌谱 ID 列表"""
    import zipfile

    zip_path = Path(f'scraw{year}.zip')
    if zip_path.exists():
        print(f"使用本地文件 {zip_path}")
        zip_data = open(zip_path, 'rb')
    else:
        url = f'https://tenhou.net/sc/raw/scraw{year}.zip'
        print(f"下载 {url} （可能需要几分钟）...")
        resp = SESSION.get(url, timeout=600, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        data = io.BytesIO()
        with tqdm(total=total, unit='B', unit_scale=True, desc=f'scraw{year}.zip') as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                data.write(chunk)
                pbar.update(len(chunk))
        data.seek(0)
        zip_data = data

    all_log_ids = []
    with zipfile.ZipFile(zip_data) as zf:
        scc_files = [f for f in zf.namelist() if 'scc' in f]
        print(f"找到 {len(scc_files)} 个鳳凰卓索引文件")
        for fname in tqdm(scc_files, desc='解析索引'):
            with zf.open(fname) as f:
                try:
                    content = gzip.decompress(f.read()).decode('utf-8', errors='replace')
                except Exception:
                    continue
                log_ids = re.findall(r'log=([a-f0-9\-gm]+)', content)
                four_player = [lid for lid in log_ids
                               if 'gm-00a9' in lid or 'gm-00e1' in lid
                               or 'gm-0029' in lid or 'gm-0061' in lid]
                all_log_ids.extend(four_player)

    if hasattr(zip_data, 'close'):
        zip_data.close()
    return all_log_ids


def run_download(args):
    """在线下载模式"""
    output_dir = Path(args.output)

    if args.year:
        print(f"模式: 下载 {args.year} 年历史牌谱")
        log_ids = download_year_index(args.year)
        sub_dir = output_dir / str(args.year)
    else:
        print("模式: 下载最近 7 天鳳凰卓牌谱")
        scc_files = get_scc_file_list()
        print(f"找到 {len(scc_files)} 个索引文件")
        log_ids = []
        for fname, size in tqdm(scc_files, desc='解析索引'):
            ids = download_scc_index(fname)
            log_ids.extend(ids)
            time.sleep(0.5)
        sub_dir = output_dir / 'recent'

    sub_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n找到 {len(log_ids)} 个四人牌谱")

    if args.limit > 0:
        log_ids = log_ids[:args.limit]
        print(f"限制为 {args.limit} 个")

    # 跳过已下载
    new_ids = []
    for lid in log_ids:
        safe_name = lid.replace('?', '_')
        if not (sub_dir / f'{safe_name}.json.gz').exists():
            new_ids.append(lid)
    skipped = len(log_ids) - len(new_ids)
    if skipped > 0:
        print(f"跳过 {skipped} 个已存在的文件")
    log_ids = new_ids

    if not log_ids:
        print("没有新的牌谱需要下载")
        return

    print(f"\n开始下载 {len(log_ids)} 个牌谱...")
    success = 0
    errors = 0

    for lid in tqdm(log_ids, desc='下载+转换'):
        try:
            resp = SESSION.get(LOG_URL + lid, timeout=30)
            resp.raise_for_status()
            mjlog = resp.content.decode('utf-8', errors='replace')
            if not mjlog or '<mjloggm' not in mjlog:
                errors += 1
                continue

            events = mjlog_to_mjai(mjlog)
            if not is_valid_mjai(events):
                errors += 1
                continue

            safe_name = lid.replace('?', '_')
            save_mjai(events, sub_dir / f'{safe_name}.json.gz')
            success += 1

        except Exception as e:
            errors += 1
            tqdm.write(f"  错误 {lid}: {e}")

        time.sleep(args.delay)

    print(f"\n完成! 成功: {success}, 失败: {errors}")
    print(f"输出目录: {sub_dir}")


# ============================================================
# 模式 2: 本地 mjlog 文件批量转换
# ============================================================

def run_convert_dir(args):
    """批量转换本地 mjlog XML 文件"""
    input_dir = Path(args.convert_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找所有 .xml 文件和无后缀的 mjlog 文件
    files = list(input_dir.rglob('*.xml'))
    files += list(input_dir.rglob('*.mjlog'))
    # 有些导出不带后缀
    for f in input_dir.rglob('*'):
        if f.is_file() and f.suffix not in ('.xml', '.mjlog', '.gz', '.json', '.db', '.zip'):
            try:
                head = f.read_bytes()[:20]
                if b'<mjloggm' in head:
                    files.append(f)
            except Exception:
                pass

    if not files:
        print(f"在 {input_dir} 下没有找到 mjlog 文件")
        return

    print(f"找到 {len(files)} 个 mjlog 文件")
    success = 0
    errors = 0

    for f in tqdm(files, desc='转换'):
        try:
            content = f.read_text(encoding='utf-8', errors='replace')
            if '<mjloggm' not in content:
                errors += 1
                continue

            events = mjlog_to_mjai(content)
            if not is_valid_mjai(events):
                errors += 1
                continue

            out_name = f.stem + '.json.gz'
            save_mjai(events, output_dir / out_name)
            success += 1

        except Exception as e:
            errors += 1
            tqdm.write(f"  错误 {f.name}: {e}")

    print(f"\n完成! 成功: {success}, 失败: {errors}")
    print(f"输出目录: {output_dir}")


# ============================================================
# 模式 3: 从 houou-logs SQLite 数据库导出
# ============================================================

def run_from_db(args):
    """从 houou-logs 的 SQLite 数据库读取并转换"""
    import sqlite3

    db_path = args.from_db
    if not os.path.exists(db_path):
        print(f"数据库文件不存在: {db_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # houou-logs 的表结构: logs(id, log_id, content, ...)
    # 先看看表结构
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"数据库表: {tables}")

    # 找包含牌谱内容的表
    content_table = None
    content_col = None
    id_col = None

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [(row[1], row[2]) for row in cursor.fetchall()]
        col_names = [c[0] for c in columns]

        for candidate in ['content', 'xml', 'log']:
            if candidate in col_names:
                content_table = table
                content_col = candidate
                id_col = 'log_id' if 'log_id' in col_names else 'id'
                break
        if content_table:
            break

    if not content_table:
        print("找不到包含牌谱内容的表。可能需要先用 houou-logs download 下载牌谱内容。")
        print(f"表结构:")
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            cols = cursor.fetchall()
            print(f"  {table}: {[c[1] for c in cols]}")
        conn.close()
        return

    # 统计数量
    cursor.execute(f"SELECT COUNT(*) FROM {content_table} WHERE {content_col} IS NOT NULL AND {content_col} != ''")
    total = cursor.fetchone()[0]
    print(f"找到 {total} 个有内容的牌谱")

    if args.limit > 0:
        total = min(total, args.limit)

    cursor.execute(f"SELECT {id_col}, {content_col} FROM {content_table} WHERE {content_col} IS NOT NULL AND {content_col} != '' LIMIT ?", (total,))

    success = 0
    errors = 0

    rows = cursor.fetchall()
    for log_id, content in tqdm(rows, desc='转换'):
        try:
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')

            if '<mjloggm' not in content:
                errors += 1
                continue

            # 过滤三人场
            go_match = re.search(r'<GO\s+type="(\d+)"', content)
            if go_match:
                game_type = int(go_match.group(1))
                if game_type & 0x10:  # 三人
                    continue

            events = mjlog_to_mjai(content)
            if not is_valid_mjai(events):
                errors += 1
                continue

            safe_name = str(log_id).replace('?', '_').replace('/', '_')
            save_mjai(events, output_dir / f'{safe_name}.json.gz')
            success += 1

        except Exception as e:
            errors += 1
            tqdm.write(f"  错误 {log_id}: {e}")

    conn.close()
    print(f"\n完成! 成功: {success}, 失败: {errors}")
    print(f"输出目录: {output_dir}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='天凤牌谱下载+转换工具 (Mortal 训练用)')
    parser.add_argument('--year', type=int, help='下载指定年份的历史牌谱')
    parser.add_argument('--output', type=str, default='data', help='输出目录 (默认: data)')
    parser.add_argument('--limit', type=int, default=0, help='限制处理数量 (0=不限)')
    parser.add_argument('--delay', type=float, default=1.2, help='下载请求间隔秒数 (默认: 1.2)')
    parser.add_argument('--convert-dir', type=str, help='本地 mjlog 文件目录，批量转换为 mjai')
    parser.add_argument('--from-db', type=str, help='从 houou-logs SQLite 数据库导出并转换')

    args = parser.parse_args()

    print("=" * 50)
    print("天凤牌谱 → mjai JSON 转换工具")
    print("=" * 50)

    if args.convert_dir:
        run_convert_dir(args)
    elif args.from_db:
        run_from_db(args)
    else:
        run_download(args)

    # 统计结果
    output_dir = Path(args.output)
    gz_files = list(output_dir.rglob('*.json.gz'))
    if gz_files:
        total_size = sum(f.stat().st_size for f in gz_files)
        print(f"\n📊 数据统计:")
        print(f"   文件数: {len(gz_files)}")
        print(f"   总大小: {total_size / 1024 / 1024:.1f} MB")
        print(f"\n🚀 下一步: python train.py")


if __name__ == '__main__':
    main()
