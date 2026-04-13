#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 CNN 特征构建 2N×2N RSM（余弦相似度）
输入：长表 (pic_name | condition | t0 | t1 …)
顺序：先 target 后 background
色带：蓝→黄，与 EEG_RSM.py 保持一致
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics.pairwise import cosine_similarity   # 新增

# ========== 用户区 ==========
CSV_FILE = Path(r'E:\EEG-RSA\retinanethva_P3_feat.csv')  # 特征  hva_class_feat_6.csv  yoloHva_feat_4
KEPT_TXT = Path(r'E:\EEG-RSA\kept_pics_clean.txt')                  # 保留图片
OUT_CSV  = Path(r'E:\EEG-RSA\RSM\rsm_retinanethva_3.csv')                        # 输出 RSM
OUT_PNG  = Path(r'E:\EEG-RSA\RSM\rsm_retinanethva_3.png')                        # 热力图
# ============================

def load_ordered(csv_file, kept_txt):
    """先 target 后 background，按 pic_name 排序，并过滤 kept 列表"""
    with open(kept_txt, 'r') as f:
        kept_stems = {Path(line.strip()).stem.lower() for line in f if line.strip()}

    df = pd.read_csv(csv_file)
    df['stem'] = df['pic_name'].str.lower().apply(lambda x: Path(x).stem)
    df = df[df['stem'].isin(kept_stems)]
    if df.empty:
        raise ValueError('❌ 过滤后 0 条，检查 pic_name 与 kept_pics.txt 是否匹配！')

    tgt = df[df['condition'] == 'target'].sort_values('pic_name')
    bkg = df[df['condition'] == 'background'].sort_values('pic_name')

    # ---------- 缺图侦探 ----------
    csv_stems = set(df['stem'])
    missing_in_csv = kept_stems - csv_stems  # kept 有但 CSV 没有
    redundant_in_csv = csv_stems - kept_stems  # CSV 有但 kept 没有
    matched = kept_stems & csv_stems  # 真正匹配

    print(f'【匹配结果】 kept:{len(kept_stems)}  csv:{len(csv_stems)}  匹配:{len(matched)}')
    if missing_in_csv:
        print(f'【缺图】kept_pics 有但 CSV 无（共 {len(missing_in_csv)} 张）:')
        for pic in sorted(missing_in_csv):
            print(pic)
    if redundant_in_csv:
        print(f'【多余】CSV 有但 kept_pics 无（共 {len(redundant_in_csv)} 张）:')
        for pic in sorted(redundant_in_csv):
            print(pic)
    # ---------- 侦探结束 ----------
    return pd.concat([tgt, bkg], ignore_index=True)

def main():
    df = load_ordered(CSV_FILE, KEPT_TXT)
    time_cols = [c for c in df.columns if c.startswith('t')]
    X = df[time_cols].values.astype(float)
    labels = df['pic_name'] + '_' + df['condition']

    # 原始余弦相似度
    rsm = cosine_similarity(X)          # [-1,1]
    rsm_abs = np.abs(rsm)               # 第一步：绝对值

    # 第二步：非对角线拉伸到 [0, 0.8]
    off_diag_mask = ~np.eye(rsm_abs.shape[0], dtype=bool)
    off_diag = rsm_abs[off_diag_mask]
    off_diag_min, off_diag_max = off_diag.min(), off_diag.max()
    off_diag_scaled = 0.8 * (off_diag - off_diag_min) / (off_diag_max - off_diag_min)

    # 填回
    rsm_vis = rsm_abs.copy()
    rsm_vis[off_diag_mask] = off_diag_scaled

    # 低饱和度蓝→黄→红
    cmap_soft = LinearSegmentedColormap.from_list(
        'soft_byr',
        [(0.00, '#6c7ee9'),   # 低饱和蓝
         (0.50, '#f5d760'),   # 低饱和黄
         (1.00, '#e06c6c')]   # 低饱和红
    )

    plt.figure(figsize=(10, 9))
    sns.heatmap(
        rsm_vis,
        cmap=cmap_soft,
        vmin=0, vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Scaled |Cosine Similarity|'}
    )
    plt.title('RSM: CNN whole-feature (|cosine|, scaled 0→0.8)')
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.show()

    # 保存原始余弦矩阵（CSV 不拉伸）
    pd.DataFrame(rsm, index=labels, columns=labels).to_csv(OUT_CSV)
    print(f'Cosine-based RSM 已保存 → {OUT_CSV}')
    print(f'热力图已保存 → {OUT_PNG}')

if __name__ == '__main__':
    main()