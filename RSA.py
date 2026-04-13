#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量 RSA：EEG × frcnn 或 EEG × hva
参数：MODE / DO_PERM / n_perm
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pathlib

# ========== 用户参数 ========== #
MODE      = 'retinanethva'       # 'frcnn' 或 'hva' yolo8 yoloHva
DO_PERM   = False        # 是否做置换检验
n_perm    = 5000          # 置换次数（仅当 DO_PERM=True 生效）
# ============================= #

root    = pathlib.Path(r'./RSM')
out_dir = pathlib.Path(r'./RSA')
out_dir.mkdir(exist_ok=True)

eeg_tags   = ['F', 'P', 'T', 'O']  #'F', 'P', 'T', 'O'   'all'
model_ids  = [3, 4, 5]        # 层号   yolo  4, 6, 9 15, 18, 21  frcnn 4, 5, 6

# # ---------- 工具 ---------- #
# def load_rsm(path):
#     return pd.read_csv(path, index_col=0).values

def load_rsm(path):
    df = pd.read_csv(path, index_col=0)
    # 逐向量 Min-Max → [0,1]（非负，不保方向）
    df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8), axis=1)
    return df.values

# def load_rsm(path):
#     df = pd.read_csv(path, index_col=0)
#     # 把对角线置 NaN，逐行 Z-score 时自动忽略
#     np.fill_diagonal(df.values, np.nan)
#     df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
#     # 把 NaN 填回 0（不影响相似度计算，只避免后续报错）
#     df = df.fillna(0.0)
#     return df.values

# ✅ 向量化加速版置换检验
def rsa_vec(rsm1, rsm2):
    mask = np.triu_indices_from(rsm1, k=1)
    v1, v2 = rsm1[mask], rsm2[mask]

    v1_ranked = np.argsort(np.argsort(v1))
    v2_ranked = np.argsort(np.argsort(v2))

    r, _ = pearsonr(v1_ranked, v2_ranked)

    if DO_PERM:
        rng = np.random.default_rng()
        idx = np.arange(len(v1_ranked))
        perm_matrix = np.array([rng.permutation(idx) for _ in range(n_perm)])
        v1_perm = v1_ranked[perm_matrix]

        v2_centered = v2_ranked - v2_ranked.mean()
        v1_centered = v1_perm - v1_perm.mean(axis=1, keepdims=True)
        numer = (v1_centered * v2_centered).sum(axis=1)
        denom = np.sqrt((v1_centered ** 2).sum(axis=1) * (v2_centered ** 2).sum())
        perm_r = numer / denom

        p = (np.abs(perm_r) >= np.abs(r)).mean()
    else:
        _, p = pearsonr(v1_ranked, v2_ranked)

    return r, p

# ---------- 主计算 ---------- #
model_name = MODE                   # 'frcnn' or 'hva'
rsa_mat = pd.DataFrame(index=eeg_tags, columns=model_ids, dtype=float)
p_mat   = rsa_mat.copy()

for eeg in eeg_tags:
    for mid in model_ids:
        eeg_path   = root / f'rsm_EEG_{eeg}.csv'
        model_path = root / f'rsm_{model_name}_{mid}.csv'
        r, p = rsa_vec(load_rsm(eeg_path), load_rsm(model_path))
        rsa_mat.loc[eeg, mid] = r
        p_mat.loc[eeg, mid]   = p
        print(r,p)

# ---------- 热力图 ---------- #
annot = rsa_mat.round(3).astype(str)
#if DO_PERM:
annot += np.where(p_mat < 0.001, '***',
          np.where(p_mat < 0.01, '**',
          np.where(p_mat < 0.05, '*', '')))

plt.figure(figsize=(4, 3))
sns.heatmap(rsa_mat, annot=annot, fmt='', cmap='coolwarm', vmin=0, vmax=0.03,
            cbar_kws={'label': 'Spearman r'})
#plt.title(f'EEG × {model_name.upper()} RSA' + (' (perm)' if DO_PERM else ''))
plt.tight_layout()
out_png = out_dir / f'heatmap_all_{model_name}{"_perm" if DO_PERM else ""}.png'
plt.savefig(out_png, dpi=300)
plt.show()

# ---------- 导出 CSV ---------- #
rsa_mat.to_csv(out_dir / f'rsa_all_{model_name}.csv')
p_mat.to_csv(out_dir / f'p_all_{model_name}.csv')
print(f'完成 → {out_png}')