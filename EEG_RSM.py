# -*- coding: utf-8 -*-
"""
Group-level EEG RSM (compute + visualize) —— 计算与可视化彻底分离
"""
import os
import glob
import pathlib
import json
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage import exposure
from sklearn.preprocessing import minmax_scale

# ---------------- 路径常量 ---------------- #
ROOT_DIR   = r'E:\EEG-RSA\eeglab\EFRP_O'
PIC_LIST   = 'all_pics.txt'                       # 纯图片名清单
SAVE_DIR   = r'E:\EEG-RSA\RSM'
SAVE_CSV   = os.path.join(SAVE_DIR, 'rsm_EEG_O.csv')
SAVE_IMG   = os.path.join(SAVE_DIR, 'rsm_EEG_O.png')
os.makedirs(SAVE_DIR, exist_ok=True)

with open(PIC_LIST) as f:
    PICS = [l.strip() for l in f if l.strip()]
ORDER = [f"{p}_target" for p in PICS] + [f"{p}_background" for p in PICS]
N     = len(ORDER)

# -------------------------------------------------
# 1. 计算部分：增量式 Fisher-Z 平均，返回干净 RSM
# -------------------------------------------------
def compute_group_rsm() -> Tuple[pd.DataFrame, List[str]]:
    """纯计算，返回 (rsm_clean_DataFrame, good_pics_list)"""
    sum_z = np.zeros((N, N), dtype=np.float64)
    count = np.zeros((N, N), dtype=np.int32)

    def _process_single(csv_path: str):
        df = pd.read_csv(csv_path)
        df = df[df['pic_name'].isin(PICS)].copy()
        df['condition'] = df['marker'].map({1: 'target', 0: 'background'})
        time_cols = [c for c in df.columns if c.startswith('t')]
        vec_map = (df.groupby(['pic_name', 'condition'])[time_cols]
                     .mean()
                     .apply(lambda x: x.values, axis=1).to_dict())

        T = len(time_cols)
        X, missing = [], []
        for pic in PICS:
            for cond in ('target', 'background'):
                key = (pic, cond)
                X.append(vec_map[key] if key in vec_map else np.zeros(T))
                if key not in vec_map:
                    missing.append(f"{pic}_{cond}")
        X = np.array(X)            # (926, T)
        rsm = cosine_similarity(X) # (926, 926)
        z   = np.arctanh(np.clip(rsm, -0.999999, 0.999999))
        return z, missing

    csv_files = sorted(glob.glob(os.path.join(ROOT_DIR, '**', '*_grandERP.csv'), recursive=True))
    if not csv_files:
        raise FileNotFoundError('未找到任何 *_grandERP.csv')

    missing_log = {}
    for csv in csv_files:
        sid = pathlib.Path(csv).stem.replace('_grandERP', '')
        z, miss = _process_single(csv)
        sum_z += z
        count += (z != 0.0)
        if miss:
            missing_log[sid] = miss

    # 平均并 Fisher-Z 逆变换
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_z = np.true_divide(sum_z, count)
        mean_z[count == 0] = 0.0
    rsm_group = np.tanh(mean_z)

    rsm_group_df = pd.DataFrame(rsm_group, index=ORDER, columns=ORDER)
    count_df     = pd.DataFrame(count, index=ORDER, columns=ORDER)

    # 后置剔除缺失整图
    bad_pics = []
    for pic in PICS:
        tgt_key, bkg_key = f"{pic}_target", f"{pic}_background"
        if (count_df.loc[tgt_key, :].sum() == 0) or (count_df.loc[bkg_key, :].sum() == 0):
            bad_pics.append(pic)
    good_pics  = [p for p in PICS if p not in bad_pics]
    good_order = [f"{p}_target" for p in good_pics] + [f"{p}_background" for p in good_pics]
    rsm_clean  = rsm_group_df.loc[good_order, good_order]

    print(f'后置剔除后保留完整图片数：{len(good_pics)}（已踢 {len(bad_pics)} 幅）')
    return rsm_clean, good_pics

# -------------------------------------------------
# 2. 可视化部分：只依赖 rsm_clean
# -------------------------------------------------
def plot_group_rsm(rsm_clean: pd.DataFrame, out_img: str):
    """绝对值 + 非对角线拉伸到 [0, 0.8] + 保留对角线显示"""
    rsm_abs = np.abs(rsm_clean.values)

    # 非对角线值复制一份用于拉伸
    off_diag = rsm_abs[~np.eye(rsm_abs.shape[0], dtype=bool)]

    # 线性拉伸到 [0, 0.8]
    off_diag_min, off_diag_max = off_diag.min(), off_diag.max()
    off_diag_scaled = 1 * (off_diag - off_diag_min) / (off_diag_max - off_diag_min)

    # 将拉伸后的值填回去
    rsm_vis = rsm_abs.copy()
    rsm_vis[~np.eye(rsm_abs.shape[0], dtype=bool)] = off_diag_scaled

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
    plt.title('Group RSM (Scaled Absolute Value)')
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    plt.show()
# -------------------------------------------------
# 3. 主控：自由开关
# -------------------------------------------------
def main(do_compute: bool = True, do_plot: bool = True):
    if do_compute:
        rsm_clean, good_pics = compute_group_rsm()
        # 落盘
        rsm_clean.to_csv(SAVE_CSV)
        with open(os.path.join(SAVE_DIR, 'kept_pics_clean.txt'), 'w') as f:
            for pic in good_pics:
                f.write(pic + '\n')
        print(f'已保存计算结果 -> {SAVE_CSV}')
    else:
        # 直接读盘
        rsm_clean = pd.read_csv(SAVE_CSV, index_col=0)

    if do_plot:
        plot_group_rsm(rsm_clean, SAVE_IMG)
        print(f'已保存可视化结果 -> {SAVE_IMG}')

# -------------------------------------------------
# 4. 入口
# -------------------------------------------------
if __name__ == '__main__':
    # 例：只想画图，把 compute 关掉
    # main(do_compute=False, do_plot=True)
    main(do_compute=True, do_plot=True)