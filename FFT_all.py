# -*- coding: utf-8 -*-
"""
強化版 FFT ➜ 特徵擷取 (含去 DC、去趨勢、窗函數、零填補、Welch、動態門檻、能量 90% top_n)
"""

import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import get_window, detrend, welch
from scipy.stats import entropy

# ======== 使用者只改這兩行 ========
src = Path("C:\\Users\\User\\Desktop\\電輔車\\加速.xlsx")  # 支援 .csv / .xlsx
fs = 200                                              # 取樣頻率 Hz
# =================================

segment     = 256            # 每段樣本數 (2 的次方方便 FFT)
dc_thr      = 0.5            # DC 判定閾值
use_welch   = True           # True=Welch；False=單段 FFT
pad_factor  = 4              # 零填補倍數 (解析度 = fs / (segment*pad_factor))
energy_cut  = (0.25, 0.75, 0.90)   # 累積能量門檻 (low, mid, high)

out_root = Path.cwd() / "output_all" / src.stem
out_root.mkdir(parents=True, exist_ok=True)

# ---------- 1. 讀檔 ----------
if src.suffix.lower() == ".csv":
    df = pd.read_csv(src, encoding="utf-8")
else:
    df = pd.read_excel(src, engine="openpyxl")
print(f"📥 讀入 {len(df)} 筆資料：{src.name}")

axes = ['AccelX','AccelY','AccelZ','GyroX','GyroY','GyroZ']
required = ['Label'] + axes
miss = set(required) - set(df.columns)
if miss:
    raise ValueError(f"❌ 缺少欄位：{miss}")

# ---------- 2. 去 DC 標記 ----------
def dc_flags(df, thr=0.5):
    flags={}
    for a in axes:
        mu, std = df[a].mean(), df[a].std()
        flags[a] = (abs(mu)/std > thr) if std else True
    return flags

flags = dc_flags(df, dc_thr)

# ---------- 3. 分段處理 ----------
records = []
win = get_window('hamming', segment)
pad = segment * pad_factor

for start in range(0, len(df) // segment * segment, segment):
    seg = df.iloc[start:start+segment]
    label = seg['Label'].mode().iloc[0]

    for ax in axes:
        sig = seg[ax].values.astype(float)
        sig = detrend(sig)                   # 去趨勢
        if flags[ax]:
            sig -= sig.mean()                # 去 DC
        sig *= win                           # 加窗

        # --- 3a. 功率譜估計 ---
        if use_welch:
            f, Pxx = welch(
                sig, fs=fs, window='hamming',
                nperseg=segment, noverlap=segment//2,
                nfft=pad, detrend=False, return_onesided=True
            )
            yf = np.sqrt(Pxx)                # 幫你轉回幅度
            xf = f
        else:
            yf = np.abs(fft(sig, pad))[:pad//2]
            xf = fftfreq(pad, 1/fs)[:pad//2]

        records += [
            dict(label=label, axis=ax, frequency=freq, amplitude=amp)
            for freq, amp in zip(xf, yf)
        ]

fft_df = pd.DataFrame(records)
fft_df.to_csv(out_root / f"fft_{src.stem}.csv",
              index=False, encoding='utf-8-sig')
print(f"🔄 已產生 FFT 頻譜：{len(fft_df)} 行")

# ---------- 4. 動態門檻 (累積能量) ----------
# 先把功率 (幅度^2) 依頻率排序 (0 → Nyquist)
fft_df['power'] = fft_df['amplitude'] ** 2
fft_sorted = fft_df.sort_values('frequency')

cum_pwr = fft_sorted.groupby('frequency')['power'].sum().cumsum()
cum_pwr /= cum_pwr.iloc[-1]               # 0~1

def freq_at_ratio(r):
    return float(cum_pwr[cum_pwr >= r].index[0])

low_max  = round(freq_at_ratio(energy_cut[0]), 2)
mid_max  = round(freq_at_ratio(energy_cut[1]), 2)
high_min = round(freq_at_ratio(energy_cut[2]), 2)

print(f"🚀 自動門檻：low≤{low_max}Hz, mid≤{mid_max}Hz, high≥{high_min}Hz")

# ---------- 5. 特徵萃取 ----------
feature_rows=[]
for (lbl, ax), g in fft_df.groupby(['label','axis']):
    f = g['frequency'].values
    a = g['amplitude'].values
    pwr = g['power'].values

    if pwr.sum() == 0: continue
    peak_idx = pwr.argmax()
    energy = pwr.sum()

    # --- 自適應 top_n (能量 90%)
    sorted_idx = pwr.argsort()[::-1]
    top_idx=[]
    cum_e=0
    for idx in sorted_idx:
        top_idx.append(idx)
        cum_e += pwr[idx]
        if cum_e/energy >= 0.9:
            break

    row = dict(
        label      = lbl,
        axis       = ax,
        peak_freq  = f[peak_idx],
        peak_amp   = a[peak_idx],
        energy     = energy,
        centroid   = (f * pwr).sum() / energy,
        entropy    = entropy(pwr / energy),
        low_freq_energy  = pwr[f <= low_max].sum(),
        mid_freq_energy  = pwr[(f > low_max) & (f <= mid_max)].sum(),
        high_freq_ratio  = pwr[f >= high_min].sum() / energy
    )

    for i, idx in enumerate(top_idx, 1):
        row[f'top{i}_freq'] = f[idx]
        row[f'top{i}_amp']  = a[idx]
    feature_rows.append(row)

feat_df = pd.DataFrame(feature_rows)
feat_df.to_csv(out_root / f"features_{src.stem}.csv",
               index=False, encoding='utf-8-sig')
print(f"✅ 特徵完成：{len(feat_df)} 行；檔案已存至 features_{src.stem}.csv")

# ---------- 6. 寫門檻 JSON ----------
cfg = dict(
    generated_at = str(datetime.datetime.now()),
    sampling_rate = fs,
    segment = segment,
    pad_factor = pad_factor,
    window = "hamming",
    use_welch = use_welch,
    energy_cut = energy_cut,
    low_freq_max = low_max,
    mid_freq_max = mid_max,
    high_freq_min = high_min
)
with open(out_root / "band_cfg.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print("📑 band_cfg.json 已寫入 (方便之後重覆使用同一門檻)")

