import pandas as pd
import numpy as np
from scipy.stats import entropy
import os

# === 步驟 1：讀取傅立葉轉換後的 CSV（輸入）===
input_path ="C:\\Users\\User\\py\\Bicycle_Center_E-Bikes\\0809output_fft\\顛簸(全)\\fft_顛簸(全).csv"  # ← 換成你的實際路徑

output_folder = os.path.join(os.getcwd(), "0809output_features", "0809fft")
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "fft_顛簸(全)_特徵.csv")

if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ 找不到檔案：{input_path}")

fft_df = pd.read_csv(input_path, encoding='utf-8-sig')

# ✅ 正確的 label 清理順序：先丟 NaN 再去空白
fft_df = fft_df.dropna(subset=['label']).copy()
fft_df['label'] = fft_df['label'].astype(str).str.strip()
fft_df = fft_df[fft_df['label'] != '']

print("📂 輸入資料已讀取，共有筆數：", len(fft_df))

# === 步驟 2：智慧特徵擷取器 ===
def smart_fft_feature_selector(df,
    low_freq_max=0.49,   # ← 依你的檔計算
    mid_freq_max=11.31,   # ← 依你的檔計算
    high_freq_min=22.13,  # ← 依你的檔計算
    top_n=3
):
    # 夾到檔案頻率範圍內（避免超過 Nyquist）
    fmin, fmax = float(df['frequency'].min()), float(df['frequency'].max())
    low_freq_max = min(max(low_freq_max, fmin), fmax)
    mid_freq_max = min(max(mid_freq_max, low_freq_max), fmax)
    high_freq_min = min(max(high_freq_min, mid_freq_max), fmax)

    features = []
    for (label, axis), group in df.groupby(['label', 'axis']):
        freqs = group['frequency'].to_numpy()
        amps  = group['amplitude'].to_numpy()
        if freqs.size == 0 or np.sum(amps) == 0:
            continue

        # 基本統計
        peak_idx   = np.argmax(amps)
        peak_freq  = freqs[peak_idx]
        peak_amp   = amps[peak_idx]
        power      = amps**2
        energy     = power.sum()

        # 用能量當權重會更合理
        centroid   = np.sum(freqs * power) / energy
        p          = power / energy
        spectral_entropy = entropy(p)

        # 分段能量
        low_energy  = power[freqs <= low_freq_max].sum()
        mid_energy  = power[(freqs > low_freq_max) & (freqs <= mid_freq_max)].sum()
        high_energy = power[freqs >= high_freq_min].sum()
        high_ratio  = (high_energy / energy) if energy > 0 else 0.0

        # Top-N（長度不足時不報錯）
        k = min(top_n, amps.size)
        top_idx = np.argpartition(amps, -k)[-k:]
        top_idx = top_idx[np.argsort(amps[top_idx])[::-1]]
        top_freqs, top_amps = freqs[top_idx], amps[top_idx]

        row = {
            'label': label,
            'axis': axis,
            'peak_freq': peak_freq,
            'peak_amp': peak_amp,
            'energy': energy,
            'centroid': centroid,
            'entropy': spectral_entropy,
            'low_freq_energy': low_energy,
            'mid_freq_energy': mid_energy,
            'high_freq_ratio': high_ratio
        }
        for i in range(top_n):
            row[f'top{i+1}_freq'] = top_freqs[i] if i < k else np.nan
            row[f'top{i+1}_amp']  = top_amps[i]  if i < k else np.nan

        features.append(row)

    return pd.DataFrame(features)

# === 步驟 3：執行特徵萃取 ===
feature_df = smart_fft_feature_selector(fft_df)

feature_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print("✅ 特徵擷取完成，儲存為：", output_path)
print("\n📊 特徵摘要前 5 筆：")
print(feature_df.head())
