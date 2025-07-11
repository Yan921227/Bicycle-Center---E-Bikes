import pandas as pd
import numpy as np
from scipy.stats import entropy
import os

# === 步驟 1：讀取傅立葉轉換後的 CSV（輸入）===
input_path = "output_fft_segments\\加速\\fft_加速.csv"  # ← 輸入檔案名稱

# 使用 os.getcwd() + 多層資料夾結構
output_folder = os.path.join(os.getcwd(), "output_features", "fft")  # ⬅️ 可以自訂資料夾結構
os.makedirs(output_folder, exist_ok=True)  # 自動建立資料夾（如果不存在）

# 最終輸出檔案位置
output_path = os.path.join(output_folder, "fft_加速_特徵.csv")

# 驗證輸入檔案是否存在
if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ 找不到檔案：{input_path}")

# 讀取 CSV
fft_df = pd.read_csv(input_path, encoding='utf-8-sig')

# 清理 label 欄位
fft_df['label'] = fft_df['label'].astype(str).str.strip()
fft_df = fft_df[fft_df['label'].notna()]
fft_df = fft_df[fft_df['label'] != '']

print("📂 輸入資料已讀取，共有筆數：", len(fft_df))


# === 步驟 2：智慧特徵擷取器定義 ===
def smart_fft_feature_selector(df, low_freq_max=4.00, mid_freq_max=61.50, high_freq_min=119.00, top_n=3):
    features = []

    for (label, axis), group in df.groupby(['label', 'axis']):
        freqs = group['frequency'].values
        amps = group['amplitude'].values

        if len(freqs) == 0 or np.sum(amps) == 0:
            continue

        # 基本統計
        peak_idx = np.argmax(amps)
        peak_freq = freqs[peak_idx]
        peak_amp = amps[peak_idx]
        energy = np.sum(amps ** 2)
        centroid = np.sum(freqs * amps) / np.sum(amps)
        p = amps / np.sum(amps)
        spectral_entropy = entropy(p)

        # 分段能量
        low_energy = np.sum((amps[freqs <= low_freq_max]) ** 2)
        mid_energy = np.sum((amps[(freqs > low_freq_max) & (freqs <= mid_freq_max)]) ** 2)
        high_energy = np.sum((amps[freqs >= high_freq_min]) ** 2)
        high_ratio = high_energy / energy if energy > 0 else 0

        # Top N 頻率與振幅
        top_indices = np.argsort(amps)[-top_n:][::-1]
        top_freqs = freqs[top_indices]
        top_amps = amps[top_indices]

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
            row[f'top{i+1}_freq'] = top_freqs[i]
            row[f'top{i+1}_amp'] = top_amps[i]

        features.append(row)

    return pd.DataFrame(features)


# === 步驟 3：執行特徵萃取 ===
feature_df = smart_fft_feature_selector(fft_df)

# 儲存結果
feature_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print("✅ 特徵擷取完成，儲存為：", output_path)

# 顯示前幾筆
print("\n📊 特徵摘要前 5 筆：")
print(feature_df.head())
