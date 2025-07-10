import pandas as pd
import numpy as np
import os
from scipy.fft import fft, fftfreq

# === 參數設定 ===
input_path = "C:\\Users\\User\\Desktop\\自行車\\左轉_右轉_訓練集.csv"  # 原始資料
segment_size = 100                     # 每段資料筆數
sampling_rate = 200                    # 每秒取樣頻率
dc_threshold_ratio = 0.5               # DC 偏移判斷閾值

# === 設定輸出資料夾結構 ===
output_dir = os.path.join(os.getcwd(), "output_fft_segments", "左轉_右轉")
os.makedirs(output_dir, exist_ok=True)  # 自動建立資料夾（若不存在）

# 最終輸出 CSV 路徑
output_path = os.path.join(output_dir, "fft_左轉_右轉.csv")

# === 健康檢查模組 ===
def check_dc_offset(df, axes, threshold_ratio=0.5):
    dc_status = {}
    print("🧪 DC 偏移健康檢查報告：\n")
    for axis in axes:
        mean_val = df[axis].mean()
        std_val = df[axis].std()
        ratio = abs(mean_val) / std_val if std_val != 0 else np.inf

        if ratio > threshold_ratio:
            status = "⚠️ 偏移明顯，建議去除 DC"
            dc_status[axis] = True
        else:
            status = "✅ 偏移可接受"
            dc_status[axis] = False

        print(f"{axis}: 平均值={mean_val:.4f}, 標準差={std_val:.4f}, 比例={ratio:.2f} → {status}")
    return dc_status

# === 讀取資料 ===
df = pd.read_csv(input_path)
axes = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']

# === 執行 DC 偏移健康檢查 ===
dc_flags = check_dc_offset(df, axes, threshold_ratio=dc_threshold_ratio)

# === FFT 處理流程 ===
segments = []
for start in range(0, len(df), segment_size):
    end = start + segment_size
    if end > len(df):
        break

    segment = df.iloc[start:end]
    label = segment['Label'].mode()[0]

    for axis in axes:
        raw_signal = segment[axis].values

        # ✅ 如果該軸被標記為偏移嚴重 → 去除平均值
        if dc_flags[axis]:
            signal = raw_signal - np.mean(raw_signal)
        else:
            signal = raw_signal

        N = len(signal)
        yf = np.abs(fft(signal))[:N // 2]
        xf = fftfreq(N, 1 / sampling_rate)[:N // 2]

        for f, a in zip(xf, yf):
            segments.append({
                'frequency': f,
                'amplitude': a,
                'axis': axis,
                'label': label
            })

# === 匯出 CSV ===
fft_df = pd.DataFrame(segments)
fft_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ FFT 完成（含自動去 DC），已儲存：{output_path}")
print(fft_df.head())
