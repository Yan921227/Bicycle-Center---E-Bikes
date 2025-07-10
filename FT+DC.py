import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.fft import fft, fftfreq

# === 參數設定 ===
input_path = Path("C:\\Users\\User\\Desktop\\電輔車\\右轉.xlsx")  # 支援 Windows 路徑處理
segment_size = 100
sampling_rate = 200
dc_threshold_ratio = 0.5

# === 輸出資料夾結構 ===
output_dir = Path.cwd() / "output_fft_segments" / input_path.stem
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"fft_{input_path.stem}.csv"

# === 讀取資料（自動辨識格式 + 容錯） ===
try:
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path, encoding="utf-8")
    else:
        df = pd.read_excel(input_path, engine="openpyxl")
    print(f"📥 已成功讀取檔案：{input_path.name}（共 {len(df)} 筆）")
except Exception as e:
    raise RuntimeError(f"❌ 無法讀取檔案：{input_path.name}\n錯誤訊息：{e}")

# === 確保欄位存在 ===
required_columns = ['Label', 'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
missing = set(required_columns) - set(df.columns)
if missing:
    raise ValueError(f"❌ 資料缺少欄位：{missing}\n實際欄位：{list(df.columns)}")

axes = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']

# === 健康檢查（DC 偏移） ===
def check_dc_offset(df, axes, threshold_ratio=0.5):
    dc_status = {}
    print("🧪 DC 偏移健康檢查報告：")
    for axis in axes:
        series = df[axis].dropna()
        mean_val = series.mean()
        std_val = series.std()
        ratio = abs(mean_val) / std_val if std_val != 0 else np.inf

        if np.isnan(ratio) or ratio > threshold_ratio:
            dc_status[axis] = True
            status = "⚠️ 建議去除 DC"
        else:
            dc_status[axis] = False
            status = "✅ 可接受"
        print(f"  {axis}: 平均={mean_val:.4f}, 標準差={std_val:.4f}, 比例={ratio:.2f} → {status}")
    return dc_status

dc_flags = check_dc_offset(df, axes, threshold_ratio=dc_threshold_ratio)

# === FFT 分段處理 ===
segments = []
for start in range(0, len(df), segment_size):
    end = start + segment_size
    if end > len(df):
        continue

    segment = df.iloc[start:end]
    label = segment['Label'].mode().iloc[0] if 'Label' in segment.columns else '未知'

    for axis in axes:
        raw = segment[axis].dropna().values
        if len(raw) < segment_size:
            continue

        signal = raw - np.mean(raw) if dc_flags[axis] else raw

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

# === 輸出 CSV ===
fft_df = pd.DataFrame(segments)
fft_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n✅ FFT 完成，共轉換 {len(fft_df)} 筆頻譜資料")
print(f"📁 輸出位置：{output_path}")
print("📊 前 5 筆資料預覽：")
print(fft_df.head())
