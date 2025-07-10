import os
import pandas as pd
import numpy as np

# === 基本設定 ===
file_path     = "C:\\Users\\User\\Desktop\\電輔車\\加速.xlsx"  # ← 請確認路徑正確
sheet_name    = 0       # 只在讀 Excel 時會用到
sampling_rate = 200     # Hz，若感測器為 100Hz，可改為 100

# === 依副檔名自動選擇讀檔方式 ===
ext = os.path.splitext(file_path)[1].lower()
if ext == '.csv':
    df = pd.read_csv(file_path)
    print("✅ 已以 CSV 格式讀取資料")
elif ext in ('.xls', '.xlsx'):
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    print("✅ 已以 Excel 格式讀取資料")
else:
    raise ValueError(f"不支援的檔案格式：{ext}，僅接受 .csv / .xls / .xlsx")

# === 正確對應感測器欄位名稱 ===
sensor_columns = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']
label_column   = 'Label'

# === 初始化儲存 FFT 結果的列表 ===
results = []

# === 對每個 Label 群組進行 FFT 分析 ===
for label, group in df.groupby(label_column):
    print(f'處理中：Label = {label}，共 {len(group)} 筆資料')

    for axis in sensor_columns:
        data = group[axis].dropna().values
        n = len(data)
        if n < 10:
            print(f'⚠️ 跳過 {label} - {axis}，資料筆數太少：{n}')
            continue

        # FFT 計算
        fft_result = np.fft.fft(data)
        freqs      = np.fft.fftfreq(n, d=1.0/sampling_rate)
        amplitude  = 2.0/n * np.abs(fft_result)

        # 僅取正頻率
        half_n = n // 2
        for f, a in zip(freqs[:half_n], amplitude[:half_n]):
            results.append({
                'label'    : label,
                'axis'     : axis,
                'frequency': f,
                'amplitude': a
            })

# === 輸出結果 ===

# 🔧 建立輸出資料夾（例如 ./output_fft/左轉_右轉）
output_dir = os.path.join(os.getcwd(), "output_fft", "加速")
os.makedirs(output_dir, exist_ok=True)

# 🔽 設定輸出檔案路徑
output_path = os.path.join(output_dir, "fft_加速.csv")

# 寫出 CSV
result_df = pd.DataFrame(results)
result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print("✅ FFT 分析完成，結果已儲存為：", output_path)
