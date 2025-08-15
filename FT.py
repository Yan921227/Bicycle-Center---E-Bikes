import os
import pandas as pd
import numpy as np

# === 基本設定 ===
file_path     = "C:\\Users\\User\\py\\Bicycle_Center_E-Bikes\\0809output_converted\\顛簸\\顛簸(全).xlsx"  # ← 改成你的檔案路徑
sheet_name    = 0
sampling_rate = 100  # Hz

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

# === 欄位設定（已把 Pitch / Roll 納入）===
sensor_columns = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'Pitch', 'Roll']
label_column   = 'Label'

# 欄位檢查
missing = [c for c in [label_column] + sensor_columns if c not in df.columns]
if missing:
    raise KeyError(f"找不到下列欄位：{missing}")

# === 初始化結果 ===
rows = []

# === 依 Label 分組做 FFT（去平均 + Hann 窗 + rFFT）===
for label, group in df.groupby(label_column):
    print(f'處理中：Label = {label}，共 {len(group)} 筆資料')

    for axis in sensor_columns:
        data = group[axis].to_numpy(dtype=float)
        data = data[~np.isnan(data)]
        n = data.size
        if n < 10:
            print(f'⚠️ 跳過 {label} - {axis}，資料筆數太少：{n}')
            continue

        # 角度欄位（Pitch/Roll）可做 unwrap（若原始資料沒有±180/360 跳變，這步不會改變數值）
        if axis in ('Pitch', 'Roll'):
            data = np.deg2rad(data)
            data = np.unwrap(data)
            data = np.rad2deg(data)

        # 去平均（移除 DC 偏移）
        data = data - np.mean(data)

        # Hann 窗
        window = np.hanning(n)
        data_win = data * window

        # rFFT 只取非負頻率
        fft_result = np.fft.rfft(data_win)
        freqs = np.fft.rfftfreq(n, d=1.0/sampling_rate)

        # 單邊幅度校正（coherent gain = Σwindow；DC/Nyquist 不加倍）
        cg = window.sum()
        amplitude = (2.0 / cg) * np.abs(fft_result)
        amplitude[0] *= 0.5
        if n % 2 == 0 and amplitude.size > 1:
            amplitude[-1] *= 0.5

        # 累積結果
        rows.append(pd.DataFrame({
            'label'    : label,
            'axis'     : axis,
            'frequency': freqs,
            'amplitude': amplitude
        }))

# === 輸出結果 ===
result_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
    columns=['label','axis','frequency','amplitude']
)

# 以來源檔名建立輸出資料夾與檔名
basename   = os.path.splitext(os.path.basename(file_path))[0]
output_dir = os.path.join(os.getcwd(), "0809output_fft", basename)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"fft_{basename}.csv")

result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print("✅ FFT 分析完成，結果已儲存為：", output_path)
