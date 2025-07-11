import pandas as pd
import numpy as np
import chardet
import os

# ✅ 通用檔案讀取器（支援 .csv 與 .xlsx）
def read_file_auto_encoding(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.csv':
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']
            print(f"🔍 偵測到 CSV 編碼: {detected_encoding}")
        df = pd.read_csv(file_path, encoding=detected_encoding)

    elif ext in ['.xls', '.xlsx']:
        print("📄 偵測到 Excel 檔案：使用 pandas.read_excel()")
        df = pd.read_excel(file_path)

    else:
        raise ValueError("❌ 不支援的檔案格式，請使用 .csv 或 .xlsx")
    
    print(f"✅ 成功讀取檔案：{file_path}")
    return df

# ✅ 主頻率提取函式
def get_peak_frequencies(df):
    peak_freqs = []
    for (label, axis), group in df.groupby(['label', 'axis']):
        idx = group['amplitude'].idxmax()
        peak_freq = group.loc[idx, 'frequency']
        peak_freqs.append(peak_freq)
    return np.array(peak_freqs)

# === ✅ 主程式執行 ===
file_path = "output_fft_segments\\顛簸\\fft_顛簸.csv"  # ← 替換為你資料的實際位置
fft_df = read_file_auto_encoding(file_path)

# 執行主頻分析
peak_freqs = get_peak_frequencies(fft_df)

# 計算分位數
q25 = np.percentile(peak_freqs, 25)
q50 = np.percentile(peak_freqs, 50)
q75 = np.percentile(peak_freqs, 75)
high_freq = q75 + (q75 - q25)

# 顯示推薦參數
print("\n🚀 自動推薦頻率區間參數（依據資料主頻分布）：")
print(f"🔹 low_freq_max  = {q25:.2f} Hz")
print(f"🔸 mid_freq_max  = {q75:.2f} Hz")
print(f"🔺 high_freq_min = {high_freq:.2f} Hz")
