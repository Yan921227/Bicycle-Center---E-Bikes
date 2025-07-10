import pandas as pd
import numpy as np
from pathlib import Path
from scipy.fft import fft, fftfreq
from scipy.stats import entropy

# === 參數 ===
src = Path("C:\\Users\\User\\Desktop\\電輔車\\上坡.xlsx")
segment = 100
fs = 200
dc_thr = 0.5
out_root = Path.cwd() / "output_all" / src.stem
out_root.mkdir(parents=True, exist_ok=True)

# === 1. 讀檔（自動辨識） ===
df = (pd.read_csv(src, encoding="utf-8") if src.suffix==".csv"
      else pd.read_excel(src, engine="openpyxl"))
axes = ['AccelX','AccelY','AccelZ','GyroX','GyroY','GyroZ']

# === 2. DC 檢查並標記需要去 DC 的軸 ===
def dc_flags(df, axes, thr):
    flags = {}
    for a in axes:
        mu, sigma = df[a].mean(), df[a].std()
        flags[a] = (abs(mu)/sigma>thr) if sigma else True
    return flags
flags = dc_flags(df, axes, dc_thr)

# === 3. 分段 FFT 產出「頻率表」 ===
records=[]
for s in range(0, len(df)//segment*segment, segment):
    seg = df.iloc[s:s+segment]
    label = seg['Label'].mode().iloc[0]
    for ax in axes:
        sig = seg[ax].values
        if flags[ax]: sig -= sig.mean()        # 去 DC
        yf = np.abs(fft(sig))[:segment//2]
        xf = fftfreq(segment, 1/fs)[:segment//2]
        records += [{'label':label,'axis':ax,'frequency':f,'amplitude':a}
                    for f,a in zip(xf,yf)]
fft_df = pd.DataFrame(records)
fft_csv = out_root / f"fft_{src.stem}.csv"
fft_df.to_csv(fft_csv, index=False, encoding='utf-8-sig')

# === 4. 根據 FFT 統計主頻峰值，推算區間門檻 ===
peak_freqs = (fft_df.loc[fft_df.groupby(['label','axis'])['amplitude'].idxmax(),
                         'frequency'].values)
q25, q75 = np.percentile(peak_freqs, [25, 75])
low_max   = round(q25, 2)
mid_max   = round(q75, 2)
high_min  = round(q75 + (q75-q25), 2)

print(f"🚀 自動推薦頻率區間： low≤{low_max} Hz,  "
      f"mid≤{mid_max} Hz,  high≥{high_min} Hz")

# === 5. 特徵萃取，使用動態門檻 ===
def feature_select(df, low, mid, high, top_n=3):
    feats=[]
    for (lbl, ax), g in df.groupby(['label','axis']):
        f, a = g['frequency'].values, g['amplitude'].values
        if a.sum()==0: continue
        peak = a.argmax()
        energy = (a**2).sum()
        row = dict(label=lbl, axis=ax,
                   peak_freq = f[peak],
                   peak_amp  = a[peak],
                   energy    = energy,
                   centroid  = (f*a).sum()/a.sum(),
                   entropy   = entropy(a/a.sum()),
                   low_freq_energy  = (a[f<=low]**2).sum(),
                   mid_freq_energy  = (a[(f>low)&(f<=mid)]**2).sum(),
                   high_freq_ratio  = (a[f>=high]**2).sum()/energy)
        top_idx = a.argsort()[-top_n:][::-1]
        for i, idx in enumerate(top_idx,1):
            row[f'top{i}_freq']=f[idx]; row[f'top{i}_amp']=a[idx]
        feats.append(row)
    return pd.DataFrame(feats)

feat_df = feature_select(fft_df, low_max, mid_max, high_min, top_n=3)
feat_csv = out_root / f"features_{src.stem}.csv"
feat_df.to_csv(feat_csv, index=False, encoding='utf-8-sig')

print(f"✅ 特徵擷取完成；檔案已存 {feat_csv}")
