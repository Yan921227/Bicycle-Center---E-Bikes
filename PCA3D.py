# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 啟用 3D 繪圖支援
import os

# ———— 支援中文字型 & 負號 ————
plt.rcParams["font.family"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# ———— 1. 資料來源檔案路徑 ————
excel_path = Path("C:\\Users\\User\\Desktop\\已區分\\訓練集\\1018_全數據_訓練.xlsx")

# ———— 2. 自動依副檔名選擇讀檔方式 ————
if excel_path.suffix == ".csv":
    df = pd.read_csv(excel_path, encoding="utf-8")
else:
    df = pd.read_excel(
        excel_path,
        sheet_name="工作表1",
        engine="openpyxl"
    )

# ———— 3. 萃取特徵與標籤 ————
features = ["AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"]
X = df[features]
y = df["Label"].astype(str)

# ———— 4. 標準化 & PCA（三維） ————
X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_std)

# 把前三主成份加回 DataFrame
df["PC1"], df["PC2"], df["PC3"] = X_pca[:, 0], X_pca[:, 1], X_pca[:, 2]

# ———— 5. 建立統一的輸出資料夾 ————
output_root = Path.cwd() / "1020output_pca_3d"
plots_dir = output_root / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# ———— 6. 定義 Label 顏色對應 ————
colors = {
      "上坡": "purple",
      "下坡": "brown",
      "平路": "red",
      "顛簸": "gray",
      "左轉": "green",
      "右轉": "blue",
      "加速": "orange",
      "減速": "pink",
}

# ———— 7. 繪製 3D PCA 散佈圖 ————
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for lab, col in colors.items():
    subset = df[df["Label"] == lab]
    ax.scatter(
        subset["PC1"], subset["PC2"], subset["PC3"],
        c=col,
        label=lab,
        s=30,
        alpha=0.7
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("PCA 三維分佈圖 (PC1 vs PC2 vs PC3)")
plt.legend(title="Label", loc="best")
plt.grid(linestyle="--", alpha=0.3)
plt.tight_layout()

# ———— 8. 儲存圖檔 ————
scatter_file = plots_dir / f"{excel_path.stem}_PCA_3D.png"
plt.savefig(scatter_file, dpi=300)
plt.close()

print(f"✔ 已將三維散佈圖儲存至：{scatter_file}")
