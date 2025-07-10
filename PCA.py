import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
import os

# ———— 支援中文字型 & 負號 ————
plt.rcParams["font.family"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# ———— 1. 資料檔案路徑（請確認副檔名正確）————
excel_path = Path("C:\\Users\\User\\Desktop\\自行車\\平路_上坡_下坡_顛簸.csv")

# ———— 2. 讀取資料：自動判斷副檔名使用對應方式 ————
if excel_path.suffix == ".csv":
    df = pd.read_csv(excel_path, encoding="utf-8")  # 如遇亂碼可改為 "cp950" 或 "big5"
else:
    df = pd.read_excel(excel_path, sheet_name="工作表1", engine="openpyxl")

# ———— 3. 萃取特徵與標籤 ————
features = ["AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"]
X = df[features]
y = df["Label"].astype(str)

# ———— 4. 標準化 & PCA ————
X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# 把結果加回 DataFrame
df["PC1"], df["PC2"] = X_pca[:, 0], X_pca[:, 1]

# ———— 5. 建立輸出資料夾結構 ————
output_root = Path.cwd() / "output_pca"
plots_dir = output_root / "plots"
models_dir = output_root / "models"
plots_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

# ———— 6. 定義各 Label 顏色（可自行擴充）————
colors = {
    "上坡": "purple",
    "下坡": "brown",
    "平路": "red",
    "顛簸": "gray",
    # "左轉": "green",
    # "右轉": "blue",
    # "加速": "orange",
    # "減速": "pink",
}

# ———— 7. 繪製 & 儲存 PCA 散佈圖 ————
scatter_file = plots_dir / f"{excel_path.stem}_PC1_vs_PC2.png"
plt.figure(figsize=(8, 6))

for lab, col in colors.items():
    subset = df[df["Label"] == lab]
    plt.scatter(
        subset["PC1"], subset["PC2"],
        c=col, label=lab,
        s=30, alpha=0.7
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 分佈圖 (PC1 vs PC2)")
plt.legend(title="Label")
plt.grid(linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(scatter_file, dpi=300)
plt.close()

print(f"✔ 已將散佈圖儲存至：{scatter_file}")

# ———— 8. 儲存 PCA 模型 ————
pca_model_path = models_dir / "pca_model.pkl"
joblib.dump(pca, pca_model_path)
print(f"✔ PCA 模型已儲存至：{pca_model_path}")
