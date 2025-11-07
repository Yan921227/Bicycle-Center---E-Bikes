import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib

# ——— 中文字體設定 ———
plt.rcParams["font.family"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 設定資料路徑（CSV 或 Excel 皆可）
file_path = Path("C:\\Users\\User\\Desktop\\已區分\\訓練集\\1018_全數據_訓練.xlsx")

# 2. 根據副檔名自動讀取資料
if file_path.suffix == ".csv":
    df = pd.read_csv(file_path, encoding="utf-8")
else:
    df = pd.read_excel(file_path, engine='openpyxl')

# 3. 篩選數值欄位並準備 X
numeric_cols = df.select_dtypes(include=['number']).columns
X = df[numeric_cols].values

# 4. 標準化
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. PCA 降維（2 維）
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 6. K-means 分群
kmeans = KMeans(
    n_clusters=8,
    init='k-means++',
    n_init=10,
    random_state=42
)
labels = kmeans.fit_predict(X_pca)

# 7. 畫圖
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=labels, cmap='viridis',
    alpha=0.7, edgecolors='none'
)
ax.set_xlabel('主成分 1')
ax.set_ylabel('主成分 2')
ax.set_title(f'{file_path.stem} - 2D PCA + K-means 聚類結果')
fig.colorbar(scatter, ax=ax, label='Cluster')
ax.grid(True)
plt.tight_layout()

# 8. 儲存圖檔至 output_cluster/plots/
output_root = Path.cwd() / "1020output_cluster"
plots_dir = output_root / "1020plots"
plots_dir.mkdir(parents=True, exist_ok=True)
image_path = plots_dir / f'{file_path.stem}.png'
fig.savefig(image_path, dpi=300)
print(f"→ 已將聚類圖儲存至：{image_path}")

plt.show()

# 9. 儲存模型至 output_cluster/models/
models_dir = output_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "KM_1018_全數據_models.pkl"

models = {
    'scaler': scaler,
    'pca': pca,
    'kmeans': kmeans
}
joblib.dump(models, model_path)
print(f"→ 模型已儲存至：{model_path}")
