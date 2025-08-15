# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, classification_report
)

# ——— 中文字體設定（沒有這字體也不影響訓練，只影響圖表顯示）———
plt.rcParams["font.family"]        = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False


def _read_one(path: Path) -> pd.DataFrame:
    """讀一個 CSV 或 Excel（帶編碼容錯；Excel 會把所有工作表合併）"""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8")
    elif suf in (".xls", ".xlsx"):
        x = pd.read_excel(path, sheet_name=None, engine="openpyxl")  # 讀全部工作表
        if isinstance(x, dict):
            df = pd.concat(x.values(), ignore_index=True)
        else:
            df = x
        return df
    else:
        raise ValueError(f"不支援的檔案格式：{path}")


def _load_and_concat(paths_or_dirs) -> pd.DataFrame:
    """支援多檔/資料夾輸入，自動合併成一個 DataFrame"""
    files = []
    for p in paths_or_dirs if isinstance(paths_or_dirs, (list, tuple)) else [paths_or_dirs]:
        p = Path(p)
        if p.is_dir():
            files += sorted(list(p.rglob("*.csv")) + list(p.rglob("*.xlsx")) + list(p.rglob("*.xls")))
        else:
            files.append(p)
    if not files:
        raise FileNotFoundError("找不到任何 .csv/.xlsx 檔案")

    dfs = []
    for f in files:
        df = _read_one(f)
        df.columns = df.columns.str.strip()
        dfs.append(df)
        print(f"✔ 讀取：{f}  shape={df.shape}")
    out = pd.concat(dfs, ignore_index=True, sort=False)
    out.columns = out.columns.str.strip()
    return out


def train_and_save_confusion(dataset_paths, target_column: str = "label"):
    """
    dataset_paths: str 或 List[str]；可同時丟多個特徵檔/資料夾
    target_column: 目標欄位名（預設 'label'）
    """
    # 1) 讀取並合併
    df = _load_and_concat(dataset_paths)

    # 目標欄位容錯（大小寫 / 空白）
    colmap = {c.lower(): c for c in df.columns}
    if target_column.lower() not in colmap:
        raise ValueError(f"❌ 找不到標籤欄位：{target_column}（目前欄位：{list(df.columns)}）")
    target_column = colmap[target_column.lower()]

    # 清理標籤內容
    df = df.dropna(subset=[target_column]).copy()
    df[target_column] = df[target_column].astype(str).str.strip()

    # 2) 特徵/標籤分離
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(str)

    # 3) One-Hot（將類別欄位如 axis 轉為數值）
    X = pd.get_dummies(X, drop_first=False)

    # 4) 分割資料
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print("資料量：", X.shape, "\n類別分布：")
    print(y.value_counts())

    # 交叉驗證折數取決於最小類別樣本數
    min_class_count = y_train.value_counts().min()
    cv_folds = min(5, int(min_class_count))
    if cv_folds < 2:
        raise ValueError(f"❌ 無法交叉驗證：某類別樣本數僅 {min_class_count}，至少需 2。")
    print(f"→ 使用 cv = {cv_folds} 進行交叉驗證")

    # 5) GridSearchCV 找最佳參數
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(rf, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
    print("→ 正在進行超參數調校...")
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    print(f"✅ 最佳參數：{grid.best_params_}")

    # 6) 評估
    y_pred = best_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"→ 測試集準確率：{acc:.4f}")
    print("→ 分類報告：")
    print(classification_report(y_test, y_pred, digits=4))

    # 7) 輸出資料夾（合併情境統一放 combined）
    output_root = Path.cwd() / "0809output_rf" / "combined"
    images_dir = output_root / "images"
    models_dir = output_root / "models"
    images_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 8) 混淆矩陣 PNG
    cm = confusion_matrix(y_test, y_pred, labels=best_rf.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=best_rf.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
    plt.title('Confusion Matrix')
    png_path = images_dir / "combined_混淆矩陣.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 混淆矩陣已儲存：{png_path}")

    # 9) 特徵重要性（列前 20 名）
    importances = best_rf.feature_importances_
    features = X.columns
    ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    print("\n→ 特徵重要性（前 20）：")
    for name, score in ranked[:20]:
        print(f"{name}: {score:.4f}")

    # 10) 模型 + 欄位順序 一起儲存
    artifact = {
        "model": best_rf,
        "feature_columns": list(X.columns)
    }
    model_path = models_dir / "combined_model.pkl"
    joblib.dump(artifact, model_path)
    print(f"🧠 模型已儲存：{model_path}")


# ——— 這裡「寫死」你的檔案路徑與標籤欄位 ———
if __name__ == "__main__":
    DATA_FILE = "C:\\Users\\User\\py\\Bicycle_Center_E-Bikes\\0809Training_set_merging\\0809平路_顛簸.xlsx"
    TARGET    = "label"   # 如果你的標籤欄位不是 label，改這裡

    train_and_save_confusion(DATA_FILE, TARGET)
