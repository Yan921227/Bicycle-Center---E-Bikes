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

# ——— 中文字體設定 ———
plt.rcParams["font.family"]        = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False


def train_and_save_confusion(dataset_path, target_column: str):
    """
    訓練 Random Forest 並輸出：
    1. 混淆矩陣圖 (PNG)
    2. 特徵重要性 (console)
    3. 模型檔 (PKL)
    """
    dataset_path = Path(dataset_path)

    # 1. 自動讀取資料
    try:
        if dataset_path.suffix.lower() == ".csv":
            df = pd.read_csv(dataset_path, encoding="utf-8")
        else:
            df = pd.read_excel(dataset_path, engine="openpyxl")
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_path, encoding="utf-8-sig")

    df.columns = df.columns.str.strip()

    if target_column not in df.columns:
        matches = [col for col in df.columns if col.lower() == target_column.lower()]
        if matches:
            target_column = matches[0]
            print(f"⚠️ 偵測到實際標籤欄位為「{target_column}」，已自動更正。")
        else:
            raise ValueError(f"❌ 找不到標籤欄位：{target_column}（目前欄位：{list(df.columns)}）")

    # 2. 特徵與標籤分離
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(str)

    # 3. One-Hot 編碼
    X = pd.get_dummies(X, drop_first=False)

    # 4. 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    min_class_count = y_train.value_counts().min()
    cv_folds = min(5, min_class_count)
    if cv_folds < 2:
        raise ValueError(f"❌ 無法交叉驗證：某類別樣本數僅 {min_class_count}，至少需 2。")
    print(f"→ 使用 cv = {cv_folds} 進行交叉驗證")

    # 5. 設定 GridSearchCV
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators':      [100, 200, 300, 400],
        'max_depth':         [None, 10, 20, 30],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(
        rf, param_grid, cv=cv_folds,
        scoring='accuracy', n_jobs=-1
    )
    print("→ 正在進行超參數調校...")
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    print(f"✅ 最佳參數：{grid.best_params_}")

    # 6. 評估
    y_pred = best_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"→ 測試集準確率：{acc:.4f}")
    print("→ 分類報告：")
    print(classification_report(y_test, y_pred, digits=4))

    # 7. 建立輸出資料夾（以 dataset 名稱為子資料夾）
    base_name = dataset_path.stem
    output_root = Path.cwd() / "output_rf" / base_name
    images_dir = output_root / "images"
    models_dir = output_root / "models"
    images_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 8. 混淆矩陣儲存
    cm = confusion_matrix(y_test, y_pred, labels=best_rf.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=best_rf.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
    plt.title('Confusion Matrix')
    png_path = images_dir / f"{base_name}_混淆矩陣.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 混淆矩陣已儲存：{png_path}")

    # 9. 特徵重要性輸出
    importances = best_rf.feature_importances_
    features = X.columns
    ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    print("\n→ 特徵重要性排序：")
    for name, score in ranked:
        print(f"{name}: {score:.4f}")

    # 10. 模型儲存
    model_path = models_dir / f"{base_name}_model.pkl"
    joblib.dump(best_rf, model_path)
    print(f"🧠 模型已儲存：{model_path}")


# ——— 主程式（執行入口）———
if __name__ == "__main__":
    # ⇩⇩⇩ 設定檔案與標籤欄位 ⇩⇩⇩
    dataset_path  = "fft_加速_減速_特徵.csv"
    target_column = "label"
    # ⇧⇧⇧ 可依情況更改 ⇧⇧⇧

    train_and_save_confusion(dataset_path, target_column)
    print("🎉 任務完成！")
