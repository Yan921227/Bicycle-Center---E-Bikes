# -*- coding: utf-8 -*-
"""
統一輸出格式版本（Route B Enhanced）
---------------------------------------
✔ 自動辨識 .csv / .xlsx 測試集
✔ 混淆矩陣圖與 Excel 評估報告自動儲存
✔ 不需手動填 img_path 或 output_excel
✔ 統一輸出於 ./output_eval/<檔名>/
"""

from pathlib import Path
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ——— 中文字體設定（Windows）———
plt.rcParams["font.family"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

def evaluate_model(
    model_path: Path,
    test_path: Path,
    scaler_path: Path | None = None,
    features: list[str] | None = None,
    label_col: str = "Label"
) -> None:
    # === 1. 載入模型 ===
    model = load(model_path)
    print(f"→ 已載入模型：{model_path}")

    # === 2. 讀取測試資料（CSV 或 Excel）===
    if test_path.suffix.lower() == ".csv":
        data = pd.read_csv(test_path, encoding="utf-8")
    else:
        data = pd.read_excel(test_path, engine="openpyxl")
    print(f"→ 已讀取測試資料：{test_path}")

    data.columns = data.columns.str.strip()

    # === 3. 特徵欄位處理 ===
    if features is None:
        features = ["AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"]

    missing = set(features + [label_col]) - set(data.columns)
    if missing:
        raise KeyError(f"❌ 測試檔缺少欄位：{missing}\n實際欄位：{list(data.columns)}")

    X_test = data[features]
    y_test = data[label_col].fillna(method="ffill")

    # === 4. 標準化 ===
    if scaler_path and scaler_path.exists():
        scaler = load(scaler_path)
        print(f"→ 已載入 scaler：{scaler_path}")
        X_test_scaled = scaler.transform(X_test)
    else:
        print("⚠️ 未提供 scaler，將重新 fit 標準化（僅供參考）")
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)

    # === 5. 預測與指標 ===
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"→ 測試集 Accuracy：{acc:.4f}")
    print("→ 分類報告：")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))

    # === 6. 建立輸出資料夾結構 ===
    base_name = test_path.stem
    output_root = Path.cwd() / "output_eval" / base_name
    plots_dir = output_root / "plots"
    reports_dir = output_root / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # === 7. 混淆矩陣圖 ===
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted(y_test.unique()),
        yticklabels=sorted(y_test.unique())
    )
    plt.xlabel("True Label", fontsize=12)
    plt.ylabel("Predicted Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=16)
    plt.tight_layout()

    img_path = plots_dir / f"{base_name}_confusion.png"
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🖼️ 混淆矩陣圖已儲存：{img_path}")

    # === 8. 輸出 Excel 報告 ===
    report_path = reports_dir / f"{base_name}_report.xlsx"
    cr_dict = classification_report(y_test, y_pred, output_dict=True)

    with pd.ExcelWriter(report_path) as writer:
        pd.DataFrame({"Accuracy": [acc]}).to_excel(writer, sheet_name="Accuracy", index=False)
        pd.DataFrame(cr_dict).T.to_excel(writer, sheet_name="Classification_Report")
        pd.DataFrame(cm, index=sorted(y_test.unique()), columns=sorted(y_test.unique()))\
            .to_excel(writer, sheet_name="Confusion_Matrix")
        pd.DataFrame({"True": y_test, "Predicted": y_pred})\
            .to_excel(writer, sheet_name="Predictions", index=False)

    print(f"✅ 評估報告已儲存：{report_path}")


# === 主程式 ===
if __name__ == "__main__":
    # 你只需更改下面兩條路徑即可：
    model_path = Path("models/左轉_右轉_model.pkl")
    test_path = Path("測試資料/左轉_右轉_測試集.xlsx")
    scaler_path = None  # 若有 scaler 可填

    evaluate_model(
        model_path=model_path,
        test_path=test_path,
        scaler_path=scaler_path,
    )
