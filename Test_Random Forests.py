# -*- coding: utf-8 -*-
"""
隨機森林模型測試程式（配合 Random_Forests.py 訓練）
---------------------------------------
✔ 自動辨識 .csv / .xlsx 測試集
✔ 使用訓練時保存的 feature_columns 進行 One-Hot 編碼
✔ 混淆矩陣圖與 Excel 評估報告自動儲存
✔ 支援「純預測模式」（無標籤）和「評估模式」（有標籤）
✔ 輸出於 ./1020output_rf/combined/predictions/ 或 test_results/
"""

from pathlib import Path
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ——— 中文字體設定（Windows）———
plt.rcParams["font.family"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False


def _read_one(path: Path) -> pd.DataFrame:
    """讀一個 CSV 或 Excel（帶編碼容錯）"""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8")
    elif suf in (".xls", ".xlsx"):
        x = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        if isinstance(x, dict):
            df = pd.concat(x.values(), ignore_index=True)
        else:
            df = x
        return df
    else:
        raise ValueError(f"不支援的檔案格式：{path}")


def evaluate_model(
    model_path: Path,
    test_path: Path,
    label_col: str = "label",
) -> None:
    """
    評估隨機森林模型
    
    工作流程：
    1. 載入測試資料（包含 Label 欄位）
    2. 移除 Label 欄位（模型預測時不使用）
    3. 模型進行純預測
    4. 比對預測結果 vs 真實標籤
    5. 計算準確率、混淆矩陣等評估指標
    
    Args:
        model_path: 訓練好的模型路徑（pkl 檔）
        test_path: 測試資料路徑（csv 或 xlsx，應包含 Label 欄位）
        label_col: 標籤欄位名稱（預設 'label'）
    """
    # === 1. 載入模型 artifact（包含 model 和 feature_columns）===
    artifact = load(model_path)
    if not isinstance(artifact, dict):
        raise TypeError(f"❌ 模型檔格式錯誤：應為 dict，實際為 {type(artifact)}")
    
    model = artifact.get("model")
    feature_columns = artifact.get("feature_columns")
    
    if model is None:
        raise ValueError("❌ 模型檔缺少 'model' 欄位")
    if feature_columns is None:
        raise ValueError("❌ 模型檔缺少 'feature_columns' 欄位")
    
    print(f"✔ 已載入模型：{model_path}")
    print(f"✔ 訓練時特徵數量：{len(feature_columns)}")

    # === 2. 讀取測試資料 ===
    df = _read_one(test_path)
    df.columns = df.columns.str.strip()
    print(f"✔ 已讀取測試資料：{test_path}  shape={df.shape}")

    # === 3. 檢查標籤欄位並保存（用於事後比對）===
    y_test = None
    has_labels = False
    
    # 先檢查是否有標籤欄位
    colmap = {c.lower(): c for c in df.columns}
    if label_col.lower() in colmap:
        label_col_actual = colmap[label_col.lower()]
        print(f"✔ 找到標籤欄位：{label_col_actual}")
        
        # 檢查是否有有效標籤
        valid_labels = df[label_col_actual].notna().sum()
        if valid_labels > 0:
            print(f"✔ 有效標籤數量：{valid_labels} / {len(df)}")
            # 保存真實標籤（用於事後評估）
            y_test_full = df[label_col_actual].copy()
            has_labels = True
        else:
            print(f"⚠️ 標籤欄位全部為空")
            y_test_full = None
    else:
        print(f"⚠️ 找不到標籤欄位：{label_col}")
        label_col_actual = None
        y_test_full = None
    
    # 分離特徵（移除標籤欄位，模型預測時不使用）
    if label_col_actual and label_col_actual in df.columns:
        X_test_raw = df.drop(columns=[label_col_actual])
        print(f"✔ 已將標籤欄位從特徵中移除（模型不會看到標籤）")
    else:
        X_test_raw = df
    
    print(f"✔ 特徵資料 shape：{X_test_raw.shape}")

    # === 4. One-Hot 編碼（與訓練時一致）===
    X_test_encoded = pd.get_dummies(X_test_raw, drop_first=False)
    print(f"✔ One-Hot 編碼後 shape：{X_test_encoded.shape}")
    
    # 確保測試集的特徵與訓練時完全一致
    # 缺少的欄位補 0，多餘的欄位刪除
    missing_cols = set(feature_columns) - set(X_test_encoded.columns)
    extra_cols = set(X_test_encoded.columns) - set(feature_columns)
    
    if missing_cols:
        print(f"⚠️ 測試集缺少 {len(missing_cols)} 個訓練特徵，將補 0")
        for col in missing_cols:
            X_test_encoded[col] = 0
    
    if extra_cols:
        print(f"⚠️ 測試集多出 {len(extra_cols)} 個特徵，將移除")
        X_test_encoded = X_test_encoded.drop(columns=list(extra_cols))
    
    # 按照訓練時的欄位順序排列
    X_test_final = X_test_encoded[feature_columns]
    print(f"✔ 測試集特徵已對齊：{X_test_final.shape}")

    # === 5. 預測 ===
    y_pred = model.predict(X_test_final)
    print(f"✔ 預測完成！共 {len(y_pred)} 筆資料")
    
    # 顯示預測結果統計
    pred_counts = pd.Series(y_pred).value_counts()
    print(f"\n預測結果統計：")
    for label, count in pred_counts.items():
        print(f"  {label}: {count} 筆 ({count/len(y_pred)*100:.2f}%)")

    # === 6. 評估與輸出（如果有標籤就進行評估）===
    if has_labels and y_test_full is not None:
        # 清理標籤（移除 NaN）
        valid_mask = y_test_full.notna()
        y_test_clean = y_test_full[valid_mask].astype(str).str.strip()
        y_pred_clean = y_pred[valid_mask]
        
        if len(y_test_clean) == 0:
            print(f"\n⚠️ 沒有有效的標籤資料，無法計算評估指標\n")
            has_labels = False
        else:
            # === 評估模式：計算指標並生成報告 ===
            acc = accuracy_score(y_test_clean, y_pred_clean)
            print(f"\n{'='*50}")
            print(f"📊 測試集準確率：{acc:.4f} ({acc*100:.2f}%)")
            print(f"{'='*50}\n")
            print("→ 分類報告：")
            print(classification_report(y_test_clean, y_pred_clean, digits=4))

            # 取得所有類別標籤
            labels_sorted = sorted(set(y_test_clean) | set(y_pred_clean))
            cm = confusion_matrix(y_test_clean, y_pred_clean, labels=labels_sorted)

            base_name = test_path.stem
            output_root = Path.cwd() / "1020output_rf" / "combined" / "test_results" / base_name
            plots_dir = output_root / "plots"
            reports_dir = output_root / "reports"
            plots_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

            # === 7. 混淆矩陣圖 ===
            disp = ConfusionMatrixDisplay(cm, display_labels=labels_sorted)
            fig, ax = plt.subplots(figsize=(10, 8))
            disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
            plt.title(f'Confusion Matrix - {base_name}\nAccuracy: {acc:.4f}', fontsize=16)
            plt.tight_layout()

            img_path = plots_dir / f"{base_name}_confusion.png"
            plt.savefig(img_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"📊 混淆矩陣圖已儲存：{img_path}")

            # === 8. 輸出 Excel 報告 ===
            report_path = reports_dir / f"{base_name}_report.xlsx"
            cr_dict = classification_report(y_test_clean, y_pred_clean, output_dict=True)

            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                # 準確率
                pd.DataFrame({"Accuracy": [acc]}).to_excel(writer, sheet_name="Accuracy", index=False)
                
                # 分類報告
                pd.DataFrame(cr_dict).T.to_excel(writer, sheet_name="Classification_Report")
                
                # 混淆矩陣
                pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_excel(
                    writer, sheet_name="Confusion_Matrix"
                )
                
                # 預測結果（完整資料，包含原始特徵）
                results_df = pd.DataFrame({
                    "True_Label": y_test_full.values,
                    "Predicted_Label": y_pred,
                    "Correct": y_test_full.values == y_pred
                })
                # 加入原始特徵
                results_df = pd.concat([X_test_raw.reset_index(drop=True), results_df], axis=1)
                results_df.to_excel(writer, sheet_name="Predictions", index=False)

            print(f"✅ 評估報告已儲存：{report_path}")
            print(f"\n{'='*50}")
            print(f"✅ 測試完成！所有結果已儲存至：{output_root}")
            print(f"{'='*50}\n")
    
    if not has_labels:
        # === 純預測模式：只輸出預測結果 ===
        print(f"\n{'='*50}")
        print(f"📋 純預測模式：無法計算評估指標")
        print(f"{'='*50}\n")
        
        base_name = test_path.stem
        output_root = Path.cwd() / "1020output_rf" / "combined" / "predictions" / base_name
        output_root.mkdir(parents=True, exist_ok=True)

        # 輸出預測結果 Excel
        report_path = output_root / f"{base_name}_predictions.xlsx"
        results_df = pd.DataFrame({
            "Predicted_Label": y_pred
        })
        
        # 將原始資料也加入（方便查看）
        results_df = pd.concat([X_test_raw.reset_index(drop=True), results_df], axis=1)
        results_df.to_excel(report_path, index=False)
        
        print(f"✅ 預測結果已儲存：{report_path}")
        print(f"✅ 所有結果已儲存至：{output_root}\n")



# === 主程式 ===
if __name__ == "__main__":
    # ========== 修改這裡的路徑 ==========
    # 1. 模型路徑（訓練時產生的 .pkl 檔）
    model_path = Path("1020output_rf\\combined\\models\\rf_all_data_model.pkl")
    
    # 2. 測試集路徑（csv 或 xlsx 格式）
    test_path = Path("C:\\Users\\User\\Desktop\\已區分\\測試集\\1018_全數據_測試集.xlsx")
    # 3. 標籤欄位名稱（與訓練時一致）
    label_col = "label"
    
    # ====================================

    print(f"\n{'='*60}")
    print(f"  隨機森林模型測試")
    print(f"{'='*60}")
    print(f"模型路徑：{model_path}")
    print(f"測試集路徑：{test_path}")
    print(f"標籤欄位：{label_col}")
    print(f"說明：模型預測時不會看到標籤，預測完才比對")
    print(f"{'='*60}\n")

    evaluate_model(
        model_path=model_path,
        test_path=test_path,
        label_col=label_col,
    )
