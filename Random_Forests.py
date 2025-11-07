# -*- coding: utf-8 -*-
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
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


# ========== 這段是「額外輸出 STM32 用 .h」的關鍵函式 ==========
def export_rf_to_c_header(rf, feature_names, class_names, out_path: Path, model_id: str = "RF_MODEL"):
    """
    將 sklearn RandomForestClassifier 匯出為 STM32 可用的 .h。
    - rf            : 已訓練好的 RandomForestClassifier
    - feature_names : 訓練後（One-Hot 後）的欄位順序
    - class_names   : 類別名稱（字串列表），若未對齊會自動改用 rf.classes_
    - out_path      : 目標 .h 路徑
    - model_id      : C 符號前綴（將自動過濾為 A-Z0-9_，並轉大寫）
    產出內容：
      - <model>.h              ：可直接在 MCU 端 #include 並呼叫 <MODEL>_predict(x)
      - <model>.features.txt   ：特徵順序對照，MCU 端組 x[] 要完全一致
      - <model>.classes.txt    ：類別名稱對照
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not hasattr(rf, "estimators_"):
        raise ValueError("模型尚未訓練（缺少 estimators_）。")
    if len(getattr(rf, "classes_", [])) != len(class_names):
        class_names = [str(c) for c in rf.classes_]

    # 乾淨的 C 符號前綴
    model_id = re.sub(r"\W+", "_", str(model_id)).upper()
    if not model_id or model_id[0].isdigit():
        model_id = "RF_MODEL"

    n_trees     = len(rf.estimators_)
    n_features  = len(feature_names)
    n_classes   = len(class_names)
    node_counts = [est.tree_.node_count for est in rf.estimators_]
    max_nodes   = max(node_counts) if node_counts else 0

    # 若節點或特徵過多，自動切換 int32，以免 int16 溢位
    use_int32 = (n_features > 32767) or (max_nodes > 32767)

    def f32(v: float) -> str:
        if v is None or not np.isfinite(v):
            return "0.0f"
        s = f"{float(v):.9g}"
    # 如果字串中沒有小數點，就補上 .0f；否則補 f
        if '.' not in s and 'e' not in s and 'E' not in s:
            return s + ".0f"
        return s + "f"



    def c_str(s: str) -> str:
        return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = []
    lines.append("// -----------------------------------------------------------------------------")
    lines.append("//  Auto-generated RandomForest header for STM32")
    lines.append(f"//  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("//  NOTE: 請用與訓練時 One-Hot 後完全相同的特徵順序與長度。")
    lines.append("// -----------------------------------------------------------------------------\n")

    guard = f"{model_id}_H"
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}\n")
    lines.append("#include <stdint.h>\n")
    lines.append("#ifdef __cplusplus")
    lines.append('extern "C" {')
    lines.append("#endif\n")

    lines.append(f"#define {model_id}_N_TREES    {n_trees}")
    lines.append(f"#define {model_id}_N_FEATURES {n_features}")
    lines.append(f"#define {model_id}_N_CLASSES  {n_classes}")
    if use_int32:
        lines.append(f"#define {model_id}_USE_INT32 1")
    lines.append(f"#define {model_id}_LEAF_INDEX (-1)\n")

    idx_t = "int32_t" if use_int32 else "int16_t"
    lines.append(f"typedef {idx_t} {model_id}_idx_t;")
    lines.append("typedef struct {")
    lines.append(f"  {model_id}_idx_t feature;   // 內部特徵索引（非葉）")
    lines.append( "  float          threshold;  // 分裂門檻（非葉）")
    lines.append(f"  {model_id}_idx_t left;      // 左子；葉節點為 -1")
    lines.append(f"  {model_id}_idx_t right;     // 右子；葉節點為 -1")
    lines.append(f"  {model_id}_idx_t value;     // 葉節點的類別索引")
    lines.append(f"}} {model_id}_Node;\n")

    # 若需要把名稱也編進 MCU（省略可省空間）：編譯加 -D{model_id}_EMBED_STRINGS
    lines.append(f"#ifdef {model_id}_EMBED_STRINGS")
    feat_str = ", ".join(c_str(n) for n in feature_names)
    cls_str  = ", ".join(c_str(c) for c in class_names)
    lines.append(f"static const char* {model_id}_FEATURE_NAMES[{n_features}] = {{ {feat_str} }};")
    lines.append(f"static const char* {model_id}_CLASS_NAMES[{n_classes}]   = {{ {cls_str}  }};")
    lines.append("#endif\n")

    tree_size_list = []
    for ti, est in enumerate(rf.estimators_):
        t = est.tree_
        left      = t.children_left.tolist()
        right     = t.children_right.tolist()
        feature   = t.feature.tolist()
        threshold = t.threshold.tolist()
        values    = t.value.squeeze(axis=1) if t.value.ndim == 3 else t.value
        if values.ndim == 1:
            values = np.expand_dims(values, axis=1)

        n_nodes = t.node_count
        tree_size_list.append(n_nodes)

        lines.append(f"static const {model_id}_Node {model_id}_TREE_{ti}[] = {{")
        rows = []
        for i in range(n_nodes):
            is_leaf = (left[i] == -1 and right[i] == -1)
            pred_cls = int(np.argmax(values[i])) if is_leaf else 0
            thr = 0.0 if is_leaf else float(threshold[i])
            fi  = -1 if is_leaf else int(feature[i])
            rows.append(
                f"  {{ ({model_id}_idx_t){fi}, {f32(thr)}, "
                f"({model_id}_idx_t){left[i]}, ({model_id}_idx_t){right[i]}, "
                f"({model_id}_idx_t){pred_cls} }}"
            )
        lines.append(",\n".join(rows))
        lines.append("};\n")

    ptrs  = ", ".join([f"{model_id}_TREE_{ti}" for ti in range(n_trees)])
    sizes = ", ".join([str(s) for s in tree_size_list])
    lines.append(f"static const {model_id}_Node* const {model_id}_FOREST[{n_trees}] = {{ {ptrs} }};")
    lines.append(f"static const {model_id}_idx_t {model_id}_TREE_SIZES[{n_trees}] = {{ {sizes} }};\n")

    # 內聯推論函式（投票）
    lines.append(f"static inline int {model_id}_predict(const float* x) {{")
    lines.append( f"  int votes[{model_id}_N_CLASSES] = {{0}};")
    lines.append( f"  for (int t = 0; t < {model_id}_N_TREES; ++t) {{")
    lines.append( f"    const {model_id}_Node* nodes = {model_id}_FOREST[t];")
    lines.append(  f"    {model_id}_idx_t idx = 0;")
    lines.append(  "    while (1) {")
    lines.append(  f"      const {model_id}_Node* n = &nodes[idx];")
    lines.append(  f"      if (n->left == {model_id}_LEAF_INDEX && n->right == {model_id}_LEAF_INDEX) {{ votes[n->value]++; break; }}")
    lines.append(  "      const float v = x[n->feature];")
    lines.append(  "      idx = (v <= n->threshold) ? n->left : n->right;")
    lines.append(  "    }")
    lines.append(  "  }")
    lines.append(  "  int best = 0; int bestv = votes[0];")
    lines.append(  f"  for (int c = 1; c < {model_id}_N_CLASSES; ++c) {{ if (votes[c] > bestv) {{ bestv = votes[c]; best = c; }} }}")
    lines.append(  "  return best;")
    lines.append(  "}\n")

    # （選配）輸出投票比例（簡易機率）
    lines.append(f"static inline void {model_id}_predict_proba(const float* x, float out[{model_id}_N_CLASSES]) {{")
    lines.append( f"  int votes[{model_id}_N_CLASSES] = {{0}};")
    lines.append( f"  for (int t = 0; t < {model_id}_N_TREES; ++t) {{")
    lines.append( f"    const {model_id}_Node* nodes = {model_id}_FOREST[t];")
    lines.append(  f"    {model_id}_idx_t idx = 0;")
    lines.append(  "    while (1) {")
    lines.append(  f"      const {model_id}_Node* n = &nodes[idx];")
    lines.append(  f"      if (n->left == {model_id}_LEAF_INDEX && n->right == {model_id}_LEAF_INDEX) {{ votes[n->value]++; break; }}")
    lines.append(  "      const float v = x[n->feature];")
    lines.append(  "      idx = (v <= n->threshold) ? n->left : n->right;")
    lines.append(  "    }")
    lines.append(  "  }")
    lines.append( f"  for (int c = 0; c < {model_id}_N_CLASSES; ++c) out[c] = (float)votes[c] / (float){model_id}_N_TREES;")
    lines.append(  "}\n")

    lines.append("#ifdef __cplusplus")
    lines.append("} // extern \"C\"")
    lines.append("#endif")
    lines.append(f"#endif // {guard}\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"🧾 已輸出 C header：{out_path}  （節點總數：{sum(node_counts)}，樹數：{n_trees}）")

    # 同步輸出對照清單（MCU 端組裝特徵時要對齊）
    out_path.with_suffix(".features.txt").write_text("\n".join(map(str, feature_names)), encoding="utf-8")
    out_path.with_suffix(".classes.txt").write_text("\n".join(map(str, class_names)), encoding="utf-8")
# ==========（關鍵函式到此）==========


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
        'n_estimators':      [40],
        'max_depth':         [10],
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
    output_root = Path.cwd() / "1020output_rf" / "combined"
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
    png_path = images_dir / "全數據_混淆矩陣.png"
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

    # 10) 模型 + 欄位順序 一起儲存（.pkl）
    artifact = {
        "model": best_rf,   
        "feature_columns": list(X.columns),
        "class_names": list(best_rf.classes_)
    }
    model_path = models_dir / "rf_all_data_model.pkl"  # 保留 .pkl 供 Python 使用
    joblib.dump(artifact, model_path)
    print(f"🧠 模型已儲存：{model_path}")

    # 11) 額外輸出 STM32 用的 .h
    header_path = models_dir / "rf_all_data.h"  # C Header
    export_rf_to_c_header(
        best_rf,
        feature_names=list(X.columns),
        class_names=list(best_rf.classes_),
        out_path=header_path,
        model_id="rf_all_data"  # C 符號前綴（宏與函式名會用到）
    )
    print("✅ 已完成：.pkl、.h、.features.txt、.classes.txt 全部輸出完成。")


# ——— 這裡「寫死」你的檔案路徑與標籤欄位 ———
if __name__ == "__main__":
    # 依你的實際路徑修改（Windows 範例：請確認路徑存在）
    DATA_FILE = "C:\\Users\\User\\Desktop\\已區分\\訓練集\\1018_全數據_訓練.xlsx"
    TARGET    = "label"   # 如果你的標籤欄位不是 label，改這裡
    train_and_save_confusion(DATA_FILE, TARGET)
