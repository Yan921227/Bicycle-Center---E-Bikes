from pathlib import Path
import pandas as pd

# === 1. 指定要處理的 txt 資料夾 ===
input_folder = Path("C:\\Users\\User\\Desktop\\0809自行車騎乘數據\\未分類")

# === 2. 指定統一輸出資料夾 ===
output_folder = Path.cwd() / "0809output_converted"/"未分類"
output_folder.mkdir(parents=True, exist_ok=True)

# === 3. 批次轉換 .txt 檔為 .csv，輸出至 output_converted 資料夾 ===
for txt_path in input_folder.glob("*.txt"):
    try:
        # 嘗試讀取檔案
        df = pd.read_csv(txt_path, sep=",", header=0, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(txt_path, sep=",", header=0, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(txt_path, sep=",", header=0, encoding="big5")

    # 設定輸出檔案名稱（保留原名，改為 .csv）
    csv_name = txt_path.stem + ".csv"
    csv_path = output_folder / csv_name

    # 儲存為 CSV（含 BOM）
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✔ 轉檔完成：{txt_path.name} → {csv_name}")

print(f"\n✅ 所有 .txt ➜ .csv 轉換完成，已儲存至：{output_folder}")
