# 台北住宅房價預測

資料探勘課程期末專案中的台北住宅部分（負責人：蔡智翔）：以內政部實價登錄台北市 112–114 年、
共 12 季的成屋買賣資料，完成資料清理、特徵工程與 Random Forest 預測。

- 原始 57,589 筆、38 欄；移除「建築完成年月」為空者 12,732 筆（22.1%），保留 44,857 筆
- 40 個特徵（含 23 個備註關鍵詞特徵），目標為 log1p(總價元)
- RandomForestRegressor：100 棵樹、max_depth 10、min_samples_split 5、min_samples_leaf 2、random_state 42

## 資料下載

資料檔不放在 repo 內。來源：[內政部不動產交易實價查詢服務網](https://plvr.land.moi.gov.tw/DownloadOpenData)
→「非本期下載」，格式選 CSV，依季下載 112 年第 1 季至 114 年第 4 季。
每季只需要臺北市的 `A_lvr_land_A.csv`（不動產買賣）與 `A_lvr_land_B.csv`（預售屋買賣），
依季別放在 `taipei/` 底下（資料夾名稱須為 `年_季`，檔名大小寫須一致）：

```
taipei/
  112_1/A_lvr_land_A.csv
  112_1/A_lvr_land_B.csv
  ...
  114_4/A_lvr_land_A.csv
  114_4/A_lvr_land_B.csv
```

資料授權：政府資料開放授權條款－第 1 版。

## 執行

在 `taipei/` 目錄下依序執行：

```bash
python full_data_analyzer_with_plots.py   # 合併 12 季、篩選住家用 → 所有住家用資料_合併.csv
python impute_time_data_remove_null.py    # 時間欄位處理、移除建築完成年月空值 → 所有住家用資料_時間處理後.csv
python analyze_residential_data.py        # （選用）EDA 報告
python final_process.py                   # 特徵工程 → 所有住家用資料_特徵處理後.csv
python Random_Forest.py                   # Random Forest 建模與評估
```

EDA 報告與圖表見 `analysis/`。
