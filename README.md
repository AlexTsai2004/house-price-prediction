# House Price Prediction：Kaggle Ames 與台北實價登錄

資料探勘課程期末專案（2025 年 12 月，三人團隊）。專案分為兩部分：

1. **Ames（Kaggle House Prices）**：以 Random Forest 特徵重要度挑選 Top-K 特徵，
   再用 TensorFlow Decision Forests 訓練 Random Forest 與 Gradient Boosted Trees，
   並測試整合特徵（TotalQualityIndex）的效果。
2. **台北住宅（內政部不動產實價登錄）**：把相同的建模流程套用到台北市 112–114 年、
   共 12 季的成屋買賣資料，完成資料清理、特徵工程與 Random Forest 預測。

## 分工

| 成員 | 負責部分 | 主要檔案 |
|---|---|---|
| 黃柏碩 | Ames 主流程：EDA、特徵重要度與 Top-K 搜尋、TF-DF RF／GBT 模型、Kaggle submission | `ames/house_price.ipynb` |
| 黃子宸 | Ames 特徵整合：設計 TotalQualityIndex 等整合特徵並重新評估 | `ames/feature_integration.ipynb` |
| 蔡智翔 | 台北住宅：實價登錄資料合併與清理、特徵工程、Random Forest 建模與評估 | `taipei/` |

## 目錄

```
ames/
  house_price.ipynb            Ames 主流程
  feature_integration.ipynb    加入整合特徵的版本
  data/README.md               Kaggle 資料下載說明
taipei/
  README.md                    資料下載與執行步驟
  full_data_analyzer_with_plots.py   合併 12 季資料、篩選住家用
  impute_time_data_remove_null.py    時間欄位處理、移除建築完成年月空值
  analyze_residential_data.py        EDA
  final_process.py                   特徵工程
  Random_Forest.py                   Random Forest 建模與評估
  analysis/                          EDA 報告與圖表
```

## 結果摘要

### Ames（validation set：固定 seed 的 80/20 隨機切分，訓練 1,142 筆）

| 模型 | 原始特徵（`house_price.ipynb`） | 加入整合特徵（`feature_integration.ipynb`） |
|---|---|---|
| Random Forest，最佳 Top-K（K=40） | RMSE 26,779 | RMSE 26,265 |
| Random Forest，Top 50 | RMSE 27,293 | RMSE 26,425 |
| Gradient Boosted Trees，Top 50 | **RMSE 22,937** | RMSE 23,907 |

整合特徵讓 Random Forest 的 RMSE 下降約 2–3%，但對 GBT 沒有幫助。

### 台北住宅（80/20 隨機切分，訓練 35,885 筆、測試 8,972 筆）

| 指標 | 測試集 |
|---|---|
| R²（log 尺度） | 0.8898 |
| R²（原始價格） | 0.8654 |

## 環境

```bash
pip install -r requirements.txt
```

Ames notebook 原本在 Google Colab 執行（第一格會安裝 `tensorflow_decision_forests`）；
台北部分在本機以 Python 3.11 執行。資料取得方式見 `ames/data/README.md` 與 `taipei/README.md`。
