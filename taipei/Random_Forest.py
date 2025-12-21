import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False

# 讀取處理後的資料
df = pd.read_csv('所有住家用資料_特徵處理後.csv')
print(f"資料形狀: {df.shape}")

# 1. 準備特徵和目標變數
# 目標變數：對數轉換後的總價
y = df['總價元_log']

# 特徵選擇：使用您指定的所有特徵
# 先列出所有可能的特徵
all_features = [
    # 數值特徵
    '主建物面積',
    '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛',
    '車位移轉總面積平方公尺',  '附屬建物面積', 
    '陽台面積', '屋齡_年', '鄉鎮市區_target_encoded',
    
    # 布林特徵（已轉換）
    '有無管理組織', '建物現況格局-隔間', '電梯',
    
    # 備註關鍵詞特徵
    '備註_親友', '備註_員工', '備註_共有人', '備註_特殊關係',
    '備註_陽台外推', '備註_夾層', '備註_頂樓加蓋', '備註_其他增建',
    '備註_車位交易', '備註_僅車位', '備註_僅建物', '備註_部分移轉',
    '備註_租約', '備註_含租約', '備註_裝潢費', '備註_傢俱設備費',
    '備註_急買急賣', '備註_債務', '備註_債權債務', '備註_畸零地',
    '備註_二親等', '備註_叔侄', '備註_直系',
    
    # 類別特徵（需要編碼）
    '交易標的', '都市土地使用分區', '主要建材', '建物型態', '車位類別'
]

# 只選擇存在於資料中的特徵
available_features = [col for col in all_features if col in df.columns]
print(f"可用的特徵數量: {len(available_features)}")

# 分離數值/布林特徵和類別特徵
numeric_bool_features = [
    col for col in available_features 
    if df[col].dtype in ['int64', 'float64', 'bool']
]

categorical_features = [
    col for col in available_features 
    if df[col].dtype == 'object'
]

print(f"數值/布林特徵: {len(numeric_bool_features)} 個")
print(f"類別特徵: {len(categorical_features)} 個")

# 2. 處理類別特徵 - 使用Label Encoding
X_numeric = df[numeric_bool_features].copy()

# 對布林特徵轉換為0/1
bool_columns = ['有無管理組織', '建物現況格局-隔間', '電梯']
for col in bool_columns:
    if col in X_numeric.columns:
        X_numeric[col] = X_numeric[col].astype(int)

# 處理類別特徵
X_encoded = pd.DataFrame()
label_encoders = {}

for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        # 處理缺失值
        temp_col = df[col].fillna('一般')
        X_encoded[col] = le.fit_transform(temp_col)
        label_encoders[col] = le
        print(f"已編碼 {col}: {len(le.classes_)} 個類別")

# 合併所有特徵
X = pd.concat([X_numeric, X_encoded], axis=1)

print(f"\n最終特徵矩陣形狀: {X.shape}")
print(f"目標變數形狀: {y.shape}")

# 3. 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n訓練集大小: {X_train.shape}")
print(f"測試集大小: {X_test.shape}")

# 4. 建立和訓練Random Forest模型
print("\n訓練Random Forest模型...")
rf_model = RandomForestRegressor(
    n_estimators=100,  # 樹的數量
    max_depth=10,      # 最大深度
    min_samples_split=5,  # 最小分割樣本數
    min_samples_leaf=2,   # 葉節點最小樣本數
    random_state=42,
    n_jobs=-1  # 使用所有CPU核心
)

rf_model.fit(X_train, y_train)
print("模型訓練完成！")

# 5. 模型評估
print("\n模型評估:")
# 預測
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# 將預測值轉回原始尺度（對數逆轉換）
y_train_original = np.expm1(y_train)
y_test_original = np.expm1(y_test)
y_pred_train_original = np.expm1(y_pred_train)
y_pred_test_original = np.expm1(y_pred_test)

# 計算指標（對數尺度）
print("對數尺度指標:")
print(f"訓練集 R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"測試集 R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"訓練集 RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"測試集 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")

print("\n原始尺度指標:")
print(f"訓練集 R²: {r2_score(y_train_original, y_pred_train_original):.4f}")
print(f"測試集 R²: {r2_score(y_test_original, y_pred_test_original):.4f}")
print(f"訓練集 MAE: {mean_absolute_error(y_train_original, y_pred_train_original):,.0f} 元")
print(f"測試集 MAE: {mean_absolute_error(y_test_original, y_pred_test_original):,.0f} 元")
print(f"訓練集 RMSE: {np.sqrt(mean_squared_error(y_train_original, y_pred_train_original)):,.0f} 元")
print(f"測試集 RMSE: {np.sqrt(mean_squared_error(y_test_original, y_pred_test_original)):,.0f} 元")

# 6. 特徵重要性分析
print("\n特徵重要性分析 (Top 20):")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20))

# 7. 可視化結果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 特徵重要性
axes[0, 0].barh(feature_importance['feature'].head(15)[::-1], 
                feature_importance['importance'].head(15)[::-1])
axes[0, 0].set_title('Top 15 特徵重要性')
axes[0, 0].set_xlabel('重要性')

# 2. 預測 vs 實際值 (對數尺度)
axes[0, 1].scatter(y_test, y_pred_test, alpha=0.5)
axes[0, 1].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('實際值 (log)')
axes[0, 1].set_ylabel('預測值 (log)')
axes[0, 1].set_title(f'預測 vs 實際 (測試集)\nR² = {r2_score(y_test, y_pred_test):.4f}')

# 3. 殘差圖 (對數尺度)
residuals = y_test - y_pred_test
axes[0, 2].scatter(y_pred_test, residuals, alpha=0.5)
axes[0, 2].axhline(y=0, color='r', linestyle='--')
axes[0, 2].set_xlabel('預測值 (log)')
axes[0, 2].set_ylabel('殘差 (log)')
axes[0, 2].set_title('殘差圖')

# 4. 預測 vs 實際值 (原始尺度)
axes[1, 0].scatter(y_test_original, y_pred_test_original, alpha=0.5)
axes[1, 0].plot([y_test_original.min(), y_test_original.max()], 
                [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('實際總價 (元)')
axes[1, 0].set_ylabel('預測總價 (元)')
axes[1, 0].set_title(f'預測 vs 實際 (原始尺度)\nR² = {r2_score(y_test_original, y_pred_test_original):.4f}')

# 5. 誤差分布
error_percentage = ((y_pred_test_original - y_test_original) / y_test_original) * 100
axes[1, 1].hist(error_percentage, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('預測誤差百分比 (%)')
axes[1, 1].set_ylabel('頻率')
axes[1, 1].set_title(f'預測誤差分布\n平均誤差: {error_percentage.mean():.1f}%')

# 6. 樹的數量與性能關係
train_scores = []
test_scores = []
n_trees_range = range(10, 201, 10)

for n_trees in n_trees_range:
    rf_temp = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_temp.fit(X_train, y_train)
    train_scores.append(rf_temp.score(X_train, y_train))
    test_scores.append(rf_temp.score(X_test, y_test))

axes[1, 2].plot(n_trees_range, train_scores, label='訓練集')
axes[1, 2].plot(n_trees_range, test_scores, label='測試集')
axes[1, 2].set_xlabel('樹的數量')
axes[1, 2].set_ylabel('R²分數')
axes[1, 2].set_title('模型性能 vs 樹的數量')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 8. 模型診斷
print("\n模型診斷:")
print(f"訓練集樣本數: {len(X_train)}")
print(f"測試集樣本數: {len(X_test)}")
print(f"特徵數量: {X.shape[1]}")

# 檢查過擬合
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
overfitting_gap = train_r2 - test_r2
print(f"過擬合差距 (訓練R² - 測試R²): {overfitting_gap:.4f}")

if overfitting_gap > 0.1:
    print("警告: 模型可能有過擬合問題，建議:")
    print("  1. 增加樹的最大深度限制")
    print("  2. 增加min_samples_split和min_samples_leaf")
    print("  3. 使用更多訓練數據")
else:
    print("過擬合情況在可接受範圍內")

# 9. 預測示例
print("\n預測示例 (測試集前5筆):")
sample_indices = X_test.index[:5]
for idx in sample_indices:
    actual_price = np.expm1(y.loc[idx])
    predicted_price = np.expm1(rf_model.predict(X.loc[[idx]]))[0]
    error = predicted_price - actual_price
    error_percent = (error / actual_price) * 100
    
    print(f"實際: {actual_price:,.0f}元, 預測: {predicted_price:,.0f}元, "
          f"誤差: {error:,.0f}元 ({error_percent:+.1f}%)")

# 10. 儲存模型結果
results_df = pd.DataFrame({
    '實際值_原始': y_test_original.values,
    '預測值_原始': y_pred_test_original,
    '實際值_log': y_test.values,
    '預測值_log': y_pred_test,
    '誤差_原始': y_pred_test_original - y_test_original.values,
    '誤差百分比': error_percentage
}, index=y_test.index)

results_df.to_csv('random_forest_predictions.csv', encoding='utf-8-sig')
print("\n預測結果已儲存為 'random_forest_predictions.csv'")