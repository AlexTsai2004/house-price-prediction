import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 讀取數據
df = pd.read_csv('所有住家用資料_時間處理後.csv')
print(f"原始資料形狀: {df.shape}")

# 2. 布林值轉換
def convert_to_bool(value):
    """將中文的'有'轉為True，'無'轉為False，其他值轉為NaN"""
    if pd.isna(value):
        return np.nan
    elif str(value).strip() == '有':
        return True
    elif str(value).strip() == '無':
        return False
    else:
        return value

# 轉換指定欄位
bool_columns = ['有無管理組織', '建物現況格局-隔間', '電梯']
for col in bool_columns:
    if col in df.columns:
        df[col] = df[col].apply(convert_to_bool)
        print(f"已轉換 {col} 為布林值")

# 3. 移除地址欄位
if '土地位置建物門牌' in df.columns:
    df = df.drop(columns=['土地位置建物門牌'])
    print(f"已移除 '土地位置建物門牌' 欄位")

# 4. 處理類別特徵 - 將空值填充為"一般"
category_columns = ['交易標的', '都市土地使用分區', '主要建材', '建物型態', '車位類別']
for col in category_columns:
    if col in df.columns:
        # 先將所有值轉為字串
        df[col] = df[col].astype(str)
        # 填充空值和空字串
        df[col] = df[col].replace(['nan', 'NaN', '', 'None', 'none', 'NaT'], '一般')
        print(f"已處理 {col} 的空值")

# 5. 鄉鎮市區Target Encoding
print("\n進行鄉鎮市區Target Encoding...")
# 檢查單價是否有缺失值
if df['單價元平方公尺'].isna().sum() > 0:
    print(f"單價元平方公尺有 {df['單價元平方公尺'].isna().sum()} 個缺失值，將用平均值填充")
    df['單價元平方公尺'] = df['單價元平方公尺'].fillna(df['單價元平方公尺'].mean())

# 計算每個鄉鎮市區的平均單價
region_mean_price = df.groupby('鄉鎮市區')['單價元平方公尺'].mean()
print(f"計算了 {len(region_mean_price)} 個鄉鎮市區的平均單價")

# 儲存原始的鄉鎮市區資料
df['鄉鎮市區_原始'] = df['鄉鎮市區'].copy()

# 使用target encoding將鄉鎮市區轉換為數值
df['鄉鎮市區_target_encoded'] = df['鄉鎮市區'].map(region_mean_price)

# 如果有缺失值，用整體平均單價填充
if df['鄉鎮市區_target_encoded'].isna().sum() > 0:
    overall_mean_price = df['單價元平方公尺'].mean()
    df['鄉鎮市區_target_encoded'] = df['鄉鎮市區_target_encoded'].fillna(overall_mean_price)
    print(f"使用整體平均單價 {overall_mean_price:.2f} 填充缺失值")

print(f"鄉鎮市區Target Encoding完成，範圍: {df['鄉鎮市區_target_encoded'].min():.2f} - {df['鄉鎮市區_target_encoded'].max():.2f}")

# 6. 備註關鍵詞特徵提取
print("\n進行備註關鍵詞特徵提取...")
# 定義關鍵詞列表
keywords = [
    '親友', '員工', '共有人', '特殊關係',  # 特殊交易關係
    '陽台外推', '夾層', '頂樓加蓋', '其他增建',  # 建築狀況
    '車位交易', '僅車位', '僅建物', '部分移轉',  # 交易類型
    '租約', '含租約', '裝潢費', '傢俱設備費',  # 附加條件
    '急買急賣', '債務', '債權債務', '畸零地',  # 其他特殊情況
    '二親等', '叔侄', '直系'  # 親屬關係
]

# 創建關鍵詞特徵
for keyword in keywords:
    df[f'備註_{keyword}'] = df['備註'].apply(
        lambda x: 1 if pd.notna(x) and keyword in str(x) else 0
    )

print(f"創建了 {len(keywords)} 個關鍵詞特徵")

# 7. 總價元對數轉換
print("\n進行總價元對數轉換...")
# 檢查總價是否有缺失值或非正值
if df['總價元'].isna().sum() > 0:
    print(f"總價元有 {df['總價元'].isna().sum()} 個缺失值，將用中位數填充")
    df['總價元'] = df['總價元'].fillna(df['總價元'].median())

# 檢查是否有非正值
if (df['總價元'] <= 0).any():
    print(f"總價元有 {(df['總價元'] <= 0).sum()} 個非正值，將用中位數替換")
    median_price = df['總價元'][df['總價元'] > 0].median()
    df.loc[df['總價元'] <= 0, '總價元'] = median_price

# 進行對數轉換
df['總價元_log'] = np.log1p(df['總價元'])  # 使用log1p避免對0取對數

print(f"對數轉換完成，總價元範圍: {df['總價元'].min():,.0f} - {df['總價元'].max():,.0f}")
print(f"對數轉換後範圍: {df['總價元_log'].min():.2f} - {df['總價元_log'].max():.2f}")

# 8. 可視化對數轉換效果
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 原始總價分布
axes[0].hist(df['總價元'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('原始總價元分布')
axes[0].set_xlabel('總價元')
axes[0].set_ylabel('頻率')

# 對數轉換後分布
axes[1].hist(df['總價元_log'], bins=50, edgecolor='black', alpha=0.7)
axes[1].set_title('對數轉換後總價元分布')
axes[1].set_xlabel('log(總價元)')
axes[1].set_ylabel('頻率')

plt.tight_layout()
plt.show()

# 9. 統計資訊摘要
print("\n" + "="*50)
print("特徵工程完成摘要")
print("="*50)
print(f"最終資料形狀: {df.shape}")
print(f"總欄位數: {len(df.columns)}")

# 分類統計
num_features = df.select_dtypes(include=[np.number]).shape[1]
cat_features = df.select_dtypes(include=['object', 'category']).shape[1]
bool_features = df.select_dtypes(include=['bool']).shape[1]

print(f"\n特徵類型統計:")
print(f"  數值特徵: {num_features}")
print(f"  類別特徵: {cat_features}")
print(f"  布林特徵: {bool_features}")

# 顯示新增的特徵
print(f"\n新增的特徵:")
new_features = ['鄉鎮市區_target_encoded', '總價元_log'] + [f'備註_{kw}' for kw in keywords]
for feature in new_features:
    if feature in df.columns:
        print(f"  ✓ {feature}")

# 檢查目標變數
print(f"\n目標變數資訊:")
print(f"  原始總價元 - 平均值: {df['總價元'].mean():,.0f}")
print(f"  對數總價元 - 平均值: {df['總價元_log'].mean():.2f}")

# 儲存處理後的資料
output_file = '所有住家用資料_特徵處理後.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n處理後的資料已儲存為: {output_file}")

# 顯示前幾行資料
print("\n處理後資料前3行:")
display_columns = ['鄉鎮市區', '鄉鎮市區_target_encoded', '總價元', '總價元_log'] + \
                  [f'備註_{kw}' for kw in keywords[:3]] + \
                  ['有無管理組織', '電梯']
display_columns = [col for col in display_columns if col in df.columns]
print(df[display_columns].head(3))