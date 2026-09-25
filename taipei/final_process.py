"""
特徵處理（修正版）

與課程繳交版本（`original` 分支）的差異：
1. 移除「鄉鎮市區 Target Encoding」。
   原版在切分訓練／測試集之前，用全部資料計算各行政區的平均單價，
   會讓測試集的價格資訊進入特徵（資料洩漏）。
   修正版改在 Random_Forest.py 中、切分之後只用訓練集計算。
2. 圖表改存成檔案，不再呼叫 plt.show()（避免批次執行時卡住）。
其餘步驟（布林轉換、類別空值、23 個備註關鍵詞、總價 log1p）與原版相同。
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
IN_FILE = HERE / '所有住家用資料_時間處理後.csv'
OUT_FILE = HERE / '所有住家用資料_特徵處理後.csv'
RESULTS = HERE / 'results'
RESULTS.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

KEYWORDS = [
    '親友', '員工', '共有人', '特殊關係',          # 特殊交易關係
    '陽台外推', '夾層', '頂樓加蓋', '其他增建',     # 建築狀況
    '車位交易', '僅車位', '僅建物', '部分移轉',     # 交易類型
    '租約', '含租約', '裝潢費', '傢俱設備費',       # 附加條件
    '急買急賣', '債務', '債權債務', '畸零地',       # 其他特殊情況
    '二親等', '叔侄', '直系',                      # 親屬關係
]
BOOL_COLUMNS = ['有無管理組織', '建物現況格局-隔間', '電梯']
CATEGORY_COLUMNS = ['交易標的', '都市土地使用分區', '主要建材', '建物型態', '車位類別']


def convert_to_bool(value):
    if pd.isna(value):
        return np.nan
    if str(value).strip() == '有':
        return True
    if str(value).strip() == '無':
        return False
    return value


def main():
    df = pd.read_csv(IN_FILE, low_memory=False)
    print(f'原始資料形狀: {df.shape}')

    for col in BOOL_COLUMNS:
        df[col] = df[col].apply(convert_to_bool)

    df = df.drop(columns=['土地位置建物門牌'])

    for col in CATEGORY_COLUMNS:
        df[col] = df[col].astype(str).replace(['nan', 'NaN', '', 'None', 'none', 'NaT'], '一般')

    for kw in KEYWORDS:
        df[f'備註_{kw}'] = df['備註'].apply(lambda x, k=kw: 1 if pd.notna(x) and k in str(x) else 0)
    print(f'建立 {len(KEYWORDS)} 個備註關鍵詞特徵')

    if df['總價元'].isna().any():
        df['總價元'] = df['總價元'].fillna(df['總價元'].median())
    if (df['總價元'] <= 0).any():
        df.loc[df['總價元'] <= 0, '總價元'] = df.loc[df['總價元'] > 0, '總價元'].median()
    df['總價元_log'] = np.log1p(df['總價元'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df['總價元'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('原始總價元分布')
    axes[1].hist(df['總價元_log'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_title('log1p(總價元) 分布')
    plt.tight_layout()
    plt.savefig(RESULTS / '總價分布_log轉換.png', dpi=150)
    plt.close(fig)

    df.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
    print(f'已儲存: {OUT_FILE.name}，形狀 {df.shape}')


if __name__ == '__main__':
    main()
