import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# 設定中文字型
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def load_and_prepare_data():
    """載入合併的資料"""
    current_dir = Path(__file__).parent
    data_file = current_dir / "所有住家用資料_合併.csv"
    
    if not data_file.exists():
        print(f"錯誤：找不到資料檔案 {data_file}")
        print("請先執行 collect_residential_data.py 生成合併資料")
        return None
    
    print(f"載入資料檔案: {data_file}")
    df = pd.read_csv(data_file, low_memory=False)
    print(f"資料維度: {df.shape[0]} 行 × {df.shape[1]} 列")
    
    return df

def analyze_missing_values(df):
    """分析缺失值"""
    print("\n" + "="*60)
    print("缺失值分析")
    print("="*60)
    
    missing_data = pd.DataFrame({
        '欄位名稱': df.columns,
        '缺失數量': df.isnull().sum().values,
        '缺失比例(%)': (df.isnull().sum().values / len(df) * 100).round(2),
    })
    
    missing_sorted = missing_data.sort_values('缺失比例(%)', ascending=False)
    
    print("\n缺失值最多的前20個欄位:")
    print("-" * 80)
    for idx, row in missing_sorted.head(20).iterrows():
        print(f"{row['欄位名稱'][:30]:<30} {row['缺失數量']:<10,} {row['缺失比例(%)']:<12.1f}")
    
    # 視覺化
    plt.figure(figsize=(15, 8))
    
    # 缺失值比例分佈
    plt.subplot(1, 2, 1)
    missing_percentage = (df.isnull().sum() / len(df) * 100)
    missing_percentage = missing_percentage[missing_percentage > 0]
    
    if len(missing_percentage) > 0:
        plt.hist(missing_percentage, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('缺失比例 (%)')
        plt.ylabel('欄位數量')
        plt.title('缺失值比例分佈')
        plt.grid(True, alpha=0.3)
    
    # 缺失值最多的前10個欄位
    plt.subplot(1, 2, 2)
    top_missing = missing_sorted.head(10)
    
    if len(top_missing) > 0:
        plt.barh(range(len(top_missing)), top_missing['缺失比例(%)'].values[::-1])
        plt.yticks(range(len(top_missing)), top_missing['欄位名稱'].values[::-1])
        plt.xlabel('缺失比例 (%)')
        plt.title('缺失值最多的前10個欄位')
    
    plt.tight_layout()
    plt.savefig('缺失值分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return missing_sorted

def analyze_numerical_features(df):
    """分析數值型特徵"""
    print("\n" + "="*60)
    print("數值型特徵分析")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"找到 {len(numeric_cols)} 個數值型欄位")
    
    if not numeric_cols:
        return None
    
    # 基本統計
    numeric_stats = df[numeric_cols].describe().T
    numeric_stats['缺失值數量'] = df[numeric_cols].isnull().sum()
    
    print("\n前10個數值型欄位統計:")
    print("-" * 100)
    for col in numeric_cols[:10]:
        if col in numeric_stats.index:
            stats = numeric_stats.loc[col]
            print(f"{col[:30]:<30} 平均:{stats['mean']:10.2f} 中位數:{stats['50%']:10.2f}")
    
    # 視覺化
    plot_cols = min(6, len(numeric_cols))
    if plot_cols > 0:
        plt.figure(figsize=(15, 10))
        
        for i, col in enumerate(numeric_cols[:plot_cols], 1):
            plt.subplot(2, 3, i)
            if df[col].nunique() > 1:
                plt.hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
                plt.title(f'{col[:20]}', fontsize=10)
                plt.xlabel('數值')
                plt.grid(True, alpha=0.3)
        
        plt.suptitle('數值型特徵分佈', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('數值型特徵分佈.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return numeric_stats

def analyze_categorical_features(df):
    """分析類別型特徵"""
    print("\n" + "="*60)
    print("類別型特徵分析")
    print("="*60)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"找到 {len(categorical_cols)} 個類別型欄位")
    
    if not categorical_cols:
        return None
    
    # 重要類別欄位
    important_cat_cols = []
    for col in categorical_cols:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['行政區', 'district', '型態', '用途']):
            if df[col].nunique() <= 50:
                important_cat_cols.append(col)
    
    print(f"\n分析 {len(important_cat_cols)} 個重要類別型欄位")
    
    # 視覺化
    if important_cat_cols:
        plot_cols = min(4, len(important_cat_cols))
        plt.figure(figsize=(15, 10))
        
        for i, col in enumerate(important_cat_cols[:plot_cols], 1):
            plt.subplot(2, 2, i)
            
            value_counts = df[col].value_counts().head(10)
            
            if len(value_counts) > 0:
                plt.bar(range(len(value_counts)), value_counts.values, alpha=0.7)
                plt.xticks(range(len(value_counts)), 
                          [str(v)[:15] for v in value_counts.index], 
                          rotation=45, ha='right')
                plt.title(f'{col[:20]} (Top 10)', fontsize=10)
                plt.ylabel('數量')
                plt.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('重要類別型特徵分佈', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('類別型特徵分佈.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return pd.DataFrame({'欄位名稱': important_cat_cols})

def analyze_price_distribution(df):
    """分析價格分佈"""
    print("\n" + "="*60)
    print("價格分佈分析")
    print("="*60)
    
    # 尋找價格欄位
    price_columns = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['價格', 'price', '單價', '總價']):
            price_columns.append(col)
    
    if not price_columns:
        print("沒有找到價格相關欄位")
        return None
    
    print(f"找到價格欄位: {price_columns}")
    
    # 分析第一個價格欄位
    price_col = price_columns[0]
    price_data = pd.to_numeric(df[price_col], errors='coerce')
    
    if price_data.notna().sum() > 0:
        print(f"\n{price_col} 基本統計:")
        print(f"有效資料: {price_data.notna().sum():,d} 筆")
        print(f"平均值: {price_data.mean():,.0f}")
        print(f"中位數: {price_data.median():,.0f}")
        
        # 視覺化
        plt.figure(figsize=(12, 8))
        
        # 直方圖
        plt.subplot(2, 2, 1)
        plt.hist(price_data.dropna(), bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{price_col} 分佈')
        plt.xlabel('價格')
        plt.grid(True, alpha=0.3)
        
        # 箱形圖
        plt.subplot(2, 2, 2)
        plt.boxplot(price_data.dropna(), vert=True)
        plt.title(f'{price_col} 箱形圖')
        plt.ylabel('價格')
        plt.grid(True, alpha=0.3)
        
        # 價格區間
        plt.subplot(2, 2, 3)
        bins = [0, 10000000, 20000000, 50000000, 100000000, float('inf')]
        labels = ['1000萬↓', '1000-2000萬', '2000-5000萬', '5000萬-1億', '1億↑']
        
        price_intervals = pd.cut(price_data, bins=bins, labels=labels, right=False)
        interval_counts = price_intervals.value_counts()
        
        plt.pie(interval_counts.values, labels=interval_counts.index, autopct='%1.1f%%')
        plt.title('價格區間分佈')
        
        plt.tight_layout()
        plt.savefig('價格分佈分析.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return price_data

def analyze_time_series_fixed(df):
    """修正版的時間序列分析"""
    print("\n" + "="*60)
    print("時間序列分析")
    print("="*60)
    
    # 尋找時間欄位
    time_columns = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['日期', 'date', '年月日']):
            time_columns.append(col)
    
    if not time_columns:
        print("沒有找到時間相關欄位")
        return
    
    print(f"找到時間欄位: {time_columns}")
    
    # 使用第一個時間欄位
    time_col = time_columns[0]
    
    # 簡單提取年份和月份
    def extract_year_month(time_val):
        if pd.isna(time_val):
            return None, None
        
        time_str = str(time_val).strip()
        if len(time_str) >= 6:
            try:
                # 假設格式為YYYMMDD或YYMMDD
                if time_str[0] == '1':  # 民國1xx年
                    year = int(time_str[:3])
                    month = int(time_str[3:5]) if len(time_str) >= 5 else None
                else:  # 民國xx年
                    year = int(time_str[:2])
                    month = int(time_str[2:4]) if len(time_str) >= 4 else None
                
                # 驗證合理性
                if 100 <= year <= 150 and 1 <= month <= 12:
                    return year, month
                elif 0 <= year <= 99 and 1 <= month <= 12:
                    return year, month
            except:
                pass
        
        return None, None
    
    # 提取時間
    years = []
    months = []
    
    for time_val in df[time_col]:
        year, month = extract_year_month(time_val)
        years.append(year)
        months.append(month)
    
    df_temp = df.copy()
    df_temp['年份'] = years
    df_temp['月份'] = months
    
    valid_data = df_temp[df_temp['年份'].notna() & df_temp['月份'].notna()]
    
    if len(valid_data) > 0:
        print(f"有效時間資料: {len(valid_data):,d} 筆")
        
        # 年份統計
        year_counts = valid_data['年份'].value_counts().sort_index()
        
        # 月份統計（確保是整數）
        month_counts = valid_data['月份'].value_counts().sort_index()
        month_counts.index = month_counts.index.astype(int)
        
        # 視覺化
        plt.figure(figsize=(15, 8))
        
        # 年度趨勢
        plt.subplot(2, 2, 1)
        plt.bar([str(int(y)) for y in year_counts.index], year_counts.values, alpha=0.7)
        plt.title('年度交易趨勢')
        plt.xlabel('民國年')
        plt.ylabel('交易筆數')
        plt.grid(True, alpha=0.3)
        
        # 月度分佈
        plt.subplot(2, 2, 2)
        
        # 建立完整月份資料
        month_data = {i: 0 for i in range(1, 13)}
        for month_idx, count in month_counts.items():
            if 1 <= month_idx <= 12:
                month_data[int(month_idx)] = count
        
        month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                     '7月', '8月', '9月', '10月', '11月', '12月']
        
        plt.bar(month_data.keys(), month_data.values(), alpha=0.7)
        plt.xticks(list(month_data.keys()), month_names, rotation=45)
        plt.title('月度交易分佈')
        plt.xlabel('月份')
        plt.ylabel('交易筆數')
        plt.grid(True, alpha=0.3)
        
        # 年度統計表
        plt.subplot(2, 2, 3)
        plt.axis('off')
        year_text = "年度交易統計:\n" + "\n".join(
            [f"民國{int(year)}年: {count:,d} 筆" 
             for year, count in year_counts.head(10).items()]
        )
        plt.text(0.1, 0.5, year_text, fontsize=10, verticalalignment='center')
        
        # 月度統計表
        plt.subplot(2, 2, 4)
        plt.axis('off')
        month_text = "月度交易統計:\n" + "\n".join(
            [f"{int(month):2d}月: {count:,d} 筆" 
             for month, count in month_counts.head(12).items()]
        )
        plt.text(0.1, 0.5, month_text, fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('時間序列分析.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 輸出統計
        print("\n年度交易統計:")
        for year, count in year_counts.items():
            print(f"民國{int(year)}年: {count:>6,d} 筆")
        
        print("\n月度交易統計:")
        for month, count in month_counts.sort_index().items():
            print(f"{int(month):2d}月: {count:>6,d} 筆")
    else:
        print("無有效的時間資料")

def main_fixed():
    """修正版的主程式"""
    print("=" * 70)
    print("內政部不動產實價登錄資料 - 視覺化分析工具（修正版）")
    print("=" * 70)
    
    # 載入資料
    df = load_and_prepare_data()
    if df is None:
        return
    
    # 建立輸出資料夾
    output_dir = Path(__file__).parent / "分析圖表"
    output_dir.mkdir(exist_ok=True)
    os.chdir(output_dir)
    
    print(f"\n圖表將儲存至: {output_dir}")
    
    try:
        # 執行各項分析（使用修正版函數）
        print("\n[1/6] 缺失值分析...")
        missing_stats = analyze_missing_values(df)
        
        print("\n[2/6] 數值特徵分析...")
        numeric_stats = analyze_numerical_features(df)
        
        print("\n[3/6] 類別特徵分析...")
        cat_stats = analyze_categorical_features(df)
        
        print("\n[4/6] 價格分佈分析...")
        price_data = analyze_price_distribution(df)
        
        print("\n[5/6] 時間序列分析...")
        analyze_time_series_fixed(df)  # 使用修正版
        
        print("\n[6/6] 生成分析報告...")
        # 簡單的報告
        report_path = Path(__file__).parent / "資料分析報告.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("資料分析報告\n")
            f.write(f"資料維度: {df.shape[0]} 行 × {df.shape[1]} 列\n")
            f.write(f"分析時間: {pd.Timestamp.now()}\n")
        
        print(f"\n{'='*70}")
        print("分析完成！")
        print(f"圖表已儲存至: {output_dir}")
        
    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
    
    input("\n按 Enter 鍵結束...")

if __name__ == "__main__":
    main_fixed()