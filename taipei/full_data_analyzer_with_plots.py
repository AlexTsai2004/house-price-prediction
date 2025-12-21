import pandas as pd
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def extract_year_month_from_roc(time_str):
    """
    從民國年格式時間字串中提取年份和月份
    支援格式：1120601 (民國112年6月1日)
    """
    if pd.isna(time_str) or str(time_str).strip() == '':
        return None, None
    
    time_str_val = str(time_str).strip()
    
    try:
        # 移除非數字字元
        time_str_val = ''.join(c for c in time_str_val if c.isdigit())
        
        if not time_str_val:
            return None, None
        
        # 根據長度判斷格式
        length = len(time_str_val)
        
        if length >= 7:  # 完整格式：YYYMMDD (如1120601)
            # 民國年：前3位
            roc_year_str = time_str_val[:3]
            month_str = time_str_val[3:5]
        elif length == 6:  # 可能格式：YYMMDD (如112601) 或 YYYMMD
            # 嘗試判斷
            if time_str_val[0] == '1':  # 民國1xx年
                roc_year_str = time_str_val[:3]
                month_str = time_str_val[3:5]
            else:  # 民國xx年
                roc_year_str = time_str_val[:2]
                month_str = time_str_val[2:4]
        elif length == 5:  # 格式：YYMMD 或 YMMDD
            if time_str_val[0] == '1':  # 民國1xx年
                roc_year_str = time_str_val[:3]
                month_str = time_str_val[3:5]
            else:
                return None, None
        else:
            return None, None
        
        # 轉換為整數
        roc_year = int(roc_year_str)
        month = int(month_str)
        
        # 驗證合理性
        if 100 <= roc_year <= 150 and 1 <= month <= 12:
            return roc_year, month
        elif 0 <= roc_year <= 99 and 1 <= month <= 12:
            return roc_year, month
        else:
            return None, None
            
    except:
        return None, None

def collect_residential_data(root_dir):
    """
    收集所有資料夾中住家用的資料
    處理內政部格式：第一列中文欄位名，第二列英文說明
    """
    all_residential_data = []
    file_count = 0
    
    root_path = Path(root_dir)
    
    for folder in root_path.iterdir():
        if folder.is_dir() and '_' in folder.name:
            print(f"處理資料夾: {folder.name}")
            
            # 處理 A、B 兩個檔案
            for file_name in ["A_lvr_land_A.csv", "A_lvr_land_B.csv"]:
                file_path = folder / file_name
                
                if file_path.exists():
                    try:
                        df = load_and_filter_internal_affairs_data(file_path)
                        if df is not None and not df.empty:
                            df['資料來源_資料夾'] = folder.name
                            df['資料來源_檔案'] = file_name
                            all_residential_data.append(df)
                            file_count += 1
                            print(f"  ✓ {file_name}: {len(df)} 筆住家用資料")
                        else:
                            print(f"  ✗ {file_name}: 無住家用資料或讀取失敗")
                    except Exception as e:
                        print(f"  ✗ {file_name}: 錯誤 - {str(e)}")
    
    # 合併所有資料
    if all_residential_data:
        combined_df = pd.concat(all_residential_data, ignore_index=True)
        
        # 儲存結果
        output_file = root_path / "所有住家用資料_合併.csv"
        combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*50}")
        print("處理完成！")
        print(f"{'='*50}")
        print(f"總處理檔案數: {file_count} 個")
        print(f"總住家用資料筆數: {len(combined_df)} 筆")
        print(f"輸出檔案: {output_file}")
        
        # 顯示詳細統計
        show_detailed_statistics(combined_df)
        
        # 創建詳細報告
        create_summary_report(combined_df, root_path)
        
        return combined_df
    else:
        print("沒有找到任何住家用資料")
        return None

def load_and_filter_internal_affairs_data(file_path):
    """
    專為內政部格式設計的資料載入函數
    第一列：中文欄位名
    第二列：英文說明（跳過）
    """
    try:
        # 方法1：先讀取前2行確認格式
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
        
        # 檢查是否是內政部格式（第二行可能是英文說明）
        is_internal_affairs_format = False
        if second_line and any(keyword in second_line.lower() for keyword in 
                              ['transaction', 'land', 'building', 'price', 'district']):
            is_internal_affairs_format = True
        
        # 根據格式決定如何讀取
        if is_internal_affairs_format:
            # 跳過第二行（英文說明）
            df = pd.read_csv(file_path, skiprows=[1], encoding='utf-8', low_memory=False)
            print(f"   偵測到內政部格式（跳過英文說明列）")
        else:
            # 正常讀取
            df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            print(f"   一般CSV格式")
        
        # 統一欄位名稱（移除多餘空格）
        df.columns = df.columns.str.strip()
        
        # 檢查並修正可能的編碼問題
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        # 尋找主要用途欄位
        main_use_column = None
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['主要用途', '用途', 'main use', 'use']):
                main_use_column = col
                break
        
        if main_use_column is None:
            print(f"    警告: 找不到'主要用途'相關欄位")
            return None
        
        # 篩選住家用資料
        print(f"    使用欄位 '{main_use_column}' 進行篩選")
        
        # 定義住家用關鍵字
        residential_keywords = [
            '住家用', '住家', '住宅', '住', 
            '住宅用', '居住', '住宅區',
            'residential', 'residence', 'house', 'housing', 'dwelling'
        ]
        
        # 建立篩選條件
        condition = pd.Series(False, index=df.index)
        
        if df[main_use_column].dtype == 'object':
            for keyword in residential_keywords:
                try:
                    condition = condition | df[main_use_column].str.contains(
                        keyword, na=False, case=False
                    )
                except:
                    continue
        
        filtered_df = df[condition].copy()
        
        # 如果資料量太少，嘗試更寬鬆的篩選
        if len(filtered_df) == 0 and df[main_use_column].dtype == 'object':
            print(f"    嚴格篩選無結果，嘗試寬鬆篩選...")
            # 檢查是否有任何包含"住"字的資料
            condition = df[main_use_column].str.contains('住', na=False)
            filtered_df = df[condition].copy()
        
        return filtered_df
        
    except UnicodeDecodeError:
        # 嘗試不同編碼
        try:
            df = pd.read_csv(file_path, encoding='big5', skiprows=[1], low_memory=False)
            df.columns = df.columns.str.strip()
            
            # 繼續篩選邏輯
            return filter_by_usage(df)
        except:
            print(f"    編碼讀取失敗")
            return None
    except Exception as e:
        print(f"    讀取失敗: {str(e)}")
        return None

def filter_by_usage(df):
    """輔助函數：篩選住家用資料"""
    main_use_column = None
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['主要用途', '用途', 'main use', 'use']):
            main_use_column = col
            break
    
    if main_use_column is None:
        return pd.DataFrame()
    
    residential_keywords = ['住家用', '住家', '住宅', '住', 'residential', 'residence']
    condition = pd.Series(False, index=df.index)
    
    if df[main_use_column].dtype == 'object':
        for keyword in residential_keywords:
            try:
                condition = condition | df[main_use_column].str.contains(
                    keyword, na=False, case=False
                )
            except:
                continue
    
    return df[condition].copy()

def show_detailed_statistics(df):
    """顯示詳細統計資訊 - 民國年專用版本"""
    print(f"\n{'='*50}")
    print("詳細統計分析")
    print(f"{'='*50}")
    
    # 1. 按資料來源統計
    if '資料來源_資料夾' in df.columns:
        print("\n1. 按年度期數統計:")
        print("-" * 30)
        source_stats = df['資料來源_資料夾'].value_counts().sort_index()
        total = 0
        for source, count in source_stats.items():
            print(f"  {source}: {count:6d} 筆")
            total += count
        print(f"  {'總計':<10} {total:6d} 筆")
    
    # 2. 按行政區統計
    district_columns = ['鄉鎮市區', '行政區', 'district', '區域']
    district_col = None
    for col in district_columns:
        if col in df.columns:
            district_col = col
            break
    
    if district_col:
        print(f"\n2. 按行政區統計:")
        print("-" * 30)
        district_stats = df[district_col].value_counts().head(20)
        for district, count in district_stats.items():
            print(f"  {str(district):<5}: {count:6d} 筆")
    
    # 3. 按建物型態統計
    building_type_cols = ['建物型態', 'building state', '建物類型']
    building_col = None
    for col in building_type_cols:
        if col in df.columns:
            building_col = col
            break
    
    if building_col:
        print(f"\n3. 按建物型態統計:")
        print("-" * 30)
        building_stats = df[building_col].value_counts().head(10)
        for btype, count in building_stats.items():
            print(f"  {str(btype):<20}: {count:6d} 筆")
    
    # 4. 價格分析
    price_cols = ['總價元', '單價元平方公尺', 'total price NTD', 'the unit price']
    for price_col in price_cols:
        if price_col in df.columns:
            try:
                prices = pd.to_numeric(df[price_col], errors='coerce')
                if prices.notna().sum() > 0:
                    print(f"\n4. 價格分析 ({price_col}):")
                    print("-" * 30)
                    print(f"  有效資料筆數: {prices.notna().sum():,d}")
                    print(f"  平均價格: {prices.mean():,.0f}")
                    print(f"  中位數: {prices.median():,.0f}")
                    print(f"  最高: {prices.max():,.0f}")
                    print(f"  最低: {prices.min():,.0f}")
                    print(f"  標準差: {prices.std():,.0f}")
                    break
            except:
                continue
    
    # 5. 時間分析（民國年格式）
    date_cols = ['交易年月日', 'transaction year month and day', '交易日期']
    for date_col in date_cols:
        if date_col in df.columns:
            try:
                print(f"\n5. 按交易時間統計（民國年）:")
                print("-" * 30)
                
                # 使用 extract_year_month_from_roc 函數提取資料
                roc_data = []
                valid_count = 0
                
                for time_val in df[date_col]:
                    roc_year, month = extract_year_month_from_roc(time_val)
                    if roc_year is not None and month is not None:
                        roc_data.append({
                            'roc_year': roc_year,
                            'month': month
                        })
                        valid_count += 1
                    else:
                        roc_data.append({
                            'roc_year': None,
                            'month': None
                        })
                
                # 創建DataFrame
                roc_df = pd.DataFrame(roc_data)
                roc_valid = roc_df.dropna()
                
                print(f"  有效時間資料: {valid_count:6d} / {len(df):6d} 筆")
                print(f"  資料完整率: {(valid_count/len(df)*100):.1f}%")
                
                if valid_count > 0:
                    # 按民國年份統計
                    print("\n  a. 按民國年份統計:")
                    print("  " + "-" * 28)
                    
                    year_stats = roc_valid['roc_year'].value_counts().sort_index()
                    
                    for roc_year, count in year_stats.items():
                        print(f"    民國{int(roc_year):3d}年: {count:6d} 筆")
                    
                    # 計算年度趨勢
                    if len(year_stats) > 1:
                        print(f"\n  b. 年度交易趨勢:")
                        print("  " + "-" * 28)
                        
                        years_sorted = sorted(year_stats.index)
                        prev_count = None
                        
                        for roc_year in years_sorted:
                            count = year_stats[roc_year]
                            
                            if prev_count is not None and prev_count > 0:
                                change_pct = ((count - prev_count) / prev_count) * 100
                                change_symbol = "↑" if change_pct > 0 else "↓"
                                print(f"    民國{int(roc_year):3d}年: {count:6d} 筆 "
                                      f"({change_symbol}{abs(change_pct):.1f}%)")
                            else:
                                print(f"    民國{int(roc_year):3d}年: {count:6d} 筆")
                            
                            prev_count = count
                    
                    # 按月份統計
                    print(f"\n  c. 按月份統計:")
                    print("  " + "-" * 28)
                    
                    month_stats = roc_valid['month'].value_counts().sort_index()
                    month_names = ['一月', '二月', '三月', '四月', '五月', '六月',
                                 '七月', '八月', '九月', '十月', '十一月', '十二月']
                    
                    for month, count in month_stats.items():
                        month_int = int(month)
                        if 1 <= month_int <= 12:
                            month_name = month_names[month_int - 1]
                            print(f"    {month_name:>4}（{month_int:02d}月）: {count:6d} 筆")
                    
                    # 熱門交易年月組合
                    print(f"\n  d. 熱門交易年月（前10名）:")
                    print("  " + "-" * 28)
                    
                    ym_stats = roc_valid.groupby(['roc_year', 'month']).size()
                    ym_stats_sorted = ym_stats.sort_values(ascending=False).head(10)
                    
                    for (roc_year, month), count in ym_stats_sorted.items():
                        print(f"    民國{int(roc_year):3d}年{int(month):02d}月: {count:6d} 筆")
                    
                    # 季節性分析
                    print(f"\n  e. 季節性分析:")
                    print("  " + "-" * 28)
                    
                    # 定義季節
                    seasons = {
                        '春季(3-5月)': [3, 4, 5],
                        '夏季(6-8月)': [6, 7, 8],
                        '秋季(9-11月)': [9, 10, 11],
                        '冬季(12-2月)': [12, 1, 2]
                    }
                    
                    season_counts = {}
                    for season_name, months in seasons.items():
                        season_data = roc_valid[roc_valid['month'].isin(months)]
                        season_counts[season_name] = len(season_data)
                    
                    for season_name, count in season_counts.items():
                        percentage = (count / len(roc_valid)) * 100
                        print(f"    {season_name:<12}: {count:6d} 筆 ({percentage:.1f}%)")
                    
                    # 最近一年的詳細分析
                    if not year_stats.empty:
                        latest_year = max(year_stats.index)
                        latest_year_data = roc_valid[roc_valid['roc_year'] == latest_year]
                        
                        if not latest_year_data.empty:
                            print(f"\n  f. 民國{int(latest_year)}年詳細分析:")
                            print("  " + "-" * 28)
                            
                            # 每月交易量
                            monthly_counts = latest_year_data['month'].value_counts().sort_index()
                            
                            for month in range(1, 13):
                                count = monthly_counts.get(month, 0)
                                month_name = month_names[month-1]
                                print(f"    {month_name:>4}: {count:6d} 筆")
                
                else:
                    print("  警告：無有效的時間資料可分析")
                    # 顯示原始資料範例
                    print("\n  原始資料範例（前10筆）:")
                    sample_data = df[date_col].head(10)
                    for i, val in enumerate(sample_data, 1):
                        print(f"    {i:2d}. {val}")
                
                break
                
            except Exception as e:
                print(f"  時間分析錯誤: {str(e)}")
                continue
    
    print(f"\n{'='*50}")

def create_summary_report(df, output_dir):
    """創建詳細的摘要報告"""
    report_path = Path(output_dir) / "住家用資料摘要報告.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("        內政部不動產實價登錄 - 住家用資料摘要報告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"資料期間: 112年第1期 ~ 114年第4期\n")
        f.write(f"總資料筆數: {len(df):,d} 筆\n")
        f.write(f"資料來源檔案數: {df['資料來源_資料夾'].nunique()} 個資料夾\n\n")
        
        # 資料結構
        f.write("【資料結構】\n")
        f.write("-" * 40 + "\n")
        f.write(f"總欄位數: {len(df.columns)}\n")
        f.write("主要欄位:\n")
        for i, col in enumerate(df.columns[:15], 1):
            f.write(f"  {i:2d}. {col}\n")
        if len(df.columns) > 15:
            f.write(f"  ... 還有 {len(df.columns)-15} 個欄位\n")
        
        # 資料來源分佈
        f.write("\n【資料來源分佈】\n")
        f.write("-" * 40 + "\n")
        for folder in sorted(df['資料來源_資料夾'].unique()):
            count = len(df[df['資料來源_資料夾'] == folder])
            f.write(f"{folder}: {count:6,d} 筆\n")
    
    print(f"摘要報告已儲存至: {report_path}")

def save_data_by_district(df, output_dir):
    """按行政區分開儲存檔案"""
    district_columns = ['鄉鎮市區', '行政區', 'district']
    district_col = None
    
    for col in district_columns:
        if col in df.columns:
            district_col = col
            break
    
    if district_col:
        output_path = Path(output_dir) / "各行政區住家用資料"
        output_path.mkdir(exist_ok=True)
        
        districts = df[district_col].unique()
        print(f"\n按行政區分開儲存 ({len(districts)} 個行政區):")
        
        for district in districts:
            if pd.isna(district):
                continue
                
            district_df = df[df[district_col] == district]
            district_name = str(district).replace('/', '_').replace('\\', '_')
            file_name = f"{district_name}_住家用資料.csv"
            file_path = output_path / file_name
            
            district_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ {district_name}: {len(district_df)} 筆")
        
        print(f"\n所有行政區資料已儲存至: {output_path}")

def main():
    """主程式"""
    # 設定根目錄
    current_dir = Path(__file__).parent
    
    print("=" * 60)
    print("內政部不動產實價登錄資料 - 住家用資料收集工具")
    print("=" * 60)
    print(f"工作目錄: {current_dir}")
    print(f"開始收集 112_1 到 114_4 的住家用資料...\n")
    
    # 執行收集
    result_df = collect_residential_data(current_dir)
    
    if result_df is not None:
        # 額外：儲存每個行政區的獨立檔案
        save_by_district = input("\n是否要按行政區分開儲存檔案？ (y/n): ").lower()
        if save_by_district == 'y':
            save_data_by_district(result_df, current_dir)
    
    input("\n按 Enter 鍵結束...")

if __name__ == "__main__":
    main()