import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class TimeDataProcessor:
    """時間資料處理器 - 處理交易年月日和建築完成年月"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = self.df.shape
        print(f"原始資料維度: {self.original_shape[0]:,d} 行 × {self.original_shape[1]} 列")
    
    def remove_missing_construction_date(self):
        """
        移除建築完成年月為空值或空字串的資料
        """
        print("\n" + "="*60)
        print("移除建築完成年月為空的資料")
        print("="*60)
        
        if '建築完成年月' not in self.df.columns:
            print("錯誤：找不到'建築完成年月'欄位")
            print("資料欄位列表:", list(self.df.columns)[:20], "...")
            return self.df
        
        # 記錄移除前的狀態
        before_count = len(self.df)
        
        # 定義空的條件
        def is_empty_or_null(value):
            if pd.isna(value):
                return True
            if isinstance(value, str) and value.strip() == '':
                return True
            if isinstance(value, (int, float)) and pd.isna(value):
                return True
            return False
        
        # 標記需要移除的資料
        self.df['_要移除標記'] = self.df['建築完成年月'].apply(is_empty_or_null)
        to_remove_count = self.df['_要移除標記'].sum()
        
        print(f"檢查建築完成年月欄位...")
        print(f"  原始資料筆數: {before_count:,d}")
        print(f"  發現空值或空字串: {to_remove_count:,d} 筆")
        
        if to_remove_count > 0:
            # 顯示要移除的資料樣本
            print(f"\n要移除的資料樣本（前5筆）:")
            samples = self.df[self.df['_要移除標記'] == 1].head()
            for idx, row in samples.iterrows():
                const_date = row['建築完成年月']
                trans_date = row.get('交易年月日', 'N/A')
                print(f"  索引 {idx}: 建築完成年月='{const_date}', 交易年月日='{trans_date}'")
            
            # 移除資料
            df_removed = self.df[self.df['_要移除標記'] == 1].copy()
            self.df = self.df[self.df['_要移除標記'] == 0].copy()
            
            after_count = len(self.df)
            removed_pct = (to_remove_count / before_count * 100)
            
            print(f"\n移除完成:")
            print(f"  移除筆數: {to_remove_count:,d} 筆 ({removed_pct:.1f}%)")
            print(f"  剩餘筆數: {after_count:,d} 筆")
            print(f"  移除比例: {removed_pct:.1f}%")
            
            # 儲存移除的資料供參考
            removed_file = Path(__file__).parent / "已移除_建築完成年月為空.csv"
            df_removed.drop('_要移除標記', axis=1, inplace=True)
            df_removed.to_csv(removed_file, index=False, encoding='utf-8-sig')
            print(f"  已移除資料儲存至: {removed_file}")
        else:
            print("  無需移除任何資料")
        
        # 移除標記欄位
        if '_要移除標記' in self.df.columns:
            self.df.drop('_要移除標記', axis=1, inplace=True)
        
        return self.df
    
    def impute_transaction_date(self):
        """
        填補交易年月日的缺失值
        使用資料來源推斷預設時間
        """
        print("\n" + "="*60)
        print("填補交易年月日缺失值")
        print("="*60)
        
        if '交易年月日' not in self.df.columns:
            print("錯誤：找不到'交易年月日'欄位")
            return self.df
        
        # 檢查原始狀態
        missing_before = self.df['交易年月日'].isnull().sum()
        empty_before = (self.df['交易年月日'].astype(str).str.strip() == '').sum()
        total_missing_before = missing_before + empty_before
        
        print(f"交易年月日原始狀態:")
        print(f"  總資料筆數: {len(self.df):,d}")
        print(f"  空值數量: {missing_before:,d}")
        print(f"  空字串數量: {empty_before:,d}")
        print(f"  總缺失數量: {total_missing_before:,d}")
        
        if total_missing_before == 0:
            print("  無缺失值，跳過填補")
            # 還是創建填補標記欄位（全部為0）
            self.df['交易年月日_填補標記'] = 0
            return self.df
        
        # 準備預設時間
        print(f"\n準備預設時間...")
        
        # 尋找資料來源欄位
        source_col = None
        source_candidates = ['資料來源_資料夾', '資料來源_檔案', '檔案名稱', '資料來源']
        
        for col in source_candidates:
            if col in self.df.columns:
                source_col = col
                print(f"  使用來源欄位: {col}")
                break
        
        if source_col is None:
            print("  警告：找不到資料來源欄位，使用預設 1140101")
            default_time = '1140101'
            default_series = pd.Series(default_time, index=self.df.index)
        else:
            # 從資料來源提取預設時間
            def get_default_time(source):
                if pd.isna(source):
                    return '1140101'
                
                source_str = str(source)
                
                # 格式：114_1, 113_2, 112_3, 112_4
                if '_' in source_str:
                    try:
                        parts = source_str.split('_')
                        if len(parts) >= 2:
                            year_part = parts[0].strip()
                            period_part = parts[1].strip()
                            
                            if year_part.isdigit() and period_part.isdigit():
                                roc_year = int(year_part)
                                period = int(period_part)
                                
                                # 期數轉月份
                                month_map = {1: '01', 2: '04', 3: '07', 4: '10'}
                                month = month_map.get(period, '01')
                                
                                return f"{roc_year:03d}{month}01"
                    except:
                        pass
                
                return '1140101'
            
            default_series = self.df[source_col].apply(get_default_time)
            
            # 顯示預設時間分布
            print(f"  預設時間分布:")
            time_counts = default_series.value_counts().head(5)
            for time_val, count in time_counts.items():
                pct = (count / len(self.df) * 100)
                print(f"    {time_val}: {count:>6,d} 筆 ({pct:.1f}%)")
        
        # 填補缺失值
        print(f"\n開始填補...")
        fill_count = 0
        
        # 創建填補標記欄位
        self.df['交易年月日_填補標記'] = 0
        
        for idx in self.df.index:
            trans_date = self.df.at[idx, '交易年月日']
            
            # 檢查是否為空
            is_empty = False
            if pd.isna(trans_date):
                is_empty = True
            elif isinstance(trans_date, str) and trans_date.strip() == '':
                is_empty = True
            
            if is_empty:
                self.df.at[idx, '交易年月日'] = default_series[idx]
                self.df.at[idx, '交易年月日_填補標記'] = 1
                fill_count += 1
        
        # 檢查結果
        missing_after = self.df['交易年月日'].isnull().sum()
        empty_after = (self.df['交易年月日'].astype(str).str.strip() == '').sum()
        total_missing_after = missing_after + empty_after
        
        print(f"\n填補結果:")
        print(f"  填補筆數: {fill_count:,d}")
        print(f"  填補後缺失: {total_missing_after:,d}")
        
        # 檢查填補標記
        marked_count = self.df['交易年月日_填補標記'].sum()
        print(f"  標記為填補: {marked_count:,d} 筆")
        
        return self.df
    
    def create_derived_features(self):
        """
        從時間欄位創建衍生特徵
        """
        print("\n" + "="*60)
        print("創建衍生時間特徵")
        print("="*60)
        
        # 1. 從交易年月日提取特徵
        if '交易年月日' in self.df.columns:
            print(f"從'交易年月日'創建衍生特徵:")
            
            # 提取民國年
            def extract_trans_year(date_str):
                if pd.isna(date_str):
                    return None
                
                date_str = str(date_str).strip()
                digits = ''.join(c for c in date_str if c.isdigit())
                
                if len(digits) >= 3:
                    try:
                        if digits[0] == '1':
                            return int(digits[:3])
                        else:
                            return int(digits[:2])
                    except:
                        pass
                return None
            
            # 提取月份
            def extract_trans_month(date_str):
                if pd.isna(date_str):
                    return None
                
                date_str = str(date_str).strip()
                digits = ''.join(c for c in date_str if c.isdigit())
                
                if len(digits) >= 5:
                    try:
                        if digits[0] == '1':
                            month_str = digits[3:5]
                        else:
                            month_str = digits[2:4]
                        
                        month = int(month_str)
                        if 1 <= month <= 12:
                            return month
                    except:
                        pass
                return None
            
            self.df['交易年月日_民國年'] = self.df['交易年月日'].apply(extract_trans_year)
            self.df['交易年月日_月份'] = self.df['交易年月日'].apply(extract_trans_month)
            
            # 提取季別
            def extract_quarter(month):
                if pd.isna(month):
                    return None
                if isinstance(month, (int, float)):
                    month_int = int(month)
                else:
                    try:
                        month_int = int(month)
                    except:
                        return None
                
                if 1 <= month_int <= 3:
                    return 1
                elif 4 <= month_int <= 6:
                    return 2
                elif 7 <= month_int <= 9:
                    return 3
                elif 10 <= month_int <= 12:
                    return 4
                return None
            
            self.df['交易年月日_季別'] = self.df['交易年月日_月份'].apply(extract_quarter)
            
            # 統計
            valid_year = self.df['交易年月日_民國年'].notna().sum()
            valid_month = self.df['交易年月日_月份'].notna().sum()
            valid_quarter = self.df['交易年月日_季別'].notna().sum()
            
            print(f"  交易民國年: {valid_year:,d} 筆有效")
            print(f"  交易月份: {valid_month:,d} 筆有效")
            print(f"  交易季別: {valid_quarter:,d} 筆有效")
        
        # 2. 從建築完成年月提取特徵
        if '建築完成年月' in self.df.columns:
            print(f"\n從'建築完成年月'創建衍生特徵:")
            
            # 提取民國年
            def extract_const_year(date_str):
                if pd.isna(date_str):
                    return None
                
                date_str = str(date_str).strip()
                digits = ''.join(c for c in date_str if c.isdigit())
                
                if len(digits) >= 3:
                    try:
                        if digits[0] == '1':
                            return int(digits[:3])
                        else:
                            return int(digits[:2])
                    except:
                        pass
                return None
            
            # 提取月份
            def extract_const_month(date_str):
                if pd.isna(date_str):
                    return None
                
                date_str = str(date_str).strip()
                digits = ''.join(c for c in date_str if c.isdigit())
                
                if len(digits) >= 5:
                    try:
                        if digits[0] == '1':
                            month_str = digits[3:5]
                        else:
                            month_str = digits[2:4]
                        
                        month = int(month_str)
                        if 1 <= month <= 12:
                            return month
                    except:
                        pass
                return None
            
            self.df['建築完成年月_民國年'] = self.df['建築完成年月'].apply(extract_const_year)
            self.df['建築完成年月_月份'] = self.df['建築完成年月'].apply(extract_const_month)
            self.df['建築完成年月_季別'] = self.df['建築完成年月_月份'].apply(extract_quarter)
            
            # 統計
            valid_year = self.df['建築完成年月_民國年'].notna().sum()
            valid_month = self.df['建築完成年月_月份'].notna().sum()
            valid_quarter = self.df['建築完成年月_季別'].notna().sum()
            
            print(f"  建築民國年: {valid_year:,d} 筆有效")
            print(f"  建築月份: {valid_month:,d} 筆有效")
            print(f"  建築季別: {valid_quarter:,d} 筆有效")
        
        # 3. 計算屋齡
        if '交易年月日_民國年' in self.df.columns and '建築完成年月_民國年' in self.df.columns:
            print(f"\n計算屋齡:")
            
            self.df['屋齡_年'] = self.df['交易年月日_民國年'] - self.df['建築完成年月_民國年']
            
            # 過濾不合理值
            def clean_age(age):
                if pd.isna(age):
                    return None
                try:
                    age_float = float(age)
                    if 0 <= age_float <= 100:
                        return age_float
                except:
                    pass
                return None
            
            self.df['屋齡_年'] = self.df['屋齡_年'].apply(clean_age)
            
            valid_age = self.df['屋齡_年'].notna().sum()
            
            if valid_age > 0:
                avg_age = self.df['屋齡_年'].mean()
                min_age = self.df['屋齡_年'].min()
                max_age = self.df['屋齡_年'].max()
                
                print(f"  有效屋齡: {valid_age:,d} 筆")
                print(f"  平均屋齡: {avg_age:.1f} 年")
                print(f"  最小屋齡: {min_age:.1f} 年")
                print(f"  最大屋齡: {max_age:.1f} 年")
            else:
                print(f"  無有效屋齡資料")
        
        return self.df
    
    def generate_summary_report(self):
        """生成處理摘要報告"""
        report = {
            '處理階段': [],
            '最終統計': {
                '總資料筆數': len(self.df),
                '總欄位數': len(self.df.columns),
                '建築完成年月為空已移除': self.original_shape[0] - len(self.df)
            }
        }
        
        # 時間欄位統計
        time_stats = {}
        time_columns = ['交易年月日', '建築完成年月']
        
        for col in time_columns:
            if col in self.df.columns:
                # 檢查缺失
                missing = self.df[col].isnull().sum()
                empty_str = (self.df[col].astype(str).str.strip() == '').sum()
                
                time_stats[col] = {
                    '總筆數': len(self.df),
                    '空值數': missing,
                    '空字串數': empty_str,
                    '總缺失數': missing + empty_str,
                    '缺失率%': ((missing + empty_str) / len(self.df) * 100) if len(self.df) > 0 else 0
                }
        
        report['時間欄位統計'] = time_stats
        
        # 填補標記統計
        fill_stats = {}
        if '交易年月日_填補標記' in self.df.columns:
            filled = self.df['交易年月日_填補標記'].sum()
            fill_stats['交易年月日_填補標記'] = {
                '填補筆數': int(filled),
                '填補率%': (filled / len(self.df) * 100) if len(self.df) > 0 else 0
            }
        
        report['填補統計'] = fill_stats
        
        # 衍生特徵統計
        derived_stats = {}
        derived_patterns = ['_民國年', '_月份', '_季別', '屋齡_年']
        
        for pattern in derived_patterns:
            matching_cols = [col for col in self.df.columns if pattern in col]
            for col in matching_cols:
                valid = self.df[col].notna().sum()
                derived_stats[col] = {
                    '有效筆數': int(valid),
                    '有效率%': (valid / len(self.df) * 100) if len(self.df) > 0 else 0
                }
        
        report['衍生特徵統計'] = derived_stats
        
        return report

def main():
    """主程式"""
    print("=" * 70)
    print("時間資料處理工具")
    print("處理策略:")
    print("  1. 移除建築完成年月為空值或空字串的資料")
    print("  2. 填補交易年月日的缺失值")
    print("  3. 創建衍生時間特徵")
    print("=" * 70)
    
    # 設定檔案路徑
    current_dir = Path(__file__).parent
    input_file = current_dir / "所有住家用資料_合併.csv"
    output_file = current_dir / "所有住家用資料_時間處理後.csv"
    report_file = current_dir / "時間處理報告.txt"
    
    if not input_file.exists():
        print(f"錯誤：找不到輸入檔案 {input_file}")
        print("請先執行 collect_residential_data.py 生成合併資料")
        return
    
    # 載入資料
    print(f"載入資料檔案: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"原始資料維度: {df.shape[0]:,d} 行 × {df.shape[1]} 列")
    
    # 顯示原始時間欄位
    print("\n原始時間欄位檢查:")
    for col in ['交易年月日', '建築完成年月']:
        if col in df.columns:
            missing = df[col].isnull().sum()
            empty = (df[col].astype(str).str.strip() == '').sum()
            total = missing + empty
            pct = (total / len(df) * 100) if len(df) > 0 else 0
            print(f"  {col}: {total:>8,d} 筆空值 ({pct:6.1f}%)")
        else:
            print(f"  {col}: 欄位不存在")
    
    # 初始化處理器
    print("\n初始化時間資料處理器...")
    processor = TimeDataProcessor(df)
    
    try:
        # 第1步：移除建築完成年月為空的資料
        print("\n執行第1步：移除建築完成年月為空的資料")
        df_processed = processor.remove_missing_construction_date()
        
        # 第2步：填補交易年月日
        print("\n執行第2步：填補交易年月日缺失值")
        df_processed = processor.impute_transaction_date()
        
        # 第3步：創建衍生特徵
        print("\n執行第3步：創建衍生時間特徵")
        df_processed = processor.create_derived_features()
        
        # 生成報告
        print("\n生成處理報告...")
        report = processor.generate_summary_report()
        
    except Exception as e:
        print(f"\n處理過程中發生錯誤: {str(e)}")
        print("嘗試保存已處理的資料...")
        
        # 嘗試保存當前狀態
        try:
            df_processed = processor.df
            df_processed.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"已保存部分處理的資料至: {output_file}")
        except:
            print("無法保存資料")
        
        input("\n按 Enter 鍵結束...")
        return
    
    # 儲存結果
    print(f"\n儲存處理後的資料...")
    df_processed.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已儲存至: {output_file}")
    
    # 儲存報告
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("                 時間資料處理報告\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("【處理策略】\n")
            f.write("-" * 80 + "\n")
            f.write("1. 移除建築完成年月為空值或空字串的資料\n")
            f.write("2. 填補交易年月日的缺失值（使用資料來源推斷預設時間）\n")
            f.write("3. 創建衍生時間特徵\n\n")
            
            # 最終統計
            final_stats = report['最終統計']
            f.write("【最終統計】\n")
            f.write("-" * 80 + "\n")
            f.write(f"原始資料筆數: {processor.original_shape[0]:,d}\n")
            f.write(f"處理後資料筆數: {final_stats['總資料筆數']:,d}\n")
            f.write(f"移除筆數: {final_stats['建築完成年月為空已移除']:,d}\n")
            f.write(f"移除比例: {(final_stats['建築完成年月為空已移除'] / processor.original_shape[0] * 100):.1f}%\n")
            f.write(f"總欄位數: {final_stats['總欄位數']}\n\n")
            
            # 時間欄位統計
            if '時間欄位統計' in report:
                f.write("【時間欄位統計】\n")
                f.write("-" * 80 + "\n")
                
                for col, stats in report['時間欄位統計'].items():
                    f.write(f"\n{col}:\n")
                    f.write(f"  總筆數: {stats['總筆數']:,d}\n")
                    f.write(f"  空值數: {stats['空值數']:,d}\n")
                    f.write(f"  空字串數: {stats['空字串數']:,d}\n")
                    f.write(f"  總缺失數: {stats['總缺失數']:,d}\n")
                    f.write(f"  缺失率: {stats['缺失率%']:.1f}%\n")
            
            # 填補統計
            if '填補統計' in report and report['填補統計']:
                f.write("\n【填補統計】\n")
                f.write("-" * 80 + "\n")
                
                for col, stats in report['填補統計'].items():
                    f.write(f"\n{col}:\n")
                    f.write(f"  填補筆數: {stats['填補筆數']:,d}\n")
                    f.write(f"  填補率: {stats['填補率%']:.1f}%\n")
            
            # 衍生特徵統計
            if '衍生特徵統計' in report and report['衍生特徵統計']:
                f.write("\n【衍生特徵統計】\n")
                f.write("-" * 80 + "\n")
                
                for col, stats in report['衍生特徵統計'].items():
                    f.write(f"\n{col}:\n")
                    f.write(f"  有效筆數: {stats['有效筆數']:,d}\n")
                    f.write(f"  有效率: {stats['有效率%']:.1f}%\n")
        
        print(f"處理報告已儲存至: {report_file}")
        
    except Exception as e:
        print(f"儲存報告時發生錯誤: {str(e)}")
    
    # 顯示最終摘要
    print("\n" + "=" * 70)
    print("處理完成摘要")
    print("=" * 70)
    
    final_stats = report['最終統計']
    print(f"\n原始資料筆數: {processor.original_shape[0]:,d}")
    print(f"最終資料筆數: {final_stats['總資料筆數']:,d}")
    print(f"移除筆數: {final_stats['建築完成年月為空已移除']:,d}")
    print(f"移除比例: {(final_stats['建築完成年月為空已移除'] / processor.original_shape[0] * 100):.1f}%")
    
    # 顯示時間欄位狀態
    print(f"\n時間欄位狀態:")
    for col in ['交易年月日', '建築完成年月']:
        if col in df_processed.columns:
            missing = df_processed[col].isnull().sum()
            empty = (df_processed[col].astype(str).str.strip() == '').sum()
            total = missing + empty
            
            if total == 0:
                print(f"  {col}: ✓ 完整無缺失")
            else:
                pct = (total / len(df_processed) * 100)
                print(f"  {col}: ⚠ 仍有 {total:,d} 筆缺失 ({pct:.1f}%)")
    
    # 檢查填補標記是否存在
    fill_mark_exists = '交易年月日_填補標記' in df_processed.columns
    
    # 顯示衍生特徵
    derived_cols = [col for col in df_processed.columns if any(
        x in col for x in ['_民國年', '_月份', '_季別', '屋齡_年', '填補標記']
    )]
    
    print(f"\n新增衍生特徵: {len(derived_cols)} 個")
    for col in derived_cols[:8]:  # 顯示前8個
        if col in df_processed.columns:
            if col.endswith('_填補標記'):
                value = df_processed[col].sum()
                pct = (value / len(df_processed) * 100)
                print(f"  {col}: {int(value):,d} 筆為1 ({pct:.1f}%)")
            else:
                valid = df_processed[col].notna().sum()
                pct = (valid / len(df_processed) * 100)
                print(f"  {col}: {valid:,d} 筆有效 ({pct:.1f}%)")
    
    if len(derived_cols) > 8:
        print(f"  ... 還有 {len(derived_cols)-8} 個衍生特徵")
    
    # 顯示資料樣本 - 檢查欄位是否存在
    print(f"\n前3筆資料樣本:")
    
    # 動態確定要顯示的欄位
    sample_cols = []
    possible_cols = ['交易年月日', '交易年月日_填補標記', '建築完成年月', '屋齡_年',
                    '交易年月日_民國年', '建築完成年月_民國年']
    
    for col in possible_cols:
        if col in df_processed.columns:
            sample_cols.append(col)
    
    if sample_cols:
        sample_data = df_processed[sample_cols].head(3)
        
        for idx, row in sample_data.iterrows():
            print(f"\n第 {idx+1} 筆:")
            for col in sample_cols:
                if col in row:
                    value = row[col]
                    if pd.isna(value):
                        display = "NaN"
                    else:
                        display = str(value)
                    print(f"  {col}: {display}")
    else:
        print("  找不到要顯示的欄位")
    
    print(f"\n處理完成！所有資料已儲存。")
    input("\n按 Enter 鍵結束...")

if __name__ == "__main__":
    main()