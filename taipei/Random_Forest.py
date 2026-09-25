"""
台北住宅房價 Random Forest（修正版）

與課程繳交版本（`original` 分支）的差異：
1. 行政區 Target Encoding 改在切分之後、只用訓練集計算（修正資料洩漏）。
2. 類別特徵以 OrdinalEncoder 在訓練集上 fit，測試集未見類別編為 -1；
   判斷類別欄位不再依賴 dtype == 'object'（新版 pandas 會漏掉類別特徵）。
3. 誤差指標修正：原版的「平均誤差 %」是有正負號的平均，高估與低估互相抵銷。
   修正版回報 MAPE（平均絕對百分比誤差）與中位數絕對百分比誤差，
   有正負號的平均另外標示為「偏差」。
4. 除原本的隨機切分外，增加時間切分驗證：以 113 年以前的交易訓練、
   114 年的交易測試，較接近「預測未來成交價」的實際情境。
5. 圖表改存成檔案；模型超參數與原版相同。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

HERE = Path(__file__).parent
DATA = HERE / '所有住家用資料_特徵處理後.csv'
RESULTS = HERE / 'results'
RESULTS.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

NUMERIC = [
    '主建物面積', '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛',
    '車位移轉總面積平方公尺', '附屬建物面積', '陽台面積', '屋齡_年',
]
BOOLEAN = ['有無管理組織', '建物現況格局-隔間', '電梯']
KEYWORDS = [
    '親友', '員工', '共有人', '特殊關係', '陽台外推', '夾層', '頂樓加蓋', '其他增建',
    '車位交易', '僅車位', '僅建物', '部分移轉', '租約', '含租約', '裝潢費', '傢俱設備費',
    '急買急賣', '債務', '債權債務', '畸零地', '二親等', '叔侄', '直系',
]
REMARK = [f'備註_{k}' for k in KEYWORDS]
CATEGORICAL = ['交易標的', '都市土地使用分區', '主要建材', '建物型態', '車位類別']
TARGET = '總價元_log'
DISTRICT_ENC = '鄉鎮市區_target_encoded'

RF_PARAMS = dict(n_estimators=100, max_depth=10, min_samples_split=5,
                 min_samples_leaf=2, random_state=42, n_jobs=-1)


def build_features(df, train_idx, test_idx):
    """所有需要學習的轉換都只在訓練集上 fit。"""
    X = pd.DataFrame(index=df.index)
    for c in NUMERIC + REMARK:
        X[c] = pd.to_numeric(df[c], errors='coerce')
    for c in BOOLEAN:
        X[c] = df[c].map({True: 1, False: 0, 'True': 1, 'False': 0})

    # 行政區 Target Encoding：只用訓練集的平均單價
    train = df.loc[train_idx]
    region_mean = train.groupby('鄉鎮市區')['單價元平方公尺'].mean()
    overall = train['單價元平方公尺'].mean()
    X[DISTRICT_ENC] = df['鄉鎮市區'].map(region_mean).fillna(overall)

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    enc.fit(df.loc[train_idx, CATEGORICAL].fillna('一般').astype(str))
    X[CATEGORICAL] = enc.transform(df[CATEGORICAL].fillna('一般').astype(str))
    return X.loc[train_idx], X.loc[test_idx]


def metrics(y_log, p_log):
    y, p = np.expm1(y_log), np.expm1(p_log)
    ape = np.abs(p - y) / y * 100
    return {
        'R2_log': round(float(r2_score(y_log, p_log)), 4),
        'R2_price': round(float(r2_score(y, p)), 4),
        'MAE_元': int(round(mean_absolute_error(y, p))),
        'RMSE_元': int(round(np.sqrt(mean_squared_error(y, p)))),
        'MAPE_%': round(float(ape.mean()), 1),
        'MedianAPE_%': round(float(np.median(ape)), 1),
        '偏差_有正負號平均_%': round(float(((p - y) / y * 100).mean()), 1),
        '誤差10%以內比例_%': round(float((ape <= 10).mean() * 100), 1),
    }


def run(df, train_idx, test_idx, tag):
    Xtr, Xte = build_features(df, train_idx, test_idx)
    ytr, yte = df.loc[train_idx, TARGET], df.loc[test_idx, TARGET]
    model = RandomForestRegressor(**RF_PARAMS).fit(Xtr, ytr)
    res = {
        'split': tag,
        'n_train': len(Xtr), 'n_test': len(Xte), 'n_features': Xtr.shape[1],
        'train': metrics(ytr, model.predict(Xtr)),
        'test': metrics(yte, model.predict(Xte)),
    }
    imp = pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    res['top10_importance'] = {k: round(float(v), 3) for k, v in imp.head(10).items()}
    return res, model, Xte, yte


def plot(model, Xte, yte, tag, fname):
    p = model.predict(Xte)
    y_o, p_o = np.expm1(yte), np.expm1(p)
    ape = np.abs(p_o - y_o) / y_o * 100
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].scatter(yte, p, s=4, alpha=0.4)
    lo, hi = yte.min(), yte.max()
    ax[0].plot([lo, hi], [lo, hi], 'r--')
    ax[0].set(xlabel='實際 log(總價)', ylabel='預測 log(總價)', title=f'{tag}：預測 vs 實際')
    ax[1].scatter(p, yte - p, s=4, alpha=0.4)
    ax[1].axhline(0, color='r', ls='--')
    ax[1].set(xlabel='預測 log(總價)', ylabel='殘差', title='殘差圖')
    ax[2].hist(np.clip(ape, 0, 100), bins=50, edgecolor='black', alpha=0.7)
    ax[2].axvline(np.median(ape), color='r', ls='--', label=f'中位數 {np.median(ape):.1f}%')
    ax[2].set(xlabel='絕對百分比誤差 %（>100% 併入 100）', ylabel='筆數', title='絕對百分比誤差分布')
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(RESULTS / fname, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA, low_memory=False)
    print(f'資料形狀: {df.shape}')

    # A. 隨機切分（與原版相同：80/20、random_state=42）
    tr, te = train_test_split(df.index, test_size=0.2, random_state=42)
    res_random, m, Xte, yte = run(df, tr, te, '隨機切分 80/20')
    plot(m, Xte, yte, '隨機切分', '隨機切分_預測結果.png')

    # B. 時間切分：113 年以前訓練、114 年測試
    year = pd.to_numeric(df['交易年月日_民國年'], errors='coerce')
    tr_t, te_t = df.index[year <= 113], df.index[year == 114]
    res_time, m_t, Xte_t, yte_t = run(df, tr_t, te_t, '時間切分 ≤113年 訓練 / 114年 測試')
    plot(m_t, Xte_t, yte_t, '時間切分', '時間切分_預測結果.png')

    out = {'random_split': res_random, 'time_split': res_time}
    (RESULTS / 'metrics.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    for r in (res_random, res_time):
        print(f"\n== {r['split']}  train {r['n_train']} / test {r['n_test']}  features {r['n_features']}")
        print('  train:', r['train'])
        print('  test :', r['test'])
        print('  top10:', r['top10_importance'])


if __name__ == '__main__':
    main()
