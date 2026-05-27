# eval.py
#%%
import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from prediction import quality_filter
import lightgbm as lgb

#%%
# パラメータ
n_days = 5
top_k = 5

#%%
# モデル読み込み
model = joblib.load('./models/lgbm_model.pkl')

#%%
# データ読み込み
test_folder = './test_data'
test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.npz')]

X_test_list = []
y_test_list = []
tickers_list = []
dates_list = []

for file in tqdm(test_files):
    data = np.load(file)

    dates = pd.to_datetime(data['date'])
    df = pd.DataFrame({
        'Open': data['open'],
        'High': data['high'],
        'Low': data['low'],
        'Close': data['close'],
        'Volume': data['volume']
    }, index=dates)

    df = quality_filter(df)  # フィルター適用

    if len(df) <= n_days:
        continue

    next_open = df['Open'].shift(-1)
    future_close = df['Close'].shift(-n_days)
    valid_idx = (~next_open.isna()) & (~future_close.isna())

    if valid_idx.sum() == 0:
        continue

    feature_matrix = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    next_open_arr = next_open.values
    future_close_arr = future_close.values
    date_arr = df.index.values

    for i in range(len(df)):
        if not valid_idx.iloc[i]:
            continue
        X_test_list.append(feature_matrix[i])
        y_test_list.append(future_close_arr[i] / next_open_arr[i])
        tickers_list.append(os.path.basename(file).replace('.npz', ''))
        dates_list.append(date_arr[i])

#%%
X_test = np.array(X_test_list)
y_test = np.array(y_test_list)
tickers_test = np.array(tickers_list)
dates_test = np.array(dates_list)

print(f"✅ テストデータ数: {len(X_test)} 件")

#%%
# 予測
y_pred = model.predict(X_test)

#%%
# 結果をDataFrameに
eval_df = pd.DataFrame({
    'date': dates_test,
    'ticker': tickers_test,
    'pred': y_pred,
    'target': y_test
})

#%%
# シミュレーション
total_return = 1.0
returns = []
wins = 0
trades = 0

unique_dates = sorted(eval_df['date'].unique())

for date in unique_dates:
    today_df = eval_df[eval_df['date'] == date].sort_values(by='pred', ascending=False)
    top5 = today_df.head(top_k)

    for _, row in top5.iterrows():
        ticker = row['ticker']
        buy_date = row['date']
        file_path = os.path.join(test_folder, f"{ticker}.npz")
        if not os.path.exists(file_path):
            continue

        data = np.load(file_path)
        dates = pd.to_datetime(data['date'])
        # シミュレーション内の1銘柄の価格再取得部分
        df_price = pd.DataFrame({
            'Open': data['open'],
            'Close': data['close'],
            'Volume': data['volume']
        }, index=dates)

        df_price = quality_filter(df_price)  # ✅ フィルター適用してもエラーにならない

        if buy_date not in df_price.index:
            continue

        try:
            buy_idx = df_price.index.get_loc(buy_date)
        except KeyError:
            continue

        sell_idx = buy_idx + n_days
        if sell_idx >= len(df_price):
            continue

        buy_price = df_price.iloc[buy_idx]['Open']
        sell_price = df_price.iloc[sell_idx]['Close']

        if buy_price == 0:
            continue

        ret = sell_price / buy_price
        returns.append(ret)
        total_return *= ret
        trades += 1
        if ret > 1.0:
            wins += 1

        if ret > 10:
            print(f"[異常リターン検出] {ticker} {buy_date}")
            print(f"Buy: {buy_price}, Sell: {sell_price}, Return: {ret:.2f}")

#%%
# 結果表示
if trades > 0:
    avg_return = (np.array(returns) - 1).mean() * 100
    win_rate = wins / trades * 100
    total_return_percentage = (total_return - 1) * 100

    print(f"✅ 総取引回数: {trades}")
    print(f"✅ 総合リターン: {total_return_percentage:.2f}%")
    print(f"✅ 平均リターン: {avg_return:.2f}%")
    print(f"✅ 勝率: {win_rate:.2f}%")
else:
    print("⚠️ 取引なし")
print(returns)