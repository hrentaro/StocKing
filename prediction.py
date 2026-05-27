#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, make_scorer, precision_recall_curve, auc
import investpy
import yfinance as yf
from io import BytesIO
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from tqdm import tqdm
import time
import logging
#%%
#---------------------------------------------------------------------
# 基本のパラメータ
#---------------------------------------------------------------------

# up_rate以上になるかどうかの判断
up_rate = 1.03

# pre_day日後の15時に売る
pre_day = 1

# 当日分のデータに(day_ago-1)日分以降のデータも加える (時系列を扱うことができないため)
day_ago = 3

# オリジナルスコアのパラメータ (空振り率の評価を1/r倍する。r=1でF1と等価)
r = 0.01

# モデルの保存先
modelfile = 'stock_model_rf1.pickle'




#---------------------------------------------------------------------
# データ集め
#---------------------------------------------------------------------
def get_all_japan_stock_tickers():
    """銘柄一覧を取得(Aがつくコードにも対応)

    Returns:
        list:日本の銘柄一覧(Aがつくものも含む)
    """
    # 最新のExcelファイルURL（都度更新の可能性あり）
    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
    response = requests.get(url)
    df = pd.read_excel(BytesIO(response.content), dtype=str)

    # 「内国株式」かつ「ETF・ETN」以外、コードが数字 or 'A' を含んで終わるもの
    valid = df[
        (df['市場・商品区分'].str.contains('内国株式')) &
        (~df['市場・商品区分'].str.contains('ETF・ETN'))
    ]

    # コードに.Tをつけて返す（4桁+任意のAなどに対応）
    tickers = valid['コード'].dropna().unique()
    tickers = [code + ".T" for code in tickers]
    return tickers

#---------------------------------------------------------------------
# データ集め
# (yfinanceを使用)
#---------------------------------------------------------------------

def get_historical_data_yfinance(ticker, start=None, end=None, progress=True):
    """
    1銘柄の過去株価データを取得する。
    """
    # ダウンロード（ログを抑えるため、progress=False）
    df = yf.download(ticker, start=start, end=end, progress=progress)

    if df.empty:
        return None

    # 列がマルチインデックスになってる場合、修正
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    df.reset_index(inplace=True)

    # 列名を統一（銘柄名除去）
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    return df


def download_all_stocks(tickers,
                        start=None,
                        end=None,
                        save_path="./data/stock_prices",
                        save_format="csv"):
    """
    全銘柄の株価データをダウンロードして保存する。
    """
    # yfinanceのロガーを無効化
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    os.makedirs(save_path, exist_ok=True)

    if tickers == 'all':
        tickers = get_all_japan_stock_tickers()

    failed_tickers = []

    # tqdmバー付きでダウンロード
    for ticker in tqdm(tickers, desc="Downloading stock data"):
        df = get_historical_data_yfinance(ticker, start=start, end=end, progress=False)
        if df is not None:
            save_file = os.path.join(save_path, ticker.replace('.T', '') + ('.csv' if save_format == 'csv' else '.npz'))
            if save_format == 'csv':
                df.to_csv(save_file, index=False)
            else:
                np.savez_compressed(save_file, date=df['Date'].values, open=df['Open'].values,
                                    high=df['High'].values, low=df['Low'].values,
                                    close=df['Close'].values, volume=df['Volume'].values)
        else:
            failed_tickers.append(ticker.replace('.T', ''))
            print(f"\n失敗: {ticker.replace('.T', '')}", end=", ")
    time.sleep(0.5)

    # 最後にまとめて
    if failed_tickers:
        print("\n\nダウンロード失敗銘柄:")
        print(",".join(failed_tickers))
    else:
        print("\n全銘柄ダウンロード成功!")

#---------------------------------------------------------------------
# データ集め
# (investpyを使用)
#---------------------------------------------------------------------
# stock: str(code)
def investpy_dl(stock, from_date, to_date):
    stock_data = investpy.get_stock_historical_data(
        stock = stock,
        country = 'japan',
        from_date = from_date,
        to_date = to_date,
        order = 'ascending')

    return stock_data[['Open','High','Low','Close','Volume']]





#---------------------------------------------------------------------
# test用データ集め
# (investpyを使用)
#---------------------------------------------------------------------
def test_dl(stock_number):
    # Caution!!
    # dd/mm/yyyy
    stock_data = investpy.get_stock_historical_data(
        stock=stock_number,
        country='japan',
        from_date='06/11/2020',
        to_date='06/04/2021',
        order='ascending')

    stock_data = stock_data[['Open','High','Low','Close','Volume']]
    stock_data.to_csv('test/' + str(stock_number) + '.csv')
    return None

#---------------------------------------------------------------------
# データ集め
# (日本経済新聞から本日分のみ取得)
#---------------------------------------------------------------------
def nikkei_dl(stock, days=10, headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15'}):
    url = 'https://www.nikkei.com/nkd/company/history/dprice/?scode={}&ba=1'.format(stock)
    r = requests.get(url, headers = headers)
    html = r.content
    soup = BeautifulSoup(html, "html.parser")
    temp1 = soup.find_all(class_="a-taR a-wordBreakAll")[:6*days]
    # [o, h, l, c, v, ad_c]のリスト
    if len(temp1) > 0:
        return np.fliplr(np.array([float(a.text.replace(',', '')) for a in temp1]).reshape([days,6]).T)
    else:
        return np.array([[],[],[],[],[],[]])


#---------------------------------------------------------------------
# 株価データの品質フィルタリング
#---------------------------------------------------------------------

def quality_filter(df: pd.DataFrame, min_history_days: int = 5) -> pd.DataFrame:
    """
    データクリーニング用フィルター:
    1. 出来高が0の行を除外
    2. 始値・高値・安値・終値がすべて同じ行を除外
    3. 異常値（前日比で始値または終値が3倍以上または1/3以下）を検出し、その日以前のデータを全除外
    """
    df = df.copy()

    # 必須カラムが存在しない場合はそのまま返す
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_columns.issubset(df.columns):
        return df

    # 出来高が0の行を除外
    df = df[df['Volume'] > 0]

    # OHLC全てが同じ値の行を除外
    df = df[~((df['Open'] == df['High']) &
              (df['Open'] == df['Low']) &
              (df['Open'] == df['Close']))]

    # 異常日（前日比でOpenやCloseが3倍以上 or 1/3以下）を検出
    open_ratio = df['Open'] / df['Open'].shift(1)
    close_ratio = df['Close'] / df['Close'].shift(1)

    abnormal = (open_ratio >= 3) | (open_ratio <= 1/3) | \
               (close_ratio >= 3) | (close_ratio <= 1/3)

    if abnormal.any():
        # 異常な行の最初のインデックスを取得
        first_abnormal_idx = abnormal[abnormal].index[0]
        # その日を含め、前 min_history_days 日も含めて除外（それ以前を除去）
        idx_pos = df.index.get_loc(first_abnormal_idx)
        df = df.iloc[idx_pos + 1:]

    return df

#---------------------------------------------------------------------
# データから特徴量を計算
#---------------------------------------------------------------------
def create_features_with_target(df, n=5):
    """
    特徴量＋ターゲット作成。

    Args:
        df (pd.DataFrame): ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']のDataFrame
        n (int): 何日後の終値をターゲットにするか

    Returns:
        pd.DataFrame: 特徴量+ターゲット列付きDataFrame
    """
    df = df.copy()

    # 翌営業日の始値
    df['NextOpen'] = df['Open'].shift(-1)
    # n日後の終値
    df['nClose'] = df['Close'].shift(-n)

    # ターゲット作成（n日後の終値 ÷ 翌日の始値）
    df['Target'] = df['nClose'] / df['NextOpen']

    # 不要な行を除去（未来が見えてしまうので）
    df.dropna(subset=['Target'], inplace=True)

    return df

#---------------------------------------------------------------------
# データから特徴量を計算 (cal_params.pyに移行したが一応残しておく)
#---------------------------------------------------------------------
def stock_params(f, pre_day=1, day_ago=3):
    #print(f)
    # base日後からのデータを取得
    # 75日移動平均を使うので、base>75とする
    base = 76
    num_index = 21
    temp1 = np.load(f, allow_pickle=True)
    # index ... ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    # 当日の終値 / 当日の始値という指標はpre_day=1のときは指標0と被る
    if pre_day >= 2:
        num_index += 1

    days = len(temp1['Open'])
    # 短すぎると計算エラー
    # 安すぎるものは1円上がるだけでもup_rateの目標を達成してしまうので100円以上を条件にする
    if len(temp1['Open']) <= base or np.min(temp1['Low']) <= 100:
        return np.array([[] for _ in range(day_ago*num_index)])



    # 移動平均 (終値)
    ave05_c = np.convolve(temp1['Close'], np.ones(5)/5, mode='full')[:-4]
    ave10_c = np.convolve(temp1['Close'], np.ones(10)/10, mode='full')[:-9]
    ave25_c = np.convolve(temp1['Close'], np.ones(25)/25, mode='full')[:-24]
    ave75_c = np.convolve(temp1['Close'], np.ones(75)/75, mode='full')[:-74]
    # 移動平均 (始値)
    ave05_o = np.convolve(temp1['Open'], np.ones(5)/5, mode='full')[:-4]
    ave25_o = np.convolve(temp1['Open'], np.ones(25)/25, mode='full')[:-24]
    ave75_o = np.convolve(temp1['Open'], np.ones(75)/75, mode='full')[:-74]

    # 移動標準偏差
    #w_std10 = np.ones(10)/10
    #std10 =  np.sqrt(np.abs(np.convolve(temp1['Close']**2, w_std10, mode='full') - (np.convolve(temp1['Close'], w_std10, mode='full'))**2))[:-9]
    std10 = moving_std(temp1['Close'], 10)
    std25 = moving_std(temp1['Close'], 25)

    # 指数平滑移動平均
    ema12 = EMA(temp1['Close'], 12)
    ema26 = EMA(temp1['Close'], 26)
    macd = ema12 - ema26
    signal = EMA(macd, 9)


    # まずはすべての要素が0の行列を作っておく
    temp3 = np.zeros((len(temp1['Close'])-base, num_index))

    # Stochastic Oscillator
    L14 = -rolled_max(-temp1['Low'], 14)
    H14 =  rolled_max(temp1['High'], 14)
    HL14 = H14 - L14
    SO = np.divide(temp1['Close']-L14, HL14, out=np.ones_like(HL14)*0.5, where=HL14>0.1)

    # Ichimoku
    L07 = -rolled_max(-temp1['Low'], 7)
    H07 =  rolled_max(temp1['High'], 7)
    kijun = H14 + L14
    tenkan = H07 + L07
    senko2 = rolled_max(temp1['High'], 28) - rolled_max(-temp1['Low'], 28)

    # 当日の終値 / (pre_day-1)日前の始値
    # pre_day日だけ未来にずらしたら答え合わせができる
    temp3[:, 0] = temp1['Close'][base:days] / temp1['Open'][base-pre_day+1:days-pre_day+1]
    # 当日の始値 / 前日の終値
    temp3[:, 1] = temp1['Open'][base:days] / temp1['Close'][base-1:days-1]
    if min(temp3[:,1])<0.7:
        return np.array([[] for _ in range(day_ago*num_index)])
    # 当日の高値 / 当日の終値
    temp3[:, 2] = temp1['High'][base:days] / temp1['Close'][base:days]
    # 当日の安値 / 当日の終値
    temp3[:, 3] = temp1['Low'][base:days] / temp1['Close'][base:days]
    # 当日の終値 / 前日の終値
    temp3[:, 4] = temp1['Close'][base:days] / temp1['Close'][base-1:days-1]
    if min(temp3[:,4])<0.7:
        return np.array([[] for _ in range(day_ago*num_index)])
    # 当日の終値 / 5日平均
    temp3[:, 5] = temp1['Close'][base:days] / ave05_c[base:days]
    # 当日の終値 / 25日平均
    temp3[:, 6] = temp1['Close'][base:days] / ave25_c[base:days]
    # 当日の終値 / 75日平均
    temp3[:, 7] = temp1['Close'][base:days] / ave75_c[base:days]
    # golden cross & dead cross
    temp3[:, 19] = ave05_c[base-1:days-1] / ave25_c[base-1:days-1]
    # 当日の始値 / 前日の始値
    #temp3[:, 8] = temp1['Open'][base:days] / temp1['Open'][base-1:days-1]
    #if min(temp3[:,8])<0.7:
        #return np.array([[] for _ in range(day_ago*num_index)])
    # 当日の始値 / 5日平均
    #temp3[:, 9] = temp1['Open'][base:days] / ave05_o[base:days]
    # 当日の始値  / 25日平均
    #temp3[:,10] = temp1['Open'][base:days] / ave25_o[base:days]
    # 当日の始値 / 75日平均
    #temp3[:,11] = temp1['Open'][base:days] / ave75_o[base:days]
    #temp3[:,11] = ave05_o[base:days] / ave25_o[base:days]
    # 10日移動標準偏差 / 10日平均
    temp3[:,13] = np.round(std10[base:days] / ave10_c[base:days], decimals=4)
    # 25日移動標準偏差 / 25日平均
    temp3[:,14] = np.round(std25[base:days] / ave25_c[base:days], decimals=4)

    # ボリンジャーバンド
    temp3[:,15] = temp1['Close'][base:days] / (2*std10[base:days] + ave10_c[base:days])
    temp3[:,16] = temp1['Close'][base:days] / (-2*std10[base:days] + ave10_c[base:days])

    # Moving Average Convergence Divergence
    temp3[:,8] = temp1['Close'][base:days] / ema12[base:days]
    temp3[:,9] = temp1['Close'][base:days] / ema26[base:days]
    #temp3[:,10] = macd[base:days] / temp1['Close'][base:days]
    # SignalLine
    #temp3[:,11] = (signal-macd)[base:days] / temp1['Close'][base:days]
    #temp3[:,12] = macd[base:days] / macd[base-1:days-1]

    # Ichimoku
    temp3[:,10] = tenkan[base:days] / kijun[base:days]
    temp3[:,11] = temp1['Close'][base:days] / temp1['Close'][base-14:days-14]
    temp3[:,12] = temp1['Close'][base-14:days-14] / (tenkan+kijun)[base:days]
    temp3[:,20] = temp1['Close'][base-14:days-14] / senko2[base:days]

    # Stochastic Oscillator
    temp3[:,17] = SO[base:days]
    #temp3[:,20] = np.where(HL14>0.5, (temp1['Close'] - L14) / HL14, 0.5)[base:days]

    # Williams %R
    #temp3[:,18] = np.divide(H14 - temp1['Close'], HL14, out=np.ones_like(HL14)*0.5, where=HL14>0.1)[base:days]

    # RSI
    temp3[:,18] = RSI(temp1['Close'], 14)[base:days]

    # 曜日 (2000-01-01を基準にした日付)
    #dates_index = np.array(temp1.index)
    #oldest = date.fromisoformat(temp2[0,0])
    #base_date = date.fromisoformat('2000-01-01')
    #temp3[:, 15] = (dates_index[base:days] + (oldest - base_date).days) % 7

    # 当日の終値 / 当日の始値 (pre_day>=2のときのみ)
    if pre_day >= 2:
        temp3[:, num_index-1] = temp1['Close'][base:days] / temp1['Open'][base:days]

    # tempX : 現在の企業のデータ
    tempX = np.zeros((len(temp3), day_ago*num_index))

    # 日にちごとに横向きに（day_ago）分並べる
    # sckit-learnは過去の情報を学習できないので、複数日（day_ago）分を特微量に加える必要がある
    # 注：tempX[0:day_ago]分は欠如データが生まれる
    for s in range(num_index):
        for i in range(day_ago):
            tempX[i:, day_ago*s+i] = temp3[:len(temp3)-i,s]

    return tempX[day_ago:]


#---------------------------------------------------------------------
# モデル設計(データ調整)
#---------------------------------------------------------------------
def stock_design(X, up_rate, pre_day=1, compression=True):
    # 何日後の値段の差を予測するのか
    pre_day = 1
    # y :  当日の終値 / pre_day日前の終値
    y = np.zeros(len(X))
    y = X[pre_day:,0]
    X = X[:-pre_day]

    # zとwの定義
    z = np.where(y>up_rate, 1, 0)
    w = np.where(y>1, 1, 0)

    return [X, y, z, w]

def classify_undersample(X, labels, weights=None):
    X_ = X
    return X_




#---------------------------------------------------------------------
# 答え合わせ (stockとdateを入力すると当日とpre_day日後の終値を返す)
#---------------------------------------------------------------------
# 答え合わせ
def scoring(stock, date, pre_day=1, data_dir='for_use/'):
    f = data_dir + stock + '.npz'
    temp1 = np.load(f)
    dates = temp1['Date'][-len(temp1['Open']):]
    x = np.where(dates == date)[0][0]
    today = temp1['Open'][x+1]
    target_day = temp1['Close'][x+pre_day]
    return [today, target_day, target_day/today]

#---------------------------------------------------------------------
# スコア集
# オリジナルスコアを考える際の注意点: greater_is_betterをFalseにするとscoreに負号がついて出力される
#---------------------------------------------------------------------
def scoring_dic(name=None, labels=None, weights=None, t=None, square_weights=None):
    if name == None:
        print('Define scoring name to input prediction.scoring_dic !!!!!')
        return None
    elif name == 'f1':
        return 'f1'
    elif name == 'accuracy':
        return 'accuracy'
    elif name == 'recall':
        return 'recall'
    elif name == 'precision':
        return 'precision'
    elif name == 'average_precision':
        return 'average_precision'
    elif name == 'transformed_accuracy':
        return make_scorer(transformed_accuracy, greater_is_better=True, r=r)
    elif name == 'transfomed_f1':
        return make_scorer(transformed_f1, greater_is_better=True, r=r)
    elif name == 'capital':
        return make_scorer(capital, greater_is_better=True)
    elif name == 'p_r_production':
        return make_scorer(precision_recall_production, greater_is_better=True)
    elif name == 'top_precision':
        return make_scorer(top_precision, greater_is_better=True,
                           needs_proba=True, needs_threshold=False, t=t)
    elif name == 'multi_precision':
        return make_scorer(multi_precision, greater_is_better=True,
                           needs_proba=False, needs_threshold=False,
                           labels=labels, weights=weights)
    elif name == 'multi_capital':
        return make_scorer(multi_capital, greater_is_better=True,
                           labels=labels, square_weights=square_weights)
    else:
        print('No scoring name on prediction.scoring_dic !!!!!')
        exit()
    return None


# オリジナルスコア (tnとfpを1/r倍したときのaccuracy)
def transformed_accuracy(y_true, y_pred, r=r):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.flatten()
    return (tn + r*tp) / (tn + fp + r*tp + r*fn)

# オリジナルスコア (fpを1/r倍したときのf値)
def transformed_f1(y_true, y_pred, r=r):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.flatten()
    return 2*r*tp / (fp + 2*r*tp + r*fn)

# オリジナルスコア (買ったときの株価変化の算術平均)
def capital(y_score, y_pred):
    s = np.count_nonzero(y_pred)
    if s > 0:
        return np.dot(y_score, y_pred)/s
    else:
        return 1.0


# オリジナルスコア (precision*recall)
def precision_recall_production(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.flatten()
    return (tp * tp) / ((tp+fp) * (fn+tp))


# 上位から割合tの部分だけを予想する
def top_precision(y_true, y_pred, t=0.05):
    #print(y_true,y_pred)
    pred_p = int(len(y_true)*t)
    if pred_p == 0:
        return 0.0
    b = np.argsort(-y_pred[:])[:pred_p]
    y_try = y_true[b]
    tp = np.count_nonzero(y_try)
    return tp / pred_p

# オリジナルスコア (多値分類用、買うものの中で得点をつける)
def multi_precision(y_true, y_pred, labels, weights, no_pred=0.0):
    cm = np.array(confusion_matrix(y_true, y_pred, labels=labels)).T
    result = cm[-1]
    pred_p = np.sum(result)
    if pred_p > 0:
        return np.dot(result, weights) / np.sum(result)
    else:
        return no_pred

# オリジナルスコア (多値分類用、正解から離れているほどペナルティが大きくなる)
def multi_capital(y_true, y_pred, labels, square_weights=None):
    cm = np.array(confusion_matrix(y_true, y_pred, labels=labels))
    n = len(labels)
    try:
        value = cm * square_weights
    except:
        square_weights = 1.0 - np.abs([np.array(labels) - i for i in range(n)]) / (n-1)
        value = cm * square_weights

    return np.sum(value) / len(y_true)


#---------------------------------------------------------------------
# stock_params 公式集
#---------------------------------------------------------------------
# 移動標準偏差
# データaに対してd日移動偏差を求める
def moving_std(a, d):
    w = np.ones(d)/d
    moving_v = np.convolve(a*a, w, mode='full')[:-d+1] - (np.convolve(a, w, mode='full')[:-d+1])**2
    return np.sqrt(moving_v, out = np.zeros_like(a), where = moving_v>0)

# EMA (Exponential Moving Average)
def EMA(a, d):
    n = len(a)
    alpha = 2 / (1+d)
    # (1-alpha) ** [0, 1, ..., n-1, n]を計算
    w = np.logspace(0, n, num=n+1, base=1-alpha)
    return np.convolve(a, w[:-1]*alpha, mode='full')[:-n+1] + a[0] * w[1:]

# RSI (Relative Strength Index)
def RSI(a, d):
    b = np.convolve(a, np.array([1,-1]), mode='full')[:-1]
    a_up = np.where(b>=0, a, 0)
    a_abs = np.abs(b)
    w = np.ones(d)
    frac1 = np.convolve(a_up,w,mode='full')[:-d+1]
    frac2 = np.convolve(a_abs,w,mode='full')[:-d+1]

    return np.divide(frac1, frac2, out=np.ones_like(frac2)*0.5, where=frac2>0.1)

# maximum over the past n days
# input -a instead of ain order to get rolled minimum
def rolled_max(a,d):
    i = 1
    temp1 = np.concatenate([np.ones(d-1)*a[0],a])
    n = len(a)
    return np.maximum.reduce(np.lib.stride_tricks.as_strided(temp1, shape=(d, n), strides=(8*i, 8*i)))



#---------------------------------------------------------------------
# 描画ツール
#---------------------------------------------------------------------

# ヒストグラム
def plot_hist(y_test, y_prob, fig=None, ax=None, t=0.05, h=0.1):
    #print(y_test, y_prob)
    if fig==None or ax==None:
        fig = plt.figure(figsize=(10,6),tight_layout=True)
        ax = fig.add_subplot(1,1,1)
    ax.set_title('Hist')
    ax.set_xlabel('up_rate')
    ax.set_ylabel('Hist')
    ax.grid(axis='both', zorder=0)
    #ax.set_axisbelow(True)
    ax.set_yscale('log')
    ax.set_xlim(0.95, 1.05)
    ax.hist(x=y_test, bins=9, range=(0.955, 1.045), color=cm.jet(0.0), zorder=0.0)
    for threshold in np.arange(0.0, 1.0, h):
        y_result = y_test[y_prob >= threshold]
        ax.hist(x=y_result, bins=9, range=(0.955, 1.045), color=cm.jet(threshold), zorder=threshold)
    pred_p = int(len(y_test)*t)
    b = np.argsort(-y_prob[:])[:pred_p]
    threshold = y_prob[b[-1]]
    hist, bins = np.histogram(a=y_test[b], bins=9, range=(0.955, 1.045), normed=False, weights=None, density=None)
    bins = np.convolve(bins, [0.5, 0.5], mode='valid')
    print(hist)
    print(bins)
    ax.plot(bins, hist, color='black', marker='o', linestyle='solid', label='Top {:.2f} selection\n(threshold = {:.2f})'.format(t,threshold), zorder=2.5)
    #fig.colorbar(mappable=cm.ScalarMappable(norm=None, cmap=cm.jet), cax=ax2, ax=ax2)
    ax.legend(loc='lower left')
    return threshold



# precision_recall曲線
def plot_pr_curve(y_test, y_prob, ax=None, h=0.05):
    if ax==None:
        fig = plt.figure(figsize=(10,6),tight_layout=True)
        ax = fig.add_subplot(1,1,1)
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis='both', zorder=0)
    ax.set_axisbelow(True)

    # PR曲線を描くためのパラメータ
    precision, recall, thresholds = precision_recall_curve(y_true=y_test, probas_pred=y_prob)
    auc_score = auc(recall, precision)
    ax.plot(recall, precision, label='PR', color='black', zorder=1)
    ax.plot([0,1], [1,1], linestyle='--', label='Ideal', color='red', zorder=0)
    # 閾値を10%刻みでプロット
    print('threshold  recall  precision')
    t0 = 0
    for t in np.arange(1.0, thresholds.min(), -h):
        t1 = np.argmin(np.abs(thresholds - t))
        if t1 < t0:
            ax.plot(recall[t1], precision[t1],'x', color=cm.jet(t1))
            ax.annotate('{:.2f}'.format(thresholds[t1]), xy=(recall[t1]+0.01, precision[t1]+0.02), color=cm.jet(t1))
        t0 = t1
        print('     {:.2f}    {:.2f}       {:.2f}'.format(thresholds[t1], recall[t1], precision[t1]))
    ax.legend(loc='upper right')
    ax.set_title('PR curve (area = {:.4f})'.format(auc_score))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    return None


# ヒートマップ
def plot_test_heatmap(grid,ax=None,scoring_name='Score',
                      param_name1=None,param_name2=None,param_name3=None,
                      param_list1=None,param_list2=None,param_list3=None):
    if ax==None:
        fig = plt.figure(figsize=(10,6),tight_layout=True)
        ax = fig.add_subplot(1,1,1)
    scores = np.array(grid.cv_results_['mean_test_score']).reshape(len(param_list2), -1)
    best_params = grid.best_params_
    # plot the mean cross-validation scores
    img1 = ax.pcolor(scores, cmap='viridis', vmin=None, vmax=None)
    img1.update_scalarmappable()
    ax.set_xlabel(param_name1)
    ax.set_ylabel(param_name2)
    ax.set_xticks(np.arange(len(param_list1)) + 0.5)
    ax.set_yticks(np.arange(len(param_list2)) + 0.5)
    ax.set_xticklabels(param_list1)
    ax.set_yticklabels(param_list2)

    for p, color, value in zip(img1.get_paths(), img1.get_facecolors(), img1.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, '{:.4f}'.format(value), color=c, ha='center', va='center')

    ax.set_title('{:} heatmap\nBest params: {:}'.format(scoring_name, best_params))
    return None


# 学習曲線
def plot_learning_curve(grid, X_train, y_train, n='?', ax=None, train_sizes=[0.01, 0.03, 0.1], cv=2, scoring_name='Score', random_state=None):
    if ax==None:
        fig = plt.figure(figsize=(10,6),tight_layout=True)
        ax = fig.add_subplot(1,1,1)
    ax.set_title('Learning curve n = {:}'.format(n))
    ax.set_xlabel('Training_examples')
    ax.set_ylabel(scoring_name)
    ax.set_xlim(xmin=0, xmax=len(y_train))
    ax.grid(axis='both', zorder=0)
    ax.set_axisbelow(True)

    # 学習曲線を描くための計算
    train_sizes_, train_scores, test_scores = learning_curve(
        grid, X_train, y_train, cv=cv,
        train_sizes=train_sizes, return_times=False, verbose=0, random_state=random_state)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    #fit_times_mean = np.mean(fit_times, axis=1)
    #fit_times_std = np.std(fit_times, axis=1)

    ax.fill_between(train_sizes_, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                    color='red')
    ax.fill_between(train_sizes_, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color='green')
    ax.plot(train_sizes_, train_scores_mean, 'o-', color='red',
             label='Training score')
    ax.plot(train_sizes_, test_scores_mean, 'o-', color='green',
             label='Cross-validation score')
    ax.legend(loc='upper right')
    return None


# parameter重要度
def plot_importances(grid, ax=None, day_ago=day_ago):
    if ax==None:
        fig = plt.figure(figsize=(10,6),tight_layout=True)
        ax = fig.add_subplot(1,1,1)
    colorlist = ['red', 'blue', 'green', 'orange', 'purple']
    best_grid = grid.best_estimator_
    importances = best_grid.feature_importances_
    indices = np.argsort(importances)
    features = [i for i in range(len(indices))]
    print('importance: ', importances)
    print('indices:', indices)
    for i in features:
        ax.bar(i, importances[i], width=0.8, color=colorlist[i%day_ago], align='center')
    #ax.set_xticks(range(len(indices)), features)
    #タイトルの設定
    #ax.set_title('Importance of every features')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_xticks(np.arange(features[0], features[-1]+0.5, 3))
    ax.set_xlim(features[0]-0.5, features[-1]+0.5)
    ax.set_title('Importance of Params')
    return None


# %%
