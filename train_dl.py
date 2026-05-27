import os
import shutil
import prediction
import investpy
import tqdm
import random
import glob
import time
import sys
import numpy as np


# 定期的にアップデートする。
# train用 dd/mm/yyyy
from_date = '01/01/2010'
to_date = '31/03/2023'

# 取得する会社の数
n = 4000

# データ取得する最低日数(これ未満の長さのデータは無視)
day_long = 80

#---------------------------------------------------------------------
# ここから下は編集しない
#---------------------------------------------------------------------
test_files  = glob.glob('train/*.csv')
if len(test_files) >= n:
    print('you have already', len(test_files), 'files (you want ', n, 'files).')
print('really want to re-download? [y/n]')
if input() == 'y':
    pass
else:
    print('interrupted.')
    sys.exit()

# ディレクトリを作り直す
data_dir = 'train_data/'
if os.path.isdir(data_dir):
    shutil.rmtree(data_dir)
os.mkdir(data_dir)

# 取得する銘柄の決定
stocks_list = investpy.stocks.get_stocks_list(country='japan')
stocks_sample = random.sample(stocks_list, len(stocks_list))

print(str(n)+' stocks')

listing = []
j = 0

# stockはstrであることに注意する
for stock in tqdm.tqdm(stocks_sample):
    try:
        stock_data = prediction.investpy_dl(stock, from_date, to_date)
        time.sleep(2)
        if len(stock_data['Close']) >= day_long:
            #[o,h,l,c,v]の順
            npz = stock_data.to_numpy().T
            dates = np.array(stock_data.index)
            # 出来高0の日の修正
            o = np.where(np.isnan(npz[0]), npz[3], npz[0])
            h = np.where(np.isnan(npz[1]), npz[3], npz[1])
            l = np.where(np.isnan(npz[2]), npz[3], npz[2])
            c = npz['Close']
            v = npz['Volume']
            np.savez(data_dir + stock + '.npz',
                     Date = dates,
                     Open = o,
                     High = h,
                     Low = l,
                     Close = c,
                     Volume = v)

        else:
            pass
            #print(stock, 'short')
    except:
        listing.append(stock)
        #print(stock, 'err')

print('listing: ', len(listing))
print('')
'''
listing = []
j = 0
columns_d = ['Open','High','Low','Close']
search_list = [stock + '.JP' for stock in stocks_sample[:n+50]]
cut_list = list(np.array_split(search_list, (len(search_list)-1)//dlsize+1))
for minilist in tqdm(cut_list):
    temp = pandas_datareader.stooq.StooqDailyReader(
        symbols = minilist,
        start = from_date, end = to_date,
        retry_count=3, pause=None, session=None, chunksize=25).read()
    # 日経新聞からスクレイピングして保存
    for search in minilist:
        time.sleep(1)
        try:
            stock = search[:-3]
            data = temp[[('Open',search),('High',search),('Low',search),('Close',search)]]
            data = np.array(data.set_axis(columns_d))


print('done!')
'''
