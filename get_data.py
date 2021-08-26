import ccxt
import pandas as pd
import datetime
import ta
import matplotlib.pyplot as plt
from ta.momentum import *
import pandas_ta as tap
import talib
import requests
import time 
import numpy as np
from binance.client import Client

exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET',
    'timeout': 30000,
    'enableRateLimit': True,
})
# client = Client(api_key, api_secret)
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
    "Accept-Encoding": "*",
    "Connection": "keep-alive"
}
# markets=exchange.load_markets()
# symbol_trade_set=[]

# for market in markets:
#     if '/USDT' in market:
#         symbol_trade_set.append(market)   
symbol_trade_set=["BTC/USDT","ETH/USDT","LINK/USDT","XRP/USDT",
                  "EOS/USDT","LTC/USDT","TRX/USDT","ETC/USDT","XLM/USDT",
                  "XMR/USDT","DASH/USDT","ATOM/USDT","ONT/USDT",
                  "IOTA/USDT","BAT/USDT","NEO/USDT","IOST/USDT","ALGO/USDT",
                  "COMP/USDT","OMG/USDT","SXP/USDT","KAVA/USDT","BAND/USDT",
                  "WAVES/USDT","SNX/USDT","YFI/USDT","CRV/USDT","SRM/USDT",
                  "EGLD/USDT","STORJ/USDT","UNI/USDT","AVAX/USDT","HNT/USDT",
                  "ENJ/USDT","TOMO/USDT","KSM/USDT","FIL/USDT","RSR/USDT",
                  "LRC/USDT","MATIC/USDT","BEL/USDT","AXS/USDT","ZEN/USDT","SKL/USDT",
                  "GRT/USDT","CHZ/USDT","UNFI/USDT","REEF/USDT","XEM/USDT","COTI/USDT",
                  "MANA/USDT","HBAR/USDT","ONE/USDT","CELR/USDT","OGN/USDT","NKN/USDT",
                  "SC/USDT","BCH/USDT","ADA/USDT","ZEC/USDT","VET/USDT","QTUM/USDT",
                  "ZIL/USDT","ZRX/USDT","RLC/USDT","MKR/USDT","DOT/USDT","YFII/USDT",
                  "SUSHI/USDT","BZRX/USDT","ICX/USDT","FLM/USDT","NEAR/USDT","AAVE/USDT",
                  "OCEAN/USDT","ALPHA/USDT","ANKR/USDT","LIT/USDT","SFP/USDT","CHR/USDT",
                  "ALICE/USDT","LINA/USDT","STMX/USDT","MTL/USDT","BTT/USDT","XTZ/USDT",
                  "DOGE/USDT","TRB/USDT","RUNE/USDT","BLZ/USDT","REN/USDT",
                  "CVC/USDT","CTK/USDT","SAND/USDT","BTS/USDT","DODO/USDT","DENT/USDT",
                  "DGB/USDT","THETA/USDT","KNC/USDT","BAL/USDT","SOL/USDT","1INCH/USDT",
                  "RVN/USDT","FTM/USDT","LUNA/USDT","ICP/USDT","BAKE/USDT","AKRO/USDT","HOT/USDT"]

def get_data(tf):
    
    for symbol in symbol_trade_set:
        
        bars=exchange.fetch_ohlcv(symbol,timeframe=tf)
        df=pd.DataFrame(bars,columns=['timestamps','open','high','low','close','volume'])
        df['candle']=df['close']-df['open']
        df['symbol']=symbol.split('/')[0]
        df['time']=pd.to_datetime(df.timestamps,unit='ms')
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA100'] = df['close'].ewm(span=100, adjust=False).mean()
        df['EMA150'] = df['close'].ewm(span=150, adjust=False).mean()
        df['logClose']= np.log(df['close'])
        
        # df['long_green_candle']=np.logical_and(df['low']<df['EMA50'] , df['low']<df['EMA100']) & np.logical_and(df.low<df.EMA150 , df.close>df.EMA50) &  np.logical_and(df.close>df.EMA100 , df.close>df.EMA150)
        
        df['distFromEMA50']=((df.low-df.EMA50)/df.low)*100
        df['distFromEMA100']=((df.low-df.EMA100)/df.low)*100
        df['distFromEMA150']=((df.low-df.EMA150)/df.low)*100
    
        df['distFromEMA50']=((df.low-df.EMA50)/df.low)*100
        df['distFromEMA100']=((df.low-df.EMA100)/df.low)*100
        df['distFromEMA150']=((df.low-df.EMA150)/df.low)*100
        
        BB=ta.volatility.BollingerBands(df.close,window= 20, window_dev= 2, fillna=False)
        df['Hband_bollinger']=BB.bollinger_hband()
        df['HbandIND_bollinger']=BB.bollinger_hband_indicator()
        df['Lband_bollinger']=BB.bollinger_lband()
        df['LbandIND_bollinger']=BB.bollinger_lband_indicator()
        df['Wband_bollinger']=BB.bollinger_wband()
        
        DC=ta.volatility.DonchianChannel(df.high,df.low,df.close,window= 20, fillna=False)
        df['dc_Hband']=DC.donchian_channel_hband()
        df['dc_Lband']=DC.donchian_channel_lband()
        df['dc_Wband']=DC.donchian_channel_wband()
        
        MFI=ta.volume.MFIIndicator(df.high,df.low,df.close,df.volume,window= 14, fillna=False)
        df['MFI']=MFI.money_flow_index()
        
        VWAP=ta.volume.VolumeWeightedAveragePrice(df.high,df.low,df.close,df.volume,window= 14, fillna=False)
        df['VWAP']=VWAP.volume_weighted_average_price()
        
        MFI=ta.volume.money_flow_index(df.high,df.low,df.close,df.volume,window= 14, fillna=False)
        df['MFI']=MFI
    
        df['RSI']=tap.rsi(df["close"], length=14)
        ##########             candle pattern           ############
        df['doji']=tap.cdl_doji(df.open,df.high,df.low,df.close)/100
        df['Two crows']=talib.CDL2CROWS(df.open,df.high,df.low,df.close)/100
        df['Hammer']=talib.CDLHAMMER(df.open,df.high,df.low,df.close)/100
        df['three_black_crows']=talib.CDL3BLACKCROWS(df.open,df.high,df.low,df.close)/100
        df['HANGINGMAN']=talib.CDLHANGINGMAN(df.open,df.high,df.low,df.close)/100
        df['HARAMICROSS']=talib.CDLHARAMICROSS(df.open,df.high,df.low,df.close)/100
        
        df['ADX']=talib.ADX(df.high, df.low, df.close, timeperiod=14)
        df['ADXR']=talib.ADXR(df.high, df.low, df.close, timeperiod=14)
        df['ATR']=talib.ATR(df.high, df.low, df.close, timeperiod=14)
        
        df['HTP']=talib.HT_DCPERIOD(df.close)
        df['HTPh']=talib.HT_DCPHASE(df.close)
        df['HTS'],df['HTSl']=talib.HT_SINE(df.close)
        
        df['BETA']=talib.BETA(df.high, df.low, timeperiod=5)
        df['CORREL']=talib.CORREL(df.high, df.low, timeperiod=30)
        df['LINEARREG']=talib.LINEARREG(df.close, timeperiod=14)
        df['LINEARREG_ANGLE']=talib.LINEARREG_ANGLE(df.close, timeperiod=14)
        
        df['LINEARREG_INTERCEPT']=talib.LINEARREG_INTERCEPT(df.close, timeperiod=14)
        df['LINEARREG_SLOPE']=talib.LINEARREG_SLOPE(df.close, timeperiod=14)
        df['STDDEV']=talib.STDDEV(df.close, timeperiod=5,nbdev=1)
        df['TSF']=talib.TSF(df.close, timeperiod=14)
        df['VAR']=talib.VAR(df.close, timeperiod=5,nbdev=1)
        
        Aroon=ta.trend.AroonIndicator(df.close,window= 25, fillna=False)
        df['Aroon_down']=Aroon.aroon_down()
        df['Aroon_up']=Aroon.aroon_up()
        df['Aroon_indicator']=Aroon.aroon_indicator()
        
        name='D:/csv/all/'+symbol.split('/')[0]+'_'+tf+'.csv'
        df.to_csv (name, index = False, header=True)   
        
        
        
        # time.sleep(0.5)
        # if df.RSI[499]<30:
        #     print('RSI is lower than 30 for: ',symbol, 'at "15min" timeframe at: ',datetime.datetime.now())
        #     text='RSI is lower than 30 for: '+symbol+ 'at "15min" timeframe at: '+str(datetime.datetime.now())
        #     base_url='https://api.telegram.org/bot1858939042:AAE_vy2jDANQGDk96hbck0B5RcMK-ngv6L8/sendMessage?chat_id=-1001376638249 &text="{}" '.format(text)
        #     requests.get(base_url,headers=headers)
        # if df.doji[498]!=0 and df.dc_Lband[499]<df.dc_Lband[498] and df.RSI[499]<30:
        #     print('DOJI is recognized for: ',symbol, 'at "15min" timeframe in min at: ',datetime.datetime.now())
        # print(symbol,': ',df.Aroon_up[499]-df.Aroon_down[499],'at: ',tf)
        
def getAll():
    tfs=['1m','3m','5m','15m','30m','1h','4h','1d']   
    for i in tfs:
        get_data(i)     
getAll()         
# plt.plot(df.close)   
# plt.plot(df.distFromEMA50)
# plt.plot(df.distFromEMA100)
# plt.plot(df.distFromEMA150)


                  
