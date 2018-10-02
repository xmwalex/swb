# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:59:07 2018

@author: ximing

1. Download kibot QQQ unadjusted 30 min data
2. Download QQQ daily from yahoo since 1999-3-10
3. add Date,Time,Open,High,Low,Close,Volume as heading to QQQ unadj 30 min data
4. change the file name in this file
5. Run this file
6. Run strategy_final_short 
7. see daily strategy and control. 
download QQQ daily from yahoo since 2018-1-2 as QQQ2018.csv
runfile('C:/stock/python/gitfolder/swb/strategy_final_dailyonly_fortest.py', wdir='C:/stock/python/gitfolder/swb')
8. see both long strategy and short strategy
 runfile('C:/stock/python/gitfolder/swb/strategy_final_testfuture.py', wdir='C:/stock/python/gitfolder/swb')
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
df = pd.read_csv('QQQunadj0830.txt')

profits = []
c,h,l,no=[],[],[],[]
da2 = []
df0=df.copy()


from datetime import datetime
def convertTime(df):
    datetimes=[]
    for i in tqdm(range(len(df.index))): 
        datetimes.append(datetime.strptime(df.loc[i,'Date']+' '+df.loc[i,'Time'],'%m/%d/%Y %H:%M'))
    df['Datetime']=datetimes
    dates=[]
    for i in tqdm(range(len(df.index))):
        dates.append(df.loc[i,'Datetime'].strftime('%Y-%m-%d'))
    df['Date'] = dates
    return df

df = convertTime(df)


df['dayOpen']=''
df['dayHigh']=''
df['dayLow']=''
df['dayClose']=''
o=0
h=0
c=0
backup_c=0
l=9999
for i in tqdm(range(len(df.index))): 
    if i>0 and df.loc[i,'Date']!=df.loc[i-1,'Date']:
        h=0
        l=9999
        o=0
        c=0
    if df.loc[i,'Time']=='9:30':
        if o!=0 or h!=0 or l!=9999 or c!=0:
            if backup_c==0:
                break
        o=df.loc[i,'Open']
        h = max(h,df.loc[i,'High'])
        l = min(l,df.loc[i,'Low'])
        backup_c=0
    elif df.loc[i,'Time']=='16:00':
        c = df.loc[i,'Open']
        if o==0 or h==0 or c==0 or l==9999:
            print('error!')
            break
        df.loc[i,'dayOpen']=o
        df.loc[i,'dayHigh']=h
        df.loc[i,'dayLow']=l
        df.loc[i,'dayClose']=c
        h=0
        l=9999
        o=0
        c=0
    elif df.loc[i,'Datetime'].hour<16 and df.loc[i,'Datetime'].hour>9:
        h = max(h,df.loc[i,'High'])
        l = min(l,df.loc[i,'Low'])
        if df.loc[i,'Time']=='13:00':
            backup_c = df.loc[i,'Open']
            df.loc[i,'dayOpen']=o
            df.loc[i,'dayHigh']=h
            df.loc[i,'dayLow']=l
            df.loc[i,'dayClose']=backup_c
        if o==0:
            o=df.loc[i,'Open']   

#tqdm.pandas()
dfdate = df[(df['Time']=='16:00')|(df['Time']=='13:00')][['Date','Time','dayOpen','dayHigh','dayLow','dayClose']]


def processUnit(x,df):
    a=df[(df['Date']==x[0]) & (df['Time']=='16:00')]
    if len(a)>0:
        x['dayOpen']=float(a['dayOpen'])
        x['dayHigh']=float(a['dayHigh'])
        x['dayLow']=float(a['dayLow'])
        x['dayClose']=float(a['dayClose'])
    else:
        b=df[(df['Date']==x[0]) & (df['Time']=='13:00')]
        if len(b)>0:
            x['dayOpen']=float(b['dayOpen'])
            x['dayHigh']=float(b['dayHigh'])
            x['dayLow']=float(b['dayLow'])
            x['dayClose']=float(b['dayClose'])
    return x

df3=df.apply(processUnit,args=(dfdate,),axis=1)


######################################################################

df=df3

# adjust QQQ price.
for i in tqdm(range(3846)):
    df.loc[i,'Open']=float(df.loc[i,'Open']/2)
    df.loc[i,'High']=float(df.loc[i,'High']/2)
    df.loc[i,'Low']=float(df.loc[i,'Low']/2)
    df.loc[i,'Close']=float(df.loc[i,'Close']/2)
    df.loc[i,'Volume']=float(df.loc[i,'Volume']*2)    
    df.loc[i,'dayOpen']=float(df.loc[i,'dayOpen']/2)
    df.loc[i,'dayHigh']=float(df.loc[i,'dayHigh']/2)
    df.loc[i,'dayLow']=float(df.loc[i,'dayLow']/2)
    df.loc[i,'dayClose']=float(df.loc[i,'dayClose']/2)   
#

def calcma50u(df):
    for i in tqdm(range(50,len(df))):
        df.loc[i,'ma50']=df.loc[i,'Close']/np.mean(df.loc[i-50:i,'Close'])
    return df

def calcma200u(df):
    for i in tqdm(range(200,len(df))):
        df.loc[i,'ma200']=df.loc[i,'Close']/np.mean(df.loc[i-200:i,'Close'])
    return df


def calcma20u(df):
    for i in tqdm(range(20,len(df))):
        df.loc[i,'ma20']=df.loc[i,'Close']/np.mean(df.loc[i-20:i,'Close'])
    return df

def calcma10u(df):
    for i in tqdm(range(10,len(df))):
        df.loc[i,'ma10']=df.loc[i,'Close']/np.mean(df.loc[i-10:i,'Close'])
    return df

def calcma50(df):
    for i in range(50,len(df)):
        df.loc[i,'ma50']=df.loc[i,'Adj Close']/np.mean(df.loc[i-50:i,'Adj Close'])
    return df

def calcma200(df):
    for i in range(200,len(df)):
        df.loc[i,'ma200']=df.loc[i,'Adj Close']/np.mean(df.loc[i-200:i,'Adj Close'])
    return df


def calcma20(df):
    for i in range(20,len(df)):
        df.loc[i,'ma20']=df.loc[i,'Adj Close']/np.mean(df.loc[i-20:i,'Adj Close'])
    return df

def calcma10(df):
    for i in range(10,len(df)):
        df.loc[i,'ma10']=df.loc[i,'Adj Close']/np.mean(df.loc[i-10:i,'Adj Close'])
    return df


def calcfeatures(df):
    df = calcma20(df)
    df = calcma200(df)
    df = calcma10(df)
    return df

qqq2=pd.read_csv('QQQall0830.csv')
qqq2=calcfeatures(qqq2)
qqq2=qqq2[['ma10','ma20','ma200','Date']]
df = pd.merge(df,qqq2,how='inner',left_on='Date',right_on='Date')

divdf = pd.read_csv('qqqdiv.csv')
divdf['Datetime']=pd.to_datetime(divdf.Date)
for i in tqdm(range(len(divdf.index))):
    divdf.loc[i,'Date'] = divdf.loc[i,'Datetime'].strftime('%Y-%m-%d')
divdf.drop('Datetime',axis=1,inplace=True)
df = pd.merge(df,divdf,how='left',on='Date')
            
def binExtended(dfintra):
    extHigh,extLow=0,999999
    for i in tqdm(range(len(dfintra))):
        if (dfintra.loc[i,'Datetime'].hour>=10 and dfintra.loc[i,'Datetime'].hour<=15):
            if extHigh!=0:
                dfintra.loc[i,'preHigh']=extHigh
                dfintra.loc[i,'preLow']=extLow
                extHigh=0
                extLow=999999
            else:
                continue
        elif dfintra.loc[i,'Datetime'].hour==9 and dfintra.loc[i,'Datetime'].minute==30:
            dfintra.loc[i,'preHigh']=extHigh
            dfintra.loc[i,'preLow']=extLow
            extHigh=0
            extLow=999999
        else:
            extHigh = max(extHigh,dfintra.loc[i,'High'])
            extLow = min(extLow,dfintra.loc[i,'Low'])
    return dfintra
            

df2 = binExtended(df)
dfdayext2 = df2[~np.isnan(df2.preHigh)]
dfdayext2.reset_index(inplace=True,drop=True)

cols=['Date', 'Time', 'Volume', 'Datetime',
       'dayOpen', 'dayHigh', 'dayLow', 'dayClose', 'ma10', 'ma20', 'ma200',
       'preHigh', 'preLow', 'div']
dfdayext3 = dfdayext2[cols]
dfdayext3.to_csv('dfdayextdiv4.csv',index=False)


df.drop(['ma10','ma20','ma200'],axis=1,inplace=True)
dfintra=df
dfintradict=dict()
dates = set(dfintra.Date)
timesets = {'11:30','13:00','14:30','10:30','12:30','9:30',
            '12:00','15:30','11:00','14:00','15:00','10:00','13:30'}
for d in tqdm(dates):
    tmp = dfintra[(dfintra['Date']==d)&(dfintra['Time'].apply(lambda x: x in timesets))]
    tmp.reset_index(inplace=True,drop=True)
    dfintradict[d]=tmp
    
np.save('dfintradict2.npy',dfintradict)



