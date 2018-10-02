# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:59:07 2018

@author: ximing
"""

import numpy as np
from scipy.stats import gmean
import pandas as pd
from tqdm import tqdm
df = pd.read_csv('QQQintra_unadj.csv')

profits = []
c,h,l,no=[],[],[],[]
da2 = []
df0=df.copy()
#df=df0
#df['Timestamp']=''
#df['Datetime']=''

%load_ext Cython

%%cython
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

%timeit
df = convertTime(df)

#df.apply(lambda x: x['Date']=x['Datetime'].strftime('%Y-%m-%d'),axis=1)

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
#        h = max(h,df.loc[i,'High'])
#        l = min(l,df.loc[i,'Low'])
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

df.to_csv('qqqv3.csv',index=False)
df = pd.read_csv('qqqv3.csv')
#from joblib import Parallel, delayed
#import multiprocessing    
## what are your inputs, and what operation do you want to 
## perform on each input. For example...
#
#num_cores = multiprocessing.cpu_count()
#    
#results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
#
#inputs = range(len(df.index))
#
#for i in tqdm(inputs):
#    processInput(i,df)
#df2=df.copy()
tqdm.pandas()
dfdate = df[(df['Time']=='16:00')|(df['Time']=='13:00')][['Date','Time','dayOpen','dayHigh','dayLow','dayClose']]
#dfdate = dfdate['Date','Time']

#def processUnit(x,df):
#    a=df[(df['Date']==x[0]) & (df['Time']=='16:00')]
#    if len(a)>0:
#        x['dayOpen']=float(a['dayOpen'])
#        x['dayHigh']=float(a['dayHigh'])
#        x['dayLow']=float(a['dayLow'])
#        x['dayClose']=float(a['dayClose'])
#    else:
#        b=df[(df['Date']==x[0]) & (df['Time']=='13:00')]
#        if len(b)>0:
#            x['dayOpen']=float(b['dayOpen'])
#            x['dayHigh']=float(b['dayHigh'])
#            x['dayLow']=float(b['dayLow'])
#            x['dayClose']=float(b['dayClose'])
#    return x


%%cython
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

df3=df.progress_apply(processUnit,args=(dfdate,),axis=1)

#error=[]
##def processInput(i,df):
#for i in tqdm(range(len(df.index))):
#    a=df[(df['Date']==df.loc[i,'Date']) & (df['Time']=='16:00')]
#    if len(a)>0:
#        df.loc[i,'dayOpen']=float(a['dayOpen'])
#        df.loc[i,'dayHigh']=float(a['dayHigh'])
#        df.loc[i,'dayLow']=float(a['dayLow'])
#        df.loc[i,'dayClose']=float(a['dayClose'])
#    else:
#        b=df[(df['Date']==df.loc[i,'Date']) & (df['Time']=='13:00')]
#        if len(b)>0:
#            df.loc[i,'dayOpen']=float(b['dayOpen'])
#            df.loc[i,'dayHigh']=float(b['dayHigh'])
#            df.loc[i,'dayLow']=float(b['dayLow'])
#            df.loc[i,'dayClose']=float(b['dayClose'])
#        else:
#            error.append(i)

######################################################################

#df3.to_csv('QQQintra_processed.csv',index=False)
df=df3
df = pd.read_csv('QQQintra_processed.csv')
df['Datetime']=pd.to_datetime(df.Datetime)

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

qqq2=pd.read_csv('QQQall.csv')
qqq2=calcfeatures(qqq2)
qqq2=qqq2[['ma10','ma20','ma200','Date']]
df = pd.merge(df,qqq2,how='inner',left_on='Date',right_on='Date')

#df=df_all
#df2 = df[['Date','dayOpen','dayHigh','dayLow','dayClose']]
#df2.drop_duplicates(inplace=True)
#df2.reset_index(inplace=True,drop=True)
#df2.dropna(inplace=True)
#df=df_all


#df2.columns=['Date','Open','High','Low','Close']
#
#for i in tqdm(range(50,len(df2))):
#    df2.loc[i,5]=df2.iloc[i,4]/np.mean(df2.iloc[i-50:i,4])


#df.to_csv('QQQintra_v4.csv',index=False)
#df = pd.read_csv('QQQintra_v4.csv')

#df.to_csv('QQQintra_v5.csv',index=False)
#df = pd.read_csv('QQQintra_v5.csv')
#df['Datetime']=pd.to_datetime(df.Datetime)
#from datetime import datetime
#from tqdm import tqdm
#for i in tqdm(range(len(df))):
#    df.loc[i,'Datetime']=datetime.strptime(df.loc[i,'Datetime'],'%Y-%m-%d %H:%M:%S')

#df = pd.read_csv('kibotQQQdailyAdjsplit.csv')

divdf = pd.read_csv('qqqdiv.csv')
divdf['Datetime']=pd.to_datetime(divdf.Date)
for i in tqdm(range(len(divdf.index))):
    divdf.loc[i,'Date'] = divdf.loc[i,'Datetime'].strftime('%Y-%m-%d')
divdf.drop('Datetime',axis=1,inplace=True)
df = pd.merge(df,divdf,how='left',on='Date')


#def testintra(df,th):
#    pin=-1
#    bar=-1
#    profits=[]
#    i=0
#    while i<len(df.index)-3:
#        i+=1
#        if df.loc[i,'Datetime'].hour>=16 or df.loc[i,'Datetime'].hour<9 or df.loc[i,'Time']=='9:00' or df.loc[i,'Datetime'].minute==00:
#            continue
#        if df.loc[i,'Time']=='9:30' and pin!=-1:
#            bar = df.loc[i,'Open']
#        if pin==-1 and df.loc[i,'Datetime'].hour!=9:
#            pin = df.loc[i,'Open']
#            bar = pin
#    #    if df.loc[i,'Datetime'].hour==9 and pin!=-1:
#    #        if df.loc[i,'Open']>1.0014*bar:
#    #            profits.append((df.loc[i,'Open']/pin)*3+1)
#    #            pin=-1
#    #            bar=-1
#    #            continue
#        if df.loc[i,'High']>th*bar and pin!=-1:
#            profits.append((th*bar/pin-1)*3+1)
#            pin=-1
#            bar=-1
#        if df.loc[i,'Datetime'].hour==15:
#            if pin==-1:
#                pin=df.loc[i,'Close']
#                bar=pin
#            else:
#                bar = df.loc[i,'Close']
#    return profits

#p=[]
#para=[]
#for th in tqdm(np.arange(1,1.01,0.001)):
#    p.append(np.prod(testintra(df,th)))
#    para.append(th)
##datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
#        
#p=[]
#para=[]
#for th in tqdm(np.arange(1.001,1.03,0.001)):
#    p.append(np.prod(sellextenddaily(dfdayext2,th)))
#    para.append(th)
    
#    
#def sellextended(df,th):
#    profits=[]
#    times=[]
#    pin=-1
#    profit=-1
#    for i in range(len(df)):
#    #        if i>100:
#    #            break
#        if df.loc[i,'Time']=='16:00':
#            if pin!=-1:
#                print(i)
#            pin = df.loc[i,'Open']
#            profit=-1
#        if pin!=-1 and ((df.loc[i,'Datetime'].hour>=16) or (df.loc[i,'Datetime'].hour<=9 and df.loc[i,'Datetime'].minute<30)):
#            if df.loc[i,'High']>th*pin:
#                profit = th*3-2
#                pin=-1
#                t=df.loc[i,'Datetime']
#        if ((df.loc[i,'Datetime'].hour in {10,11}) or (df.loc[i,'Datetime'].hour==9 and df.loc[i,'Datetime'].minute==30)) and pin!=-1:
#            profit = min((df.loc[i,'Open']/pin-1)*3+1,th*3-2)
#            pin=-1
#            t=df.loc[i,'Datetime']
#    #        print(str(pin)+' '+str(i))
#    #    if profit>1.024:
#    #        break
#        if profit!=-1:
#            profits.append(profit-1e-4)
#            times.append(t)
#            profit=-1
#    return profits
#
##for i in range(len(times)):
##    print(' '.join([str(profits[i]),str(times[i])]))
#            
#def sellextendbase(df):
#    profits=[]
#    pin=-1
#    for i in range(len(df)):
#        if df.loc[i,'Datetime'].hour==16 and df.loc[i,'Datetime'].minute==00:
#            if pin!=-1:
#                print(i)
#            pin = df.loc[i,'Open']
#        if ((df.loc[i,'Datetime'].hour in {10,11}) or (df.loc[i,'Datetime'].hour==9 and df.loc[i,'Datetime'].minute==30)) and pin!=-1:
#            profits.append((df.loc[i,'Open']/pin-1)*3+1)
#            pin=-1
#    return profits
#
#def sellextbasedaily(df):
#    profits = []
#    for i in range(1,len(df)):
#        profits.append((df.loc[i,'dayOpen']/df.loc[i-1,'dayClose']-1)*3+1)
#    return profits
#
#def sellextenddaily(df,th):
#    profits = []
#    for i in range(1,len(df)):
#        if df.loc[i,'preHigh']>th*df.loc[i-1,'dayClose']:
#            profits.append(th*3-2-1e-4)
#        else:
#            if (df.loc[i,'dayOpen']/df.loc[i-1,'dayClose'])>th:
#                profits.append(th*3-2-1e-4)                
#            else:
#                profits.append((df.loc[i,'dayOpen']/df.loc[i-1,'dayClose']-1)*3+1-1e-4)
#    return profits
            
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
            
#df3=df.loc[9990:]
#df3.reset_index(inplace=True,drop=True)
#df4 = binExtended(df3)            
#dfdayext = df4[~np.isnan(df4.preHigh)]
#dfdayext.reset_index(inplace=True,drop=True)

df2 = binExtended(df)
dfdayext2 = df2[~np.isnan(df2.preHigh)]
dfdayext2.reset_index(inplace=True,drop=True)
#divdf = pd.read_csv('qqqdiv.csv')
#divdf['Datetime']=pd.to_datetime(divdf.Date)
#
#for i in tqdm(range(len(divdf.index))):
#    divdf.loc[i,'Date'] = divdf.loc[i,'Datetime'].strftime('%Y-%m-%d')
#divdf.drop('Datetime',axis=1,inplace=True)
#dfdayext3 = pd.merge(dfdayext2,divdf,how='left',on='Date')

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
#    print(d)
    tmp = dfintra[(dfintra['Date']==d)&(dfintra['Time'].apply(lambda x: x in timesets))]
    tmp.reset_index(inplace=True,drop=True)
    dfintradict[d]=tmp
    
np.save('dfintradict2.npy',dfintradict)



