# -*- coding: utf-8 -*-
"""
Created on Sun May  6 10:28:13 2018

@author: ximing
"""

import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
nasdaq = pd.read_csv('nasdaq.csv')
#
#profit = 1
#for i in range(1,len(nasdaq)):
#    profit = profit * nasdaq.loc[i,'Open']/nasdaq.loc[i-1,'Close']
    
    
qqq1y = pd.read_csv('QQQ1y.csv')
qqq=qqq1y
tqqq = pd.read_csv('tqqq.csv')
tqqq1y = pd.read_csv('tqqq1y.csv')
tqqq5y = pd.read_csv('tqqq5y.csv')
qqq5y = pd.read_csv('qqq5y.csv')
qqqmt = pd.read_csv('qqqmatcht.csv')

qqqbear1 = pd.read_csv('qqqbear1.csv')
qqqbear2 = pd.read_csv('qqqbear2.csv')

#
#profit = 1
#for i in range(1,len(qqq)):
#    profit = profit * qqq.loc[i,'Open']/qqq.loc[i-1,'Close']

df=tqqq
def triplecoreturn(df):
    profit =1
    prevpro=1
    profits = [1]
    gap = df.loc[0,'Close']-df.loc[0,'Adj Close']
    for i in range(1,len(df)):
        diff = df.loc[i,'Close']-df.loc[i,'Adj Close']
        div = gap-diff
        if i+5<len(df):
            diff2 = df.loc[i+5,'Close']-df.loc[i-1,'Adj Close']
            if abs(div/df.loc[i,'Close'])<1.4e-3 or (abs(gap-diff2)/df.loc[i,'Close'])<1.4e-3:
                div=0
        if div!=0:
            gap = diff
#        if(abs(div)>1):
#            print(i)
#            print(div)
        if i%250==0:
            profits.append(profit/prevpro)
            prevpro=profit
        profit = profit*((((df.loc[i,'Open']+div)/df.loc[i-1,'Close'])-1)*3+1-3e-5)
        if ((((df.loc[i,'Open']+div)/df.loc[i-1,'Close'])-1)*3+1-3e-5)>1.05:
            print(i)
            print(div)
    profits.append(profit/prevpro)
    return profits


def tripleconodivreturn(df):
    profit = 1
    profits=[]
    for i in range(1,len(df)):
        profit = (((df.loc[i,'Open'])/df.loc[i-1,'Close'])-1)*3+1+8e-5
        if ((((df.loc[i,'Open'])/df.loc[i-1,'Close'])-1)*3+1+8e-5)>1.05:
            print(i)
            print(profit/prevpro)
        profits.append(profit)
    return profits

def triplecoreturn(df):
    profit =1
    gap = df.loc[0,'Close']-df.loc[0,'Adj Close']
    for i in range(1,len(df)):
        diff = df.loc[i,'Close']-df.loc[i,'Adj Close']
        div = gap-diff
        if abs(div/df.loc[i,'Close'])<1.4e-3:
            div=0
        gap = diff
        if(abs(div)>1):
            print(i)
            print(div)
        profit = profit*((((df.loc[i,'Open']+div)/df.loc[i-1,'Close'])-1)*3+1)
    return profit


def totalcoreturn(df):
    profit =1 
    gap = df.loc[0,'Close']-df.loc[0,'Adj Close']
    for i in range(1,len(df)):
        diff = df.loc[i,'Close']-df.loc[i,'Adj Close']
        div = gap-diff
        gap = diff
        if(abs(div)>1e-3):
            print(i)
            print(div)
        profit = profit*((df.loc[i,'Open']+div)/df.loc[i-1,'Close'])
    return profit

def nodivcoreturn(df):
    profit = 1
    for i in range(1,len(df)):
        profit = profit * df.loc[i,'Open']/df.loc[i-1,'Close']
    return profit

def triplereturn(df):
    profit = 1
    for i in range(1,len(df)):
        profit = profit * ((df.loc[i,'Adj Close']/df.loc[i-1,'Adj Close']-1)*3+1)
    return profit


def holdreturn(df):
    adjcl = list(df['Adj Close'])
    return adjcl[-1]/adjcl[0]

def leverageReturn(df,lev,marginrate):
    adjcl = list(df['Adj Close'])
    moneyin = adjcl[0]*1
    share = lev
    marginrate = marginrate**(1/250)-1
    return ((share*adjcl[-1]/moneyin)**(1/len(df))-marginrate*(lev-1))**(len(df))

#
#print(totalreturn(qqq))
#
#print(totalreturn(qqqall))
#holdreturn(qqq)
#holdreturn(qqqall)    
#totalreturn(spyall)
#print(nodivreturn(spyall))
#print(nodivreturn(qqqall))
#print(nodivreturn(tqqq))


import numpy as np
import pandas as pd
qqqall = pd.read_csv('QQQall.csv')

profit = 1
for i in range(1,len(qqqall)):
    profit = profit * qqqall.loc[i,'Open']/qqqall.loc[i-1,'Close']
    
spyall = pd.read_csv('SPYall.csv')

profit = 1
for i in range(1,len(spyall)):
    profit = profit * spyall.loc[i,'Open']/spyall.loc[i-1,'Close']

getprofit = 1.002
#cutprofit = 
profit=1
    
    

def testdist(df):
    profits=[]
    for i in range(2,len(df)):
        if df.loc[i-1,'dayClose']>.992*df.loc[i-1,'dayOpen']:
            profits.append(i)
#            profits.append((df.loc[i,'dayClose']/df.loc[i,'dayOpen']-1)*3+1)
#    for i in range(len(df)):
#        dftmp = dfintradict[df.loc[i,'Date']]
#        try:
#            if dftmp[dftmp['Time']==t].Close.iloc[0]<t2*dftmp[dftmp['Time']==t].dayOpen.iloc[0]:
#                profits.append((1-df.loc[i,'dayClose']/dftmp[dftmp['Time']==t].Close.iloc[0])*3+1)
#        except:
#            print(i)
    return profits

from tqdm import tqdm
p=[]
para=[]
for th in tqdm(np.arange(.9,1,.01)):
    profits = triplessdivextreturn(dfdayext3,th)
    p.append(np.prod(profits))
    para.append(th)

# 9:30, .998
p=[]
para=[]
for t2 in tqdm(np.arange(.98,.999,0.001)):
    for th in {'12:00','12:30','13:00','13:30','14:00','14:30','15:00'}:
        profits = testdist(dfdayext3,th,t2)
        p.append(np.prod(profits))
        para.append([th,t2])


p=[]
para = []
for th1 in tqdm(np.arange(.998,1.00,.0001)):
    for th2 in tqdm(np.arange(1.01,1.013,0.0001)):
        for th3 in np.arange(1.0003,1.0007,0.0001):        
            profits = triplessnodivreturn(df_all,th1,th2,th3)
            p.append(np.prod(profits))
#            print(len(p))
            para.append([th1,th2,th3])

def triplessnodivreturn(df):
    profit = 1
    profits=[]
    profits2=[]
    monthlyprof = [1]   
    prevmon = '03'
    prevdiff = round(df.loc[0,'Adj Close']-df.loc[0,'Close'],2)
    for i in range(3,len(df)):     
        profit=1
        if df.loc[i-1,'Close']>1.01*df.loc[i-2,'Close'] and df.loc[i-2,'ma20']>1.08:
            profit = (1-df.loc[i,'Open']/df.loc[i-1,'Close'])*3+1#short
        elif(df.loc[i-1,'Close']/df.loc[i-1,'Open']>1.005 and df.loc[i-1,'ma10']>.88):
            profit = (df.loc[i,'Open']/df.loc[i-1,'Close']-1)*3+1#buy
            profits.append(profit-1e-4)
            if df.loc[i,'Low']<.925*df.loc[i,'Open']:
                profit = (1-.925)*3+1
            else:
                profit = (1-df.loc[i,'Close']/df.loc[i,'Open'])*3+1#short
        elif df.loc[i,'Open']/df.loc[i-1,'Close']>1.006:
                profit = (((df.loc[i,'Close'])/df.loc[i-1,'Close'])-1)*3+1
        elif df.loc[i,'Open']/df.loc[i-1,'Close']<.989:
                profit = (((df.loc[i,'Close'])/df.loc[i-1,'Close'])-1)*3+1
        elif df.loc[i,'Open']/df.loc[i-1,'Close']<.9994:
            if df.loc[i,'High']/df.loc[i,'Open']>1.0118:
                profit = (1.012*df.loc[i,'Open']/df.loc[i-1,'Close']-1)*3+1
            else:
                profit = (df.loc[i,'Close']/df.loc[i-1,'Close']-1)*3+1
        elif df.loc[i,'High']/df.loc[i,'Open']>1.0004:
                profit = (1.0004*df.loc[i,'Open']/df.loc[i-1,'Close']-1)*3+1
        else:
                profit = (df.loc[i,'Close']/df.loc[i-1,'Close']-1)*3+1

        if profit!=1:
            profits.append(profit-1e-4)
        if ~np.isnan(df_all.loc[i,'div']):
#            print(df_all.loc[i,'div'])
            profits.append(df_all.loc[i,'div']/df_all.loc[i,'Close']+1)
#        cdiff = round(df.loc[i,'Adj Close']-df.loc[i,'Close'],2)
#        if cdiff!=prevdiff:
#            profits.append(abs(cdiff-prevdiff)/df.loc[i,'Close']+1)
#        if (qqqall.loc[i,'Date']).split('-')[1]!=prevmon:
#            monthlyprof.append(np.prod(profits))
#            prevmon = (qqqall.loc[i,'Date']).split('-')[1]
    monthlyprof.append(np.prod(profits))
    return profits

#monthlyprof=[]
#for i in range(1,len(profits)):
#    monthlyprof.append(profits[i]/profits[i-1])
p=[]
para=[]
for th1 in tqdm(np.arange(.9,1.1,.01)):
    for th2 in np.arange(.9,1.0,.01):
        p.append(np.prod(testdist2(df_all,th1,th2)))
        para.append([th1,th2])
    
def testdist2(df,th1,th2):
    profit = 1
    profits=[]
    for i in range(2,len(df)):
#        if df.loc[i-1,'Close']>df.loc[i-2,'Close'] and df.loc[i-2,'ma20']>1.05 and df.loc[i-2,'ma20']<=1.1:
#            profit = (df.loc[i,'Close']/df.loc[i-1,'Close'])*3-2
#        if(df.loc[i-1,'Close']/df.loc[i-1,'Open']>1.0077 and df.loc[i-1,'ma10']>.95):
##            profit = (df.loc[i,'Open']/df.loc[i-1,'Close']-1)*3+1#buy
##            profits.append(profit-1e-4)
#            if df.loc[i,'Low']<th*df.loc[i,'Open']:
#                profit = (1-th)*3+1#short
#            else:
#                profit = (1-df.loc[i,'Close']/df.loc[i,'Open'])*3+1#short
        if df.loc[i-1,'Close']<th1*df.loc[i-2,'Close'] and df.loc[i-2,'ma20']<th2:
            profit = df.loc[i,'Close']/df.loc[i-1,'Close']
#            if df.loc[i,'Low']<th*df.loc[i,'Open']:
#                profit = (1-(df.loc[i,'Open']*th)/df.loc[i-1,'Close'])*3+1#short                
#            else:
#            profit = (1-df.loc[i,'Close']/df.loc[i-1,'Close'])*3+1#short
            profits.append(profit)
    return profits
# 实际测试显示，weekday白天做空没有用。 
#

def ssnodivreturn(df):
    profit = 1
    profits=[]
    for i in range(1,len(df)):
        if df.loc[i-1,'Close']/df.loc[i-1,'Open']>1.0225:
            profit = df.loc[i,'Open']/df.loc[i-1,'Close']+df.loc[i,'Open']/df.loc[i,'Close']-1
        elif df.loc[i,'Open']/df.loc[i-1,'Close']>1.06 or df.loc[i,'Open']/df.loc[i-1,'Close']<0.97:
            profit = (df.loc[i,'Close'])/df.loc[i-1,'Close']
        elif df.loc[i,'High']/df.loc[i,'Open']>1.003:
            profit = 1.003*df.loc[i,'Open']/df.loc[i-1,'Close']
        else:
#            profit = 1
            profit = df.loc[i,'Close']/df.loc[i-1,'Close']
        profits.append(profit)
    return profits        

#df2.columns=['Date','Open','High','Low','Close']
#
#for i in tqdm(range(50,len(df2))):
#    df2.loc[i,5]=df2.iloc[i,4]/np.mean(df2.iloc[i-50:i,4])


def calcma50a(df):
    for i in tqdm(range(50,len(df))):
        df.loc[i,'ma50']=df.loc[i,'dayClose']/np.mean(df.loc[i-50:i,'dayClose'])
    return df

def calcma200a(df):
    for i in tqdm(range(200,len(df))):
        df.loc[i,'ma200']=df.loc[i,'dayClose']/np.mean(df.loc[i-200:i,'dayClose'])
    return df


def calcma20a(df):
    for i in tqdm(range(20,len(df))):
        df.loc[i,'ma20']=df.loc[i,'dayClose']/np.mean(df.loc[i-20:i,'dayClose'])
    return df

def calcma10a(df):
    for i in tqdm(range(10,len(df))):
        df.loc[i,'ma10']=df.loc[i,'dayClose']/np.mean(df.loc[i-10:i,'dayClose'])
    return df



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


import datetime
def weekday(dtstr):
    dt = datetime.datetime.strptime(dtstr,'%Y-%m-%d')
    return dt.weekday()

def calcfeatures(df):
    df = calcma20(df)
    df = calcma200(df)
    df = calcma10(df)
    return df
#
##qqqwk = pd.read_csv('QQQallwk.csv')
#qqqall = calcfeatures(qqqall)
#qqqmt = calcfeatures(qqqmt)
#qqqbear1 = calcfeatures(qqqbear1)
#qqqbear2 = calcfeatures(qqqbear2)
#tqqq = calcfeatures(tqqq) 

def calcannualreturn(profits,df):
    return (np.prod(profits)**(1/len(df)))**252

def testssreturn(df):
    profits = triplessnodivreturn(df)
    print(np.prod(profits))
    print(gmean(profits))
    print(calcannualreturn(profits,df))
    
def adjust_dividend(df):
    for i in tqdm(range(len(df))):
        cdiff = round(df.loc[i,'Adj Close']-df.loc[i,'Close'],3)
        df.loc[i,'Open']=df.loc[i,'Open']+cdiff
        df.loc[i,'High']=df.loc[i,'High']+cdiff
        df.loc[i,'Low']=df.loc[i,'Low']+cdiff
        df.loc[i,'Close']=df.loc[i,'Close']+cdiff
    return df

kqqqdu = pd.read_csv('kibotQQQdailyUnadj.csv')
for i in range(1,len(kqqqdu)):
    if kqqqdu.loc[i,'Close']/kqqqdu.loc[i-1,'Close']<.75 or kqqqdu.loc[i,'Close']/kqqqdu.loc[i-1,'Close']>1.25:
        print(i)

for i in range(0,260):
    kqqqdu.loc[i,'Open']=kqqqdu.loc[i,'Open']/2
    kqqqdu.loc[i,'High']=kqqqdu.loc[i,'High']/2
    kqqqdu.loc[i,'Low']=kqqqdu.loc[i,'Low']/2
    kqqqdu.loc[i,'Close']=kqqqdu.loc[i,'Close']/2    
    kqqqdu.loc[i,'Volume']=kqqqdu.loc[i,'Volume']*2    

from math import floor
prevdif = round(kqqqdu2.loc[i,'Adj Close']/kqqqdu2.loc[i,'Close'],3)
    div=[]
    for i in range(0,len(kqqqdu2)):
        currdif = floor(kqqqdu2.loc[i,'Adj Close']/kqqqdu2.loc[i,'Close'],3)
    #    if currdif!=prevdif:
        div.append(currdif)
        prevdif=currdif

#kqqqdu.to_csv('kibotQQQdailyAdjsplit.csv')

kqqqdu2 = pd.merge(kqqqdu,kqqqda,on='Date',how='inner')
kqqqdu2.columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
kqqqdu2.to_csv('kibotQQQdailyAdjsplit.csv',index=False)
kqqqdu2=calcfeatures(kqqqdu2)
testssreturn(kqqqdu2)

qqqdiv=pd.read_csv('qqqdiv.csv')

p=[]
para=[]
for t1 in np.arange(1.003,1.009,0.001):
    for t2 in np.arange(.98,1,0.005):
        for t3 in np.arange(1.001,1.02,0.001):
            for t4 in np.arange(.985,.995,0.001):                
                for t5 in np.arange(1,1.01,0.001):
                    for t6 in np.arange(.999,1.0,0.0001):
                        para.append((t1,t2,t3,t4,t5,t6))

p=[]
para=[]
for t7 in tqdm(np.arange(.98,1,0.0001)):
    p.append(np.prod(triplessdivextreturn(dfdayext3,dfintradict,t7)))
    para.append(t7)


import multiprocessing
from itertools import product
with multiprocessing.Pool(processes=2) as pool:
    results = pool.map(ups,para)


p=[0]*len(para)
for pa in tqdm(range(len(para))):
    p[pa] = ups(dfdayext3,para[pa])
    
def ups(df,pas):
    [t1,t2,t3,t4,t5,t6]=pas
    profits = triplessdivextreturn(df,dfintradict,t1,t2,t3,t4,t5,t6)
    return np.prod(profits)

%%cython
import numpy as np
def triplessdivextreturn(df,dfintradict,t7):
    t5=1.0004
    t2=.985
    t3=1.0118
    t1=1.006
    t4=.989
    t6=.9994
    th=1.01
    profit = 1
    profits=[]
    profits2=[]
    monthlyprof = [1]   
    prevmon = '03'
    for i in range(3,len(df)):     
        profit=1
        if df.loc[i-1,'dayClose']>1.01*df.loc[i-2,'dayClose'] and df.loc[i-2,'ma20']>1.08:
            profit = (1-df.loc[i,'dayClose']/df.loc[i-1,'dayClose'])*3+1#short
        elif(df.loc[i-1,'dayClose']/df.loc[i-1,'dayOpen']>1.005 and df.loc[i-1,'ma10']>.88):
            if df.loc[i,'preHigh']>th*df.loc[i-1,'dayClose']:
                profit = 3*th-2
            else:
                profit = (df.loc[i,'dayOpen']/df.loc[i-1,'dayClose']-1)*3+1#buy
            profits.append(profit-1e-4)
            if df.loc[i,'dayLow']<.925*df.loc[i,'dayOpen']:#short
                profit = (1-.925)*3+1
            else:
                profit = (1-df.loc[i,'dayClose']/df.loc[i,'dayOpen'])*3+1#short
        elif df.loc[i,'preHigh']>th*df.loc[i-1,'dayClose']:
            profit = 3*th-2
            if df.loc[i,'dayLow']<t2*df.loc[i,'dayOpen']:
                profits.append(profit-1e-4)
                profit = (df.loc[i,'dayClose']/(t2*df.loc[i,'dayOpen'])-1)*3+1
#                profits2.append(profit)
        elif df.loc[i,'dayOpen']/df.loc[i-1,'dayClose']>t1:
            profit = (((df.loc[i,'dayClose'])/df.loc[i-1,'dayClose'])-1)*3+1
        elif df.loc[i,'dayOpen']/df.loc[i-1,'dayClose']<t4:
            profit = (((df.loc[i,'dayClose'])/df.loc[i-1,'dayClose'])-1)*3+1
        elif df.loc[i,'dayOpen']/df.loc[i-1,'dayClose']<t6:
            earlyOut=0
            if df.loc[i,'dayHigh']/df.loc[i,'dayOpen']>t3:
#                print(i)
                outTime = checkOutTime(dfintradict[df.loc[i,'Date']],t3*df.loc[i,'dayOpen'])
#                print(int(outTime.split(':')[0]))
                if int(outTime.split(':')[0])<13:
                    profit = (t3*df.loc[i,'dayOpen']/df.loc[i-1,'dayClose']-1)*3+1                
                    earlyOut=1
            if earlyOut==0:
                profit = (df.loc[i,'dayClose']/df.loc[i-1,'dayClose']-1)*3+1
        elif df.loc[i,'dayHigh']/df.loc[i,'dayOpen']>t5:
            profit = (t5*df.loc[i,'dayOpen']/df.loc[i-1,'dayClose']-1)*3+1
            outTime = checkOutTime(dfintradict[df.loc[i,'Date']],t5*df.loc[i,'dayOpen'])
            if int(outTime.split(':')[0])<13:
                p2 = checkLow2(dfintradict[df.loc[i,'Date']],t7)
                if p2>0:
                    profits2.append(p2)
        else:
            profit = (df.loc[i,'dayClose']/df.loc[i-1,'dayClose']-1)*3+1
        if profit!=1:
            profits.append(profit-1e-4)
        if ~np.isnan(df.loc[i,'div']):
#            print(df_all.loc[i,'div'])
            profits.append(df.loc[i,'div']/df.loc[i,'dayClose']+1)
#        cdiff = round(df.loc[i,'Adj Close']-df.loc[i,'Close'],2)
#        if cdiff!=prevdiff:
#            profits.append(abs(cdiff-prevdiff)/df.loc[i,'Close']+1)
        if (df.loc[i,'Date']).split('-')[1]!=prevmon:
            monthlyprof.append(np.prod(profits))
            prevmon = (df.loc[i,'Date']).split('-')[1]
    monthlyprof.append(np.prod(profits))
    for i in range(len(monthlyprof)-1,0,-1):
        monthlyprof[i]=monthlyprof[i]/monthlyprof[i-1]
    return profits2


def checkLow2(df,p):
    if '14:30' not in df['Time']:
        return -1
    if df.loc[df['Time']=='14:30','Open'].iloc[0]<p*df.loc[df['Time']=='14:00','Open'].iloc[0]:
        return df.loc[1,'dayClose']/list(df.loc[df['Time']=='14:30','Open'])[0]
    if df.loc[df['Time']=='15:00','Open'].iloc[0]<p*df.loc[df['Time']=='14:30','Open'].iloc[0]:
        return df.loc[1,'dayClose']/list(df.loc[df['Time']=='15:00','Open'])[0]
    if df.loc[df['Time']=='15:30','Open'].iloc[0]<p*df.loc[df['Time']=='15:00','Open'].iloc[0]:
        return df.loc[1,'dayClose']/list(df.loc[df['Time']=='15:30','Open'])[0]
    return -1
def checkLow(df,p):
    for i in range(len(df)):
        pbar=0
        if df.loc[i,'Time']=='14:00':
            pbar = df.loc[i,'Open']*p
        if pbar!=0 and df.loc[i,'Low']<pbar:
            return df.loc[i,'dayClose']/pbar
    return -1


def checkOutTime(df,p):
    for i in range(len(df)):
        if df.loc[i,'High']>p:
            return df.loc[i,'Time']
    return 'None'

def testssreturnext(df,dfintradict):
    profits = triplessdivextreturn(df,dfintradict)
    print(np.prod(profits))
    print(gmean(profits))
    print(calcannualreturn(profits,df))
    
dset = set()
for i in range(len(dfdayext3)):
    if dfdayext3.loc[i,'Date'] in dset:
        break
    else:
        dset.add(dfdayext3.loc[i,'Date'])

#dfdayext3.to_csv('dfdayext3div.csv',index=False)
dfdayext3 = pd.read_csv('dfdayextdiv.csv')
dfdayext3['Datetime']=pd.to_datetime(dfdayext3.Date)

dfintradict = np.load('dfintradict.npy').item()

for i in range(len(dfdayext3)):
    if dfdayext3.loc[i,'preHigh']<dfdayext3.loc[i,'dayOpen'] or dfdayext3.loc[i,'preLow']>dfdayext3.loc[i,'dayOpen']:
        print(i)

profits=[]
weights=[1]
for i in np.arange(0,len(qqqall)-250,21):
    df = qqqall.loc[i:]
    df.reset_index(drop=True,inplace=True)
#    profits.append(triplereturn(df)*weights[-1])
    weights.append(weights[-1]*1.0025)
    profits.append(calcannualreturn(triplereturn(df),df))