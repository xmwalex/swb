# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:45:09 2018

@author: ximing
"""
# dfdayext3 2016-4-14 day open should be 110.9. original intraday data is wrong

import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import datetime

pretradingdate = pd.Timestamp(2017,2,9)
 
def triplessdivextreturn(df,dfintradict):
    t1 = 1.008
    t2 = .985
    t3 = 1.0117
    t4 = .982
    t5=1.0004
    t6 = 1.0005
    t7 = 1.001
    t8 = 1.08
    t9 = 1.005
    t10 = .87
    th=1.011    
    t13 = .999
    profit = 1
    profits=[]
    profits2=[]
    monthlyprof = [1]   
    prevmon = '03'
    premax=df.loc[0,'Close']
    for i in range(2,len(df)):     
        profit=1
        Open = df.loc[i,'Open']
        High = df.loc[i,'High']
        Low = df.loc[i,'Low']
        Close = df.loc[i,'Close']
        if df.loc[i-1,'Close']>t7*df.loc[i-2,'Close'] and df.loc[i-2,'ma20']>t8:#t7 1.001, t8 1.08
            profit = (1-df.loc[i,'Close']/df.loc[i-1,'Close'])*3+1#short
        elif i>100 and df.loc[i-1,'Close']>premax:
            if Open/df.loc[i-1,'Close']>.998:
                profit = ((Open/df.loc[i-1,'Close']-1)*3+1-1e-4)*((1-Close/Open)*3+1-1e-4)
            else:
                profit = (Close/df.loc[i-1,'Close']-1)*3+1-1e-4
        elif(df.loc[i-1,'Close']/df.loc[i-1,'Open']>t9 and df.loc[i-1,'ma20']>t10):
            profit = (Open/df.loc[i-1,'Close']-1)*3+1#buy
            profits.append(profit-1e-4)
            profit = (1-df.loc[i,'Close']/Open)*3+1#short
        elif Open/df.loc[i-1,'Close']>t1 or Open/df.loc[i-1,'Close']<t4:
            profit = ((Close/df.loc[i-1,'Close'])-1)*3+1
        elif Open/df.loc[i-1,'Close']<t6:
            if High/Open>t3:
                profit = (t3*Open/df.loc[i-1,'Close']-1)*3+1                
            else:
                profit = (Close/df.loc[i-1,'Close']-1)*3+1
        elif High/Open>t5:
            profit = (t5*Open/df.loc[i-1,'Close']-1)*3+1
        else:
            profit = (df.loc[i,'Close']/df.loc[i-1,'Close']-1)*3+1
        premax = max(df.loc[i-1,'Close'],premax) 
        if profit!=1:
            profits.append(profit-1e-4)
#        if ~np.isnan(df.loc[i,'div']):
#            profits.append(df.loc[i,'div']/df.loc[i,'Close']+1)
#        if (df.loc[i,'Date']).split('-')[1]!=prevmon:
#            monthlyprof.append(np.prod(profits))
#            prevmon = (df.loc[i,'Date']).split('-')[1]
#    monthlyprof.append(np.prod(profits))
#    for i in range(len(monthlyprof)-1,0,-1):
#        monthlyprof[i]=monthlyprof[i]/monthlyprof[i-1]
    return profits

def triplessdivextreturn_4(df,dfintradict):
    t2 = .985
    t3 = 1.011
    t4 = .98
    t5=1.0004
    t6 = .999
    t7 = 1.001
    t8 = 1.08
    th=1.011
    profit = 1
    profits=[]
    profits2=[]
    monthlyprof = [1]   
    prevmon = '03'
    premax=df.loc[0,'Close']
    for i in range(2,len(df)):     
        profit=1
        dayOpen = df.loc[i,'Open']
        dayHigh = df.loc[i,'High']
        dayLow = df.loc[i,'Low']
        dayClose = df.loc[i,'Close']

        if df.loc[i-1,'Close']>t7*df.loc[i-2,'Close'] and df.loc[i-2,'ma20']>t8:
            profit = (1-df.loc[i,'Close']/df.loc[i-1,'Close'])*3+1#short 61% win, gmean 2.28%
            
        elif dayOpen/df.loc[i-1,'Close']<t4:
#            print('wrong'+str(i))
            profit = ((dayClose/df.loc[i-1,'Close'])-1)*3+1-1e-4

        elif dayOpen/df.loc[i-1,'Close']<t6:
            profit = (df.loc[i,'Close']/df.loc[i-1,'Close']-1)*3+1-1e-4

        elif dayHigh/dayOpen>t5:
            profit = (t5*dayOpen/df.loc[i-1,'Close']-1)*3+1-1e-4
        else:
            profit = (df.loc[i,'Close']/df.loc[i-1,'Close']-1)*3+1-1e-4
        premax = max(df.loc[i-1,'Close'],premax) 
        if profit!=1:
            profits.append(profit)
    return profits

def calcannualreturn(profits,df):
    return (np.prod(profits)**(1/len(df)))**252

def checkOutTime(df,p):
    for i in range(len(df)):
        if df.loc[i,'High']>p:
            return df.loc[i,'Time']
    return 'None'

def testssreturnext(df,dfintradict):
    profits = triplessdivextreturn(df,dfintradict)
    print("{:,}".format(np.prod(profits)))
    print(gmean(profits))
    print(calcannualreturn(profits,df))
    
def testssreturnext_4(df,dfintradict):
    profits = triplessdivextreturn_4(df,dfintradict)
    print("{:,}".format(np.prod(profits)))
    print(gmean(profits))
    print(calcannualreturn(profits,df))

dfdayext3 = pd.read_csv('dfdayext3div.csv')
dfdayext3['Datetime']=pd.to_datetime(dfdayext3.Date)
dfdayext3['Close'] = dfdayext3['dayClose']
dfdayext3['Open'] = dfdayext3['dayOpen']
dfdayext3['High'] = dfdayext3['dayHigh']
dfdayext3['Low'] = dfdayext3['dayLow']
QQQ = pd.read_csv('QQQ2018.csv')
dfintradict = np.load('dfintradict.npy').item()

dfdayl10 = dfdayext3.loc[2748:]
dfdayl10.reset_index(inplace=True,drop=True)
print('hist, stra 1')
testssreturnext(dfdayext3,dfintradict)
print('valid, stra 1')
testssreturnext(dfdayl10,dfintradict)




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


def triplereturn(df):
    profit = 1
    for i in range(1,len(df)):
        profit = profit * ((df.loc[i,'Adj Close']/df.loc[i-1,'Adj Close']-1)*3+1)
    return profit

QQQ = calcfeatures(QQQ)
print('This year,stra 1')
testssreturnext(QQQ,dfintradict)
#
#for i in range(0,len(dfdayext3),252):
#    dfdayl10 = dfdayext3.loc[i:i+250]
#    dfdayl10.reset_index(inplace=True,drop=True)
#    testssreturnext(dfdayl10,dfintradict)
#    print('\n')


# 从五月七日开始测试新策略
dfdayl102 = QQQ.loc[86:]
dfdayl102.reset_index(inplace=True,drop=True)
print('test set, stra 1')
testssreturnext(dfdayl102,dfintradict)

print('hist, stra 2')
testssreturnext_4(dfdayext3,dfintradict)
print('valid , stra 2')
testssreturnext_4(dfdayl10,dfintradict)
print('this year, stra 2')
testssreturnext_4(QQQ,dfintradict)
print('test set, stra 2')
testssreturnext_4(dfdayl102,dfintradict)

print('hold TQQQ baseline, YTD')
print(triplereturn(QQQ))

print('hold TQQQ baseline, test set')
print(triplereturn(dfdayl102))
#dfdayl10 = QQQ.loc[2300:2500]
#dfdayl10.reset_index(inplace=True,drop=True)
#print(triplereturn(dfdayl10))
#calcannualreturn(triplereturn(dfdayl10),dfdayl10)
#
#testssreturnext(dfdaytrain,dfintradict)
##
#testssreturnext(dfdaytest,dfintradict)

#
## best result on train
#t1 =1.0229
#t2 = .985
#t3 = 1.0118
#t4 = .978
#t5=1.0004
#t6 = 1.0004
#th = 1.011
#
## best result on all
#t1 = 1.0229
#t2 = .985
#t3 = 1.0115
#t4 = .978
#t5=1.0004
#t6 = 1.0005
#th = 1.01
#
#p=[]
#para=[]
##for t1 in tqdm(np.arange(1.004,1.007,.001)):
#t1 = 1.001
#for t2 in tqdm(np.arange(1.005,1.015,0.001)):
#    p.append(np.prod(triplessdivextreturn(dfdayext3,dfintradict,t2)))
#    para.append([t1,t2])
#
##
#p=[]
#para=[]
#for t1 in tqdm(np.arange(.98,1.0,.001)):
#        p.append(np.prod(triplessdivextreturn(dfdayl10,dfintradict,t1)))
#        para.append(t1)

#p=[]
#para = ['10:00','10:30','11:00','11:30','12:00','12:30','13:00','13:30','14:00','14:30','15:00']
#for i in tqdm(range(len(para))):
#    t = para[i]
#    p.append(np.prod(triplessdivextreturn(dfdayext3,dfintradict,t)))
