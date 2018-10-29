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

def triplessdivextreturnshort(df,dfintradict):
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
    premax=df.loc[0,'dayClose']
    for i in range(2,len(df)):     
        profit=1
        dayOpen = df.loc[i,'dayOpen']
        dayHigh = df.loc[i,'dayHigh']
        dayLow = df.loc[i,'dayLow']
        dayClose = df.loc[i,'dayClose']
        preHigh = df.loc[i,'preHigh']
        if df.loc[i-1,'dayClose']>t7*df.loc[i-2,'dayClose'] and df.loc[i-2,'ma20']>t8:
            profit = (1-df.loc[i,'dayClose']/df.loc[i-1,'dayClose'])*3+1#short 61% win, gmean 2.28%
        elif (preHigh>th*df.loc[i-1,'dayClose'] or dayOpen>th*df.loc[i-1,'dayClose']) and (preHigh<1.15*df.loc[i-1,'dayClose'] or preHigh<1.07*dayOpen):
            profit = 3*th-2-1e-4#buy
            if dayLow<t2*dayOpen:
#                profits.append(profit-1e-4)
                profit2 = (dayClose/(t2*dayOpen)-1)*3+1-1e-4
                profit = profit*profit2
        elif dayOpen/df.loc[i-1,'dayClose']<t4:
#            print('wrong'+str(i))
            profit = ((dayClose/df.loc[i-1,'dayClose'])-1)*3+1-1e-4

        elif dayOpen/df.loc[i-1,'dayClose']<t6:
            earlyOut=0
            profit=1
            if dayHigh/dayOpen>t3:
                outTime = checkOutTime(dfintradict[df.loc[i,'Date']],t3*dayOpen)
                if int(outTime.split(':')[0])<13:
                    profit = (t3*dayOpen/df.loc[i-1,'dayClose']-1)*3+1-1e-4                
                    earlyOut=1
            if earlyOut==0:
                profit = (df.loc[i,'dayClose']/df.loc[i-1,'dayClose']-1)*3+1-1e-4

        elif dayHigh/dayOpen>t5:
            profit = (t5*dayOpen/df.loc[i-1,'dayClose']-1)*3+1-1e-4
        else:
            profit = (df.loc[i,'dayClose']/df.loc[i-1,'dayClose']-1)*3+1-1e-4
        premax = max(df.loc[i-1,'dayClose'],premax) 
        if profit!=1:
            profits.append(profit)
        if ~np.isnan(df.loc[i,'div']):
            profits.append(df.loc[i,'div']/df.loc[i,'dayClose']+1)
        if (df.loc[i,'Date']).split('-')[1]!=prevmon:
            monthlyprof.append(np.prod(profits))
            prevmon = (df.loc[i,'Date']).split('-')[1]
    monthlyprof.append(np.prod(profits))
    for i in range(len(monthlyprof)-1,0,-1):
        monthlyprof[i]=monthlyprof[i]/monthlyprof[i-1]
    return profits


def triplessdivextreturn(df,dfintradict,t='10:30'):
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
    t14=.996
    profit = 1
    profits=[]
    profits2=[]
    monthlyprof = [1]   
    prevmon = '03'
    premax=df.loc[0,'dayClose']
    for i in range(2,len(df)):     
        profit=1
        dayOpen = df.loc[i,'dayOpen']
        dayHigh = df.loc[i,'dayHigh']
        dayLow = df.loc[i,'dayLow']
        dayClose = df.loc[i,'dayClose']
        preLow = df.loc[i,'preLow']
        preHigh = df.loc[i,'preHigh']

        if df.loc[i-1,'dayClose']>t7*df.loc[i-2,'dayClose'] and df.loc[i-2,'ma20']>t8:
            profit = (1-df.loc[i,'dayClose']/df.loc[i-1,'dayClose'])*3+1#short 61% win, gmean 2.28%
        elif i>100 and df.loc[i-1,'dayClose']>premax:
            if dayOpen/df.loc[i-1,'dayClose']>.998:
                profit1 = (dayOpen/df.loc[i-1,'dayClose']-1)*3+1-1e-4# 74% win, gmean 0.85%
                profit2 = (1-dayClose/dayOpen)*3+1-1e-4 # 49% win gmean 4.17%
                profit = profit1*profit2
            else:
                profit = (dayClose/df.loc[i-1,'dayClose']-1)*3+1-1e-4 # 8% win, gmean -3%            
        elif(df.loc[i-1,'dayClose']/df.loc[i-1,'dayOpen']>t9 and df.loc[i-1,'ma20']>t10):
            dftmp = dfintradict[df.loc[i,'Date']]
            try:
                dayOpen1 = dftmp[dftmp['Time']==t].iloc[0].Open
            except:
                dayOpen1 = df.loc[i,'dayOpen']            
            try:
                preLow1 = min(min(dftmp[dftmp.Datetime<dftmp[dftmp['Time']==t].iloc[0].Datetime].Low),df.loc[i].preLow)
                preHigh1 = max(max(dftmp[dftmp.Datetime<dftmp[dftmp['Time']==t].iloc[0].Datetime].High),df.loc[i].preHigh)
            except:
                preLow1=dayOpen
                preHigh1=dayOpen
            try:
                dayLow1 = min(dftmp[dftmp.Datetime>=dftmp[dftmp['Time']==t].iloc[0].Datetime].Low)
                dayHigh1 = max(dftmp[dftmp.Datetime>=dftmp[dftmp['Time']==t].iloc[0].Datetime].High)
            except:
                dayLow1 = min(dftmp.Low)
                dayHigh1 = max(dftmp.High)
            
            if preHigh1>th*df.loc[i-1,'dayClose'] or dayOpen1>th*df.loc[i-1,'dayClose']  and (preHigh1<1.15*df.loc[i-1,'dayClose'] or preHigh1<1.07*dayOpen1):
                profit1 = 3*th-2-1e-4
                profit2 = (1-df.loc[i,'dayClose']/dayOpen1)*3+1-1e-4#short 50% win, gmean 0.52%
                profit = profit1*profit2
            elif dayOpen1<t14*df.loc[i-1,'dayClose']:
                profit1 = (dayOpen1/df.loc[i-1,'dayClose']-1)*3+1-1e-4
                profit2 = (1-df.loc[i,'dayClose']/dayOpen1)*3+1-1e-4#short 61% win, gmean 1%
                profit = profit1*profit2
            else:
                profit = (dayClose/df.loc[i-1,'dayClose']-1)*3+1-1e-4# 62% win, gmean 0.4%
 
        elif (preHigh>th*df.loc[i-1,'dayClose'] or dayOpen>th*df.loc[i-1,'dayClose']) and (preHigh<1.15*df.loc[i-1,'dayClose'] or preHigh<1.07*dayOpen):
            profit = 3*th-2-1e-4#buy
            if dayLow<t2*dayOpen:
                profit2 = (dayClose/(t2*dayOpen)-1)*3+1-1e-4
                profit = profit*profit2

        elif dayOpen/df.loc[i-1,'dayClose']>t1 or dayOpen/df.loc[i-1,'dayClose']<t4:
            profit = ((dayClose/df.loc[i-1,'dayClose'])-1)*3+1-1e-4

        elif dayOpen/df.loc[i-1,'dayClose']<t6:
            earlyOut=0
            profit=1
            if dayHigh/dayOpen>t3:
                outTime = checkOutTime(dfintradict[df.loc[i,'Date']],t3*dayOpen)
                if int(outTime.split(':')[0])<13:
                    profit = (t3*dayOpen/df.loc[i-1,'dayClose']-1)*3+1-1e-4                
                    earlyOut=1
            if earlyOut==0:
                profit *= (df.loc[i,'dayClose']/df.loc[i-1,'dayClose']-1)*3+1-1e-4

        elif dayHigh/dayOpen>t5:
            earlyOut=0
            profit=1
            if dayHigh/dayOpen>t5:
                outTime = checkOutTime(dfintradict[df.loc[i,'Date']],t5*dayOpen)
                if outTime in {'9:30'}:
                    earlyOut=1
            if earlyOut==1:
                dftmp = dfintradict[df.loc[i,'Date']]
                if dftmp[dftmp['Time']=='10:00'].iloc[0].Open<t6*df.loc[i-1,'dayClose']:
                    dayLow = min(dftmp[dftmp.Datetime>=dftmp[dftmp['Time']=='10:00'].iloc[0].Datetime].Low)
                    if dayLow<t13*dftmp[dftmp['Time']=='10:00'].iloc[0].Open:
                        pin = t13*dftmp[dftmp['Time']=='10:00'].iloc[0].Open
                        profit*=(dayClose/pin-1)*3+1-1e-4
            profit *= (t5*dayOpen/df.loc[i-1,'dayClose']-1)*3+1-1e-4
        else:
                profit = (df.loc[i,'dayClose']/df.loc[i-1,'dayClose']-1)*3+1-1e-4
        premax = max(df.loc[i-1,'dayClose'],premax) 
        if profit!=1:
            profits.append(profit)
        if ~np.isnan(df.loc[i,'div']):
            profits.append(df.loc[i,'div']/df.loc[i,'dayClose']+1)
        if (df.loc[i,'Date']).split('-')[1]!=prevmon:
            monthlyprof.append(np.prod(profits))
            prevmon = (df.loc[i,'Date']).split('-')[1]
    monthlyprof.append(np.prod(profits))
    for i in range(len(monthlyprof)-1,0,-1):
        monthlyprof[i]=monthlyprof[i]/monthlyprof[i-1]
    return monthlyprof

def calcannualreturn(profits,df):
    return (np.prod(profits)**(1/len(df)))**252

def checkOutTime(df,p):
    for i in range(len(df)):
        if df.loc[i,'High']>p:
            return df.loc[i,'Time']
    return 'None'

def testssreturnext(df,dfintradict,method='short'):
    if method=='short':
        profits = triplessdivextreturnshort(df,dfintradict)
    else:
        profits = triplessdivextreturn(df,dfintradict)
#    print("{:,}".format(np.prod(profits)))
#    print(gmean(profits))
#    print(calcannualreturn(profits,df))
    return calcannualreturn(profits,df)
    
dfdayext4 = pd.read_csv('dfdayextdiv4.csv')
dfdayext4['Datetime']=pd.to_datetime(dfdayext4.Date)
#dfdayext3 = pd.read_csv('dfdayext3div.csv')
#dfdayext3['Datetime']=pd.to_datetime(dfdayext4.Date)

#dfintradict = np.load('QQQdfintradict.npy').item()
dfintradict = np.load('dfintradict2.npy').item()
#profits = triplessdivextreturn(dfdayext4,dfintradict)
#profits=np.array(profits)
print('overall')
#t2 = .985
#t3 = 1.011
#t4 = .98
#t5=1.0004
#t6 = .999
#t7 = 1.001
#t8 = 1.08
#th=1.011
#params=[]
#res=[]
#for th in tqdm(np.arange(1.001,1.02,0.001)):
#    for t2 in (np.arange(0.97,0.999,0.001)):
#for t3 in tqdm(np.arange(1.001,1.03,0.001)):
#    for t4 in np.arange(0.97,0.99,0.002):
#for t5 in tqdm(np.arange(1.0001,1.01,0.0005)):
#    for t6 in np.arange(.995,1.005,0.001):
#for t7 in tqdm(np.arange(0.999,1.005,0.001)):
#    for t8 in np.arange(1.01,1.15,0.001):
#        res.append(testssreturnext(dfdayext4,dfintradict,t7,t8))
#        params.append((t7,t8))
                    
import time
start=time.time()
print(testssreturnext(dfdayext4,dfintradict))
print((time.time()-start)*166320000000/3600/24/30/12)

#
#print('since 2010-2-11')
#dfdayl10 = dfdayext4.loc[2748:]
#dfdayl10.reset_index(inplace=True,drop=True)
#testssreturnext(dfdayl10,dfintradict)
#
#print('since 2018-1-2, 2018')
#dfdayl10 = dfdayext4.loc[4734:]
#dfdayl10.reset_index(inplace=True,drop=True)
#testssreturnext(dfdayl10,dfintradict)
#
#print('since 2018-5-7, test set')
#dfdayl10 = dfdayext4.loc[4820:]
#dfdayl10.reset_index(inplace=True,drop=True)
#testssreturnext(dfdayl10,dfintradict)
#
#
#def triplereturnMonth(df):
#    monthlyprof = [1]   
#    prevmon = '03'
#    profits=[]
#    df=dfdayext3
#    for i in range(2,len(df)):
#        profits.append((df.loc[i,'dayClose']/df.loc[i-1,'dayClose']-1)*3+1)
#        if (df.loc[i,'Date']).split('-')[1]!=prevmon:
#            monthlyprof.append(np.prod(profits))
#            prevmon = (df.loc[i,'Date']).split('-')[1]
#    monthlyprof.append(np.prod(profits))
#    for i in range(len(monthlyprof)-1,0,-1):
#        monthlyprof[i]=monthlyprof[i]/monthlyprof[i-1]
#    return monthlyprof

