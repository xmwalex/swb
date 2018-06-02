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
 
def triplessdivextreturn(df,dfintradict,t7,t8,t='10:00'):
    t1 = 1.007
    t2 = .985
    t3 = 1.0114
    t4 = .995
    t5=1.0004
    t6 = 1.0003
#    t7 = 1.0007
#    t8 = 1.08
    t9 = 1.005
    t10 = .88
    th=1.011
    profit = 1
    profits=[]
#    monthlyprof = [1]   
#    prevmon = '03'
    for i in range(2,len(df)):     
        profit=1
        dftmp = dfintradict[df.loc[i,'Date']]
        try:
            dayOpen = dftmp[dftmp['Time']==t].iloc[0].Open
        except:
            dayOpen = df.loc[i,'dayOpen']            
        try:
            preLow = min(min(dftmp[dftmp.Datetime<dftmp[dftmp['Time']==t].iloc[0].Datetime].Low),df.loc[i].preLow)
            preHigh = max(max(dftmp[dftmp.Datetime<dftmp[dftmp['Time']==t].iloc[0].Datetime].High),df.loc[i].preHigh)
        except:
            preLow=dayOpen
            preHigh=dayOpen
#            print(i)
        try:
            dayLow = min(dftmp[dftmp.Datetime>=dftmp[dftmp['Time']==t].iloc[0].Datetime].Low)
            dayHigh = max(dftmp[dftmp.Datetime>=dftmp[dftmp['Time']==t].iloc[0].Datetime].High)
        except:
            dayLow = min(dftmp.Low)
            dayHigh = max(dftmp.High)
        dayClose = df.loc[i,'dayClose']
        if df.loc[i-1,'dayClose']>t7*df.loc[i-2,'dayClose'] and df.loc[i-2,'ma20']>t8:
            profit = (1-df.loc[i,'dayClose']/df.loc[i-1,'dayClose'])*3+1#short
        elif(df.loc[i-1,'dayClose']/df.loc[i-1,'dayOpen']>t9 and df.loc[i-1,'ma10']>t10):
            if preHigh>th*df.loc[i-1,'dayClose'] or dayOpen>th*df.loc[i-1,'dayClose']:
                profit = 3*th-2
            else:
                profit = (dayOpen/df.loc[i-1,'dayClose']-1)*3+1#buy
            profits.append(profit-1e-4)
            if dayLow<.925*dayOpen:#short
                profit = (1-.925)*3+1
            else:
                profit = (1-df.loc[i,'dayClose']/dayOpen)*3+1#short
        elif preHigh>th*df.loc[i-1,'dayClose'] or dayOpen>th*df.loc[i-1,'dayClose']:
            profit = 3*th-2#buy
            if dayLow<t2*dayOpen:
                profits.append(profit-1e-4)
                profit = (dayClose/(t2*dayOpen)-1)*3+1
        elif dayOpen/df.loc[i-1,'dayClose']>t1:
#            print('wrong'+str(i))
            profit = ((dayClose/df.loc[i-1,'dayClose'])-1)*3+1
        elif dayOpen/df.loc[i-1,'dayClose']<t4:
            profit = (((df.loc[i,'dayClose'])/df.loc[i-1,'dayClose'])-1)*3+1
        elif dayOpen/df.loc[i-1,'dayClose']<t6:
            earlyOut=0
            if dayHigh/dayOpen>t3:
                outTime = checkOutTime(dfintradict[df.loc[i,'Date']],t3*dayOpen)
                if int(outTime.split(':')[0])<13:
                    profit = (t3*dayOpen/df.loc[i-1,'dayClose']-1)*3+1                
                    earlyOut=1
            if earlyOut==0:
                profit = (df.loc[i,'dayClose']/df.loc[i-1,'dayClose']-1)*3+1
        elif dayHigh/dayOpen>t5:
            profit = (t5*dayOpen/df.loc[i-1,'dayClose']-1)*3+1
        else:
            profit = (df.loc[i,'dayClose']/df.loc[i-1,'dayClose']-1)*3+1
        if profit!=1:
            profits.append(profit-1e-4)
        if ~np.isnan(df.loc[i,'div']):
            profits.append(df.loc[i,'div']/df.loc[i,'dayClose']+1)
#        if (df.loc[i,'Date']).split('-')[1]!=prevmon:
#            monthlyprof.append(np.prod(profits))
#            prevmon = (df.loc[i,'Date']).split('-')[1]
#    monthlyprof.append(np.prod(profits))
#    for i in range(len(monthlyprof)-1,0,-1):
#        monthlyprof[i]=monthlyprof[i]/monthlyprof[i-1]
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
    print("{:,}".format(np.prod(profits)/1e8))
    print(gmean(profits))
    print(calcannualreturn(profits,df))
    

dfdayext3 = pd.read_csv('dfdayextdiv.csv')
dfdayext3['Datetime']=pd.to_datetime(dfdayext3.Date)

dfintradict = np.load('dfintradict.npy').item()

#testssreturnext(dfdayext3,dfintradict)
#
#testssreturnext(dfdaytrain,dfintradict)
#
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
p=[]
para=[]
for t1 in tqdm(np.arange(.995,1.005,.001)):
    for t2 in np.arange(.85,.95,0.01):
        p.append(np.prod(triplessdivextreturn(dfdayext3,dfintradict,t1,t2)))
        para.append([t1,t2])
#
#
#p=[]
#para=[]
#for t1 in tqdm(np.arange(1.001,1.0015,.0001)):
#        p.append(np.prod(triplessdivextreturn(dfdayext3,dfintradict,t1)))
#        para.append(t1)

#p=[]
#para = ['10:00','10:30','11:00','11:30','12:00','12:30','13:00','13:30','14:00','14:30','15:00']
#for i in tqdm(range(len(para))):
#    t = para[i]
#    p.append(np.prod(triplessdivextreturn(dfdayext3,dfintradict,t)))
