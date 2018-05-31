# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:45:09 2018

@author: ximing
"""


import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def triplessdivextreturn(df,dfintradict):
    t5=1.0004
    t2=.985
    t3=1.0118
    t1=1.006
    t4=.989
    t6=.9994
    th=1.01
    profit = 1
    profits=[]
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
        elif df.loc[i,'dayOpen']/df.loc[i-1,'dayClose']>t1:
            profit = (((df.loc[i,'dayClose'])/df.loc[i-1,'dayClose'])-1)*3+1
        elif df.loc[i,'dayOpen']/df.loc[i-1,'dayClose']<t4:
            profit = (((df.loc[i,'dayClose'])/df.loc[i-1,'dayClose'])-1)*3+1
        elif df.loc[i,'dayOpen']/df.loc[i-1,'dayClose']<t6:
            earlyOut=0
            if df.loc[i,'dayHigh']/df.loc[i,'dayOpen']>t3:
                outTime = checkOutTime(dfintradict[df.loc[i,'Date']],t3*df.loc[i,'dayOpen'])
                if int(outTime.split(':')[0])<13:
                    profit = (t3*df.loc[i,'dayOpen']/df.loc[i-1,'dayClose']-1)*3+1                
                    earlyOut=1
            if earlyOut==0:
                profit = (df.loc[i,'dayClose']/df.loc[i-1,'dayClose']-1)*3+1
        elif df.loc[i,'dayHigh']/df.loc[i,'dayOpen']>t5:
            profit = (t5*df.loc[i,'dayOpen']/df.loc[i-1,'dayClose']-1)*3+1
        else:
            profit = (df.loc[i,'dayClose']/df.loc[i-1,'dayClose']-1)*3+1
        if profit!=1:
            profits.append(profit-1e-4)
        if ~np.isnan(df.loc[i,'div']):
            profits.append(df.loc[i,'div']/df.loc[i,'dayClose']+1)
        if (df.loc[i,'Date']).split('-')[1]!=prevmon:
            monthlyprof.append(np.prod(profits))
            prevmon = (df.loc[i,'Date']).split('-')[1]
    monthlyprof.append(np.prod(profits))
    for i in range(len(monthlyprof)-1,0,-1):
        monthlyprof[i]=monthlyprof[i]/monthlyprof[i-1]
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
    print(np.prod(profits))
    print(gmean(profits))
    print(calcannualreturn(profits,df))
    

dfdayext3 = pd.read_csv('dfdayextdiv.csv')
dfdayext3['Datetime']=pd.to_datetime(dfdayext3.Date)

dfintradict = np.load('dfintradict.npy').item()

testssreturnext(dfdayext3,dfintradict)