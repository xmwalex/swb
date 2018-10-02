# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 13:30:53 2018

@author: ximing
"""
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

dfdayext3 = pd.read_csv('dfdayext3div.csv')
dfdayext3['Datetime']=pd.to_datetime(dfdayext3.Date)

cash = 0.3
stock = 0.7
stockbase = stock
nav=[]
for i in range(2,len(dfdayext3)):
    stock*=(dfdayext3.loc[i,'dayClose']/dfdayext3.loc[i-1,'dayClose']-1)*3+1
    nav.append(cash+stock)
    if stock>stockbase*2:
        cash = 0.3*(cash+stock)
        stock = 0.7*(cash+stock)
        stockbase = stock
    elif stock<.1*stockbase:
        cash = 0.3*(cash+stock)
        stock = 0.7*(cash+stock)
        stockbase = stock
        
def testdf(df,day,per,pre):
    profits=[]
    i=0
    while i<len(df):
        if i>pre and df.loc[i,'dayClose']<per*df.loc[i-pre,'dayClose']:
            profits.append((df.loc[i+day,'dayClose']/df.loc[i,'dayClose']-1)*3+1)
            i+=day
        else:
            i+=1
    return profits

prof=[]
para=[]
for day in tqdm(range(1,10)):
    for per in np.arange(.9,.99,0.01):
        for pre in range(1,10):
           para.append([day,per,pre])
           prof.append(np.prod(testdf(dfdayl10,day,per,pre)))
       
per=.94
day=4
for i in range(len(dfdayext3)):
    if i>3 and df.loc[i,'dayClose']<per*df.loc[i-3,'dayClose']:
        profits.append((df.loc[i+day,'dayClose']/df.loc[i,'dayClose']-1)*3+1)    
        
        
        
prof=[]
para=[]
for day in tqdm(range(1,10)):
    for per in np.arange(.9,.99,0.01):
        for pre in range(1,10):
           para.append([day,per,pre])
           prof.append(np.prod(testdf(dfdayl10,day,per,pre)))
        
profits=[]
weights=[2500]
for i in np.arange(0,len(qqqall)-250,21):
    df = qqqall.loc[i:]
    df.reset_index(drop=True,inplace=True)
    profits.append(triplereturn(df)*weights[-1])
    weights.append(weights[-1]*1.003)
#    profits.append(calcannualreturn(triplereturn(df),df))



def longtermstable(dfall,fund,cashoutrate=2,cashoutquota=.5):
    stock=1
    profits=[]
    weights=[1]
    cashout=0
    cashouts=[]
    stocks=[]
    monthlyprice=[] 
    if len(fund)<len(dfall):
        a=[fund.loc[0,'Adj Close']]*(len(dfall)-len(fund))
        adf = pd.DataFrame({'Adj Close':a})
        fund = pd.concat([adf,fund])
        fund.reset_index(drop=True,inplace=True)
    elif len(fund)>len(dfall):
        fund = fund.loc[len(fund)-len(dfall):]
        fund.reset_index(drop=True,inplace=True)
    for i in np.arange(0,len(dfall),21):
        if i==0:
            df = dfall[i:i+21]
            dffund = fund[i:i+21]
        else:
            df = dfall[i-1:i+21]
            dffund = fund[i-1:i+21]
        df.reset_index(drop=True,inplace=True)
        dffund.reset_index(drop=True,inplace=True)
        if i<len(dfall):
            if i>250 and dfall.loc[i,'Adj Close']<.5*max(dfall.loc[:i-1]['Adj Close']):
                weights.append(weights[-1]*1.0025)
            else:
                weights.append(weights[-1]*1.0025)
            stock = stock*triplereturn(df)+weights[-1]
#            cashout-=weights[-1]
        else:
            stock = stock*triplereturn(df)
        cashout*=(holdreturn(dffund)-1)+1
#        cashout = 1.003*cashout
        if stock>cashoutrate*(sum(weights[:-1])):
            cashout+=stock*cashoutquota
            stock=stock*(1-cashoutquota)
        stocks.append(stock)
        cashouts.append(cashout)
#        profits.append(stock+cashout)
        if len(weights)>12:
#            profits.append((stock+cashout)/sum(weights))
            profits.append((stock+cashout-sum(weights[-1:]))/sum(weights[:-1]))
#            break
    return profits
#    if min(profits)>.7:
#        return profits[-1]
#    else:
#        return 0
        
p=[]
para=[]
for t in tqdm(np.arange(1.1,15,1.1)):
    for q in np.arange(.3,.99,.01):
        avg=[]
#        for s in np.arange(0,4801,100):
#            dftmp = dworst.loc[s:s+1e4]
#            dftmp.reset_index(inplace=True,drop=True)
        avg.append(longtermstable(dworst,t,q)[-1])
        p.append(np.mean(avg))
        para.append([t,q])

def dingtou(dfall):
    profits=[]
    weights=[1]
    for i in np.arange(0,len(dfall),21):
        df = dfall.loc[i:]
        df.reset_index(drop=True,inplace=True)
        profits.append(triplereturn(df)*weights[-1])
        weights.append(weights[-1]*1.0025)
#        profits.append(weights[-1]*calcannualreturn(triplereturn(df),df))
    return [profits,weights]
        

def dingtoutriple(dfall):
    stock=1
    profits=[]
    weights=[1]
    cashout=0
    cashouts=[]
    stocks=[]
    monthlyprice=[] 
    for i in np.arange(0,len(dfall),21):
        if i==0:
            df = dfall[i:i+21]
        else:
            df = dfall[i-1:i+21]
        df.reset_index(drop=True,inplace=True)
        if i<len(dfall):
            weights.append(weights[-1]*1.0025)
            stock = stock*triplereturn(df)+weights[-1]
#            cashout-=weights[-1]
        else:
            stock = stock*triplereturn(df)
#        cashout*=holdreturn(df)
#        cashout = 1.006*cashout
#        if stock>2*(sum(weights[:-1])):
#            cashout+=stock/2
#            stock/=2
        stocks.append(stock)
        cashouts.append(cashout)
#        profits.append(stock+cashout)
        profits.append((stock+cashout)/sum(weights))
    return [profits,weights,stocks,cashouts,stock,cashout]
        
def dingtou(dfall):
    stock=1
    profits=[]
    weights=[1]
    cashout=0
    cashouts=[]
    stocks=[]
    monthlyprice=[] 
    for i in np.arange(0,len(dfall),21):
        if i==0:
            df = dfall[i:i+21]
        else:
            df = dfall[i-1:i+21]
        df.reset_index(drop=True,inplace=True)
        if i<len(dfall):
            weights.append(weights[-1]*1.0025)
            stock = stock*holdreturn(df)+weights[-1]
#            cashout-=weights[-1]
        else:
            stock = stock*holdreturn(df)
#        cashout*=holdreturn(df)
#        cashout = 1.006*cashout
#        if stock>2*(sum(weights[:-1])):
#            cashout+=stock/2
#            stock/=2
        stocks.append(stock)
        cashouts.append(cashout)
#        profits.append(stock+cashout)
        profits.append((stock+cashout)/sum(weights))
    return [profits,weights,stocks,cashouts,stock,cashout]


for i in np.arange(0,len(profits),12):
    if i+12<len(profits):
        print(profits[i+12]/profits[i])
    else:
        print(profits[-1]/profits[i])        
        
        
        

def dingtou(dfall):
    stock=1
    profits=[1]
    weights=[1]
    cashout=0
    cashouts=[]
    stocks=[]
    monthlyprice=[] 
    month=0
    for i in np.arange(0,len(dfall)-1):
        stock *= (dfall.loc[i+1,'Adj Close']/dfall.loc[i,'Adj Close']-1)*3+1
        if i//21>month:
            if dfall.loc[i,'Adj Close']<.985*dfall.loc[i-3,'Adj Close']:
                weights.append(weights[-1]*1.0025)
                stock += weights[-1]
                month+=1
    
    
    
stock=1
profits=[]
weights=[1]
cashout=0
cashouts=[]
stocks=[]
monthlyprice=[] 
for i in np.arange(0,len(navs),21):
    if i<len(navs)-20:
        weights.append(weights[-1]*1.0025)
        stock = stock*navs[i+21]/navs[i]+weights[-1]
#            cashout-=weights[-1]
    else:
        stock = stock*(navs[-1]/navs[i])
#        cashout*=holdreturn(df)
    cashout = 1.006*cashout
    if stock>cashoutrate*(sum(weights[:-1])):
        cashout+=stock/2
        stock/=2
    stocks.append(stock)
    cashouts.append(cashout)
#        profits.append(stock+cashout)
    if len(weights)>16:
        profits.append((stock+cashout-sum(weights[-12:]))/sum(weights[:-12]))

bear = qqqall.loc[240:1000]
bear.reset_index(drop=True,inplace=True)
qqqall['dayClose']=qqqall['Adj Close']
bear.loc[0,'dayClose']=qqqall.loc[len(qqqall)-1,'Adj Close']
for i in tqdm(range(1,len(bear))):
    bear.loc[i,'dayClose']=bear.loc[i-1,'dayClose']*bear.loc[i,'Adj Close']/bear.loc[i-1,'Adj Close']
dworst = pd.concat([qqqall,bear])
dworst.reset_index(drop=True,inplace=True)
dworst['Adj Close']=dworst['dayClose']

spymq=pd.read_csv('spymq.csv')
bear1 = spymq.loc[240:1000]
bear1.reset_index(drop=True,inplace=True)
spyworst = pd.concat([spymq,bear1])
spyworst.reset_index(drop=True,inplace=True)

spy=pd.read_csv('spyall.csv')
bear1 = spy.loc[3700:4150]
bear1.reset_index(drop=True,inplace=True)
spy['dayClose']=spy['Adj Close']
bear1.loc[0,'dayClose']=spy.loc[len(spy)-1,'Adj Close']
for i in tqdm(range(1,len(bear1))):
    bear1.loc[i,'dayClose']=bear1.loc[i-1,'dayClose']*bear1.loc[i,'Adj Close']/bear1.loc[i-1,'Adj Close']
spyworst = pd.concat([spy,bear1])
spyworst.reset_index(drop=True,inplace=True)
spyworst['Adj Close']=spyworst['dayClose']
plt.plot(spyworst['Adj Close'])

qqqall=dworst
bear = qqqall
bear.reset_index(drop=True,inplace=True)
qqqall['dayClose']=qqqall['Adj Close']
bear.loc[0,'dayClose']=qqqall.loc[len(qqqall)-1,'Adj Close']
for i in tqdm(range(1,len(bear))):
    bear.loc[i,'dayClose']=bear.loc[i-1,'dayClose']*bear.loc[i,'Adj Close']/bear.loc[i-1,'Adj Close']
dworst = pd.concat([qqqall,bear])
dworst.reset_index(drop=True,inplace=True)
dworst['Adj Close']=dworst['dayClose']
