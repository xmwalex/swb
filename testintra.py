# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:14:50 2018

@author: ximing
#"""
#def test(dftradict,t1,t2):
#    profits=[]
#    for day,dfday in dfintradict.items():
#    #    print(day)
#        pin=0
#        for i in range(len(dfday)):
#            if pin==0 and dfday.loc[i,'Close']<t1*dfday.loc[i,'Open']:
#                pin=dfday.loc[i,'Close']
#            elif pin!=0:
#                if dfday.loc[i,'Close']>t2*pin:
#                    profits.append(dfday.loc[i,'Close']/pin)
#                    pin=0
#        if pin!=0:
#            profits.append(dfday.loc[i,'Close']/pin)
#    return profits
#
#p=[]
#para=[]
#for t1 in tqdm(np.arange(.99,.999,0.001)):
#    for t2 in np.arange(1.001,1.01,0.001):
#       para.append([t1,t2])
#       p.append(np.prod(test(dfintradict,t1,t2)))

t1 = 1.008
t2 = .985
t3 = 1.0117
t4 = .982
t5=1.0004
t6 = 1.0005
t7 = 1.001
t8 = 1.08
t9 = 1.005
t10 = .88
t11 = .925
th=1.011
profit = 1
profits=[]
profits2 = []
#    monthlyprof = [1]   
#    prevmon = '03'
for i in range(2,len(df)):     
    profit=1
    dayOpen = df.loc[i,'dayOpen']
    dayHigh = df.loc[i,'dayHigh']
    dayLow = df.loc[i,'dayLow']
    dayClose = df.loc[i,'dayClose']
    preLow = df.loc[i,'preLow']
    preHigh = df.loc[i,'preHigh']
    if df.loc[i-1,'dayClose']>t7*df.loc[i-2,'dayClose'] and df.loc[i-2,'ma20']>t8:
        profit = (1-df.loc[i,'dayClose']/df.loc[i-1,'dayClose'])*3+1#short +2.28% 
    elif(df.loc[i-1,'dayClose']/df.loc[i-1,'dayOpen']>t9 and df.loc[i-1,'ma10']>t10):
        if df.loc[i,'preHigh']>th*df.loc[i-1,'dayClose'] or dayOpen>th*df.loc[i-1,'dayClose']:
            profit = 3*th-2
        else:
            profit = (dayOpen/df.loc[i-1,'dayClose']-1)*3+1#buy
        profits.append(profit-1e-4)# long, +0.144%
        if df.loc[i,'dayLow']<t11*dayOpen:#short
            profit = (1-t11)*3+1
        else:
            profit = (1-df.loc[i,'dayClose']/dayOpen)*3+1#short
        # short, 0.22%
    elif df.loc[i,'preHigh']>th*df.loc[i-1,'dayClose'] or dayOpen>th*df.loc[i-1,'dayClose']:
        profit = 3*th-2#buy
        if df.loc[i,'dayLow']<t2*dayOpen:
            profits.append(profit-1e-4)
            profit = (df.loc[i,'dayClose']/(t2*dayOpen)-1)*3+1
    elif dayOpen/df.loc[i-1,'dayClose']>t1:
#            print('wrong'+str(i))
        profit = (((df.loc[i,'dayClose'])/df.loc[i-1,'dayClose'])-1)*3+1
    elif dayOpen/df.loc[i-1,'dayClose']<t4:
        profit = (((df.loc[i,'dayClose'])/df.loc[i-1,'dayClose'])-1)*3+1
    elif dayOpen/df.loc[i-1,'dayClose']<t6:
        earlyOut=0
        if df.loc[i,'dayHigh']/dayOpen>t3:
            outTime = checkOutTime(dfintradict[df.loc[i,'Date']],t3*dayOpen)
            if int(outTime.split(':')[0])<13:
                profit = (t3*dayOpen/df.loc[i-1,'dayClose']-1)*3+1      
                earlyOut=1
        if earlyOut==0:
            profit = (df.loc[i,'dayClose']/df.loc[i-1,'dayClose']-1)*3+1
    elif df.loc[i,'dayHigh']/dayOpen>t5:
        profit = (t5*dayOpen/df.loc[i-1,'dayClose']-1)*3+1
    else:
        profit = (df.loc[i,'dayClose']/df.loc[i-1,'dayClose']-1)*3+1
    if profit!=1:
        profits.append(profit-1e-4)
    if ~np.isnan(df.loc[i,'div']):
        profits.append(df.loc[i,'div']/df.loc[i,'dayClose']+1)