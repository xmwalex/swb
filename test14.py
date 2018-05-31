# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:25:57 2018

@author: ximing
"""

def checkLow2(df,p):
    if '14:30' not in set(df['Time']) or '14:00' not in set(df['Time']):
        return -1
    if df.loc[df['Time']=='14:30','Open'].iloc[0]<p*df.loc[df['Time']=='14:00','Open'].iloc[0]:
        return df.loc[1,'dayClose']/df.loc[df['Time']=='14:30','Open'].iloc[0]
    if df.loc[df['Time']=='15:00','Open'].iloc[0]<p*df.loc[df['Time']=='14:30','Open'].iloc[0]:
        return df.loc[1,'dayClose']/df.loc[df['Time']=='15:00','Open'].iloc[0]
    if df.loc[df['Time']=='15:30','Open'].iloc[0]<p*df.loc[df['Time']=='15:00','Open'].iloc[0]:
        return df.loc[1,'dayClose']/df.loc[df['Time']=='15:30','Open'].iloc[0]
    return -1

def checkLow(df,p):
    if '14:30' not in set(df['Time']) or '14:00' not in set(df['Time']):
        return -1
    for i in range(len(df)):
        pbar=0
        if df.loc[i,'Time']=='14:00':
            pbar = df.loc[i,'Open']*p
        if pbar!=0 and df.loc[i,'Low']<pbar:
            return df.loc[i,'dayClose']/pbar
    return -1


import numpy as np
def triplessdivextreturn(df,dfintradict):
    t5=1.0004
    t2=.985
    t3=1.0118
    t1=1.006
    t4=.989
    t6=.9994
#    t7=.999
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
            dftmp = dfintradict[df.loc[i,'Date']]
            if (('14:30' in set(dftmp['Time'])) and ('14:00' in set(dftmp['Time']))) and dftmp.loc[dftmp['Time']=='15:00','Open'].iloc[0]/dftmp.loc[1,'dayOpen']<.99:
                if int(outTime.split(':')[0])<13:
                    profits.append(profit)
                    profit = dftmp.loc[1,'dayClose']/dftmp.loc[dftmp['Time']=='15:30','Open'].iloc[0]
#                try:
#                    p2 = checkLow2(dfintradict[df.loc[i,'Date']],t7)
#                    if p2>0:
#                        profits2.append(p2)
#                except:
#                    print(i)
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
    return profits

profits=triplessdivextreturn(dfdayext3,dfintradict)
print(np.prod(profits))
print(gmean(profits))
print(len(profits))
#p=[]
#para=[]
#for t7 in tqdm(np.arange(.98,1,0.0001)):
#    p.append(np.prod(triplessdivextreturn(dfdayext3,dfintradict,t7)))
#    para.append(t7)

#profits=triplessdivextreturn(dfdayext3,dfintradict,.999)