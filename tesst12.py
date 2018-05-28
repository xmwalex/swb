# -*- coding: utf-8 -*-
"""
Created on Fri May 25 22:50:40 2018

@author: ximing
"""
profits=[]
times=[]
pin=-1
profit=-1
for i in range(len(df)):
#        if i>100:
#            break
    if df.loc[i,'Datetime'].hour==15 and df.loc[i,'Datetime'].minute==30:
        if pin!=-1:
            print(i)
        pin = df.loc[i,'Close']
        profit=-1
    if pin!=-1 and ((df.loc[i,'Datetime'].hour>=16) or (df.loc[i,'Datetime'].hour<=9 and df.loc[i,'Datetime'].minute<30)):
        if df.loc[i,'High']>th*pin:
            profit = th*3-2
            pin=-1
            t=df.loc[i,'Datetime']
    if ((df.loc[i,'Datetime'].hour in {10,11}) or (df.loc[i,'Datetime'].hour==9 and df.loc[i,'Datetime'].minute==30)) and pin!=-1:
        profit = min((df.loc[i,'Open']/pin-1)*3+1,th*3-2)
        pin=-1
        t=df.loc[i,'Datetime']
#        print(str(pin)+' '+str(i))
#    if profit>1.024:
#        break
    if profit!=-1:
        profits.append(profit-1e-4)
        times.append(t)
        profit=-1