# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 11:23:30 2018

@author: ximing
"""
qqq = pd.read_csv('QQQall.csv')
dia = pd.read_csv('DIA.csv')
dia2 = dia.loc[286:]
dia2.reset_index(drop=True,inplace=True)
dia = dia2
profits=[]
profits2=[]
for i in range(1,len(dia)-1):
    if dia.loc[i,'Adj Close']/dia.loc[i-1,'Adj Close']<qqq.loc[i,'Adj Close']/qqq.loc[i-1,'Adj Close']:
        profits.append((dia.loc[i+1,'Adj Close']/dia.loc[i,'Adj Close']-1)*3+1)
    else:
        profits.append((qqq.loc[i+1,'Adj Close']/qqq.loc[i,'Adj Close']-1)*3+1)
    profits2.append((qqq.loc[i+1,'Adj Close']/qqq.loc[i,'Adj Close']-1)*3+1)
        