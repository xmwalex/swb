# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:01:25 2018

@author: ximing
"""

prev_ratio=0
c=0
for i in (range(len(unadj_df))):
    un_adj_p = unadj_df.loc[i,'dayClose']
    day = unadj_df.loc[i,'Date']
    adj_p = adj_df[adj_df['Date']==day].iloc[0]['Close']
    ratio = un_adj_p/adj_p
    print(ratio)
    if prev_ratio>0 and (ratio/prev_ratio>1.1 or ratio/prev_ratio<.9):
        print(ratio/prev_ratio)
        print(day)
        c+=1
        if c>2:
            break
    prev_ratio = ratio