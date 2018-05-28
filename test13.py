# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:30:08 2018

@author: ximing
"""

extHigh,extLow=0,999999
for i in tqdm(range(8438,len(dfintra))):
    if (dfintra.loc[i,'Datetime'].hour>=10 and dfintra.loc[i,'Datetime'].hour<=15):
        if extHigh!=0:
            dfintra.loc[i,'preHigh']=extHigh
            dfintra.loc[i,'preLow']=extLow
            extHigh=0
            extLow=999999
        else:
            continue
    elif dfintra.loc[i,'Datetime'].hour==9 and dfintra.loc[i,'Datetime'].minute==30:
        dfintra.loc[i,'preHigh']=extHigh
        dfintra.loc[i,'preLow']=extLow
        extHigh=0
        extLow=999999
    else:
        extHigh = max(extHigh,dfintra.loc[i,'High'])
        extLow = min(extLow,dfintra.loc[i,'Low'])