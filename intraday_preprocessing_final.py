# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:51:12 2018

@author: ximing
"""

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def preprocessing(unadj_intra,file_base_name):
    df = pd.read_csv(unadj_intra)    
    c,h,l=[],[],[]
    df['Datetime']=''
    for i in tqdm(range(len(df.index))): 
        df.iloc[i,7]=datetime.strptime(df.loc[i,'Date']+' '+df.loc[i,'Time'],'%m/%d/%Y %H:%M')
    
    for i in tqdm(range(len(df.index))):
        df.loc[i,'Date'] = df.loc[i,'Datetime'].strftime('%Y-%m-%d')
    df['dayOpen']=''
    df['dayHigh']=''
    df['dayLow']=''
    df['dayClose']=''
    o=0
    h=0
    c=0
    backup_c=0
    l=9999
    for i in tqdm(range(len(df.index))): 
        if df.loc[i,'Time']=='9:30':
            if o!=0 or h!=0 or l!=9999 or c!=0:
                if backup_c==0:
                    break
            o=df.loc[i,'Open']
            h = max(h,df.loc[i,'High'])
            l = min(l,df.loc[i,'Low'])
            backup_c=0
        elif df.loc[i,'Time']=='16:00':
            c = df.loc[i,'Open']
            if o==0 or h==0 or c==0 or l==9999:
                print('error!')
                break
            df.loc[i,'dayOpen']=o
            df.loc[i,'dayHigh']=h
            df.loc[i,'dayLow']=l
            df.loc[i,'dayClose']=c
            h=0
            l=9999
            o=0
            c=0
        elif df.loc[i,'Datetime'].hour<16 and df.loc[i,'Datetime'].hour>9:
            h = max(h,df.loc[i,'High'])
            l = min(l,df.loc[i,'Low'])
            if df.loc[i,'Time']=='13:00':
                backup_c = df.loc[i,'Open']
                df.loc[i,'dayOpen']=o
                df.loc[i,'dayHigh']=h
                df.loc[i,'dayLow']=l
                df.loc[i,'dayClose']=backup_c
            if o==0:
                o=df.loc[i,'Open']   
        
    df.to_csv(file_base_name+'v1.csv',index=False)
    #df = pd.read_csv('qqqv3.csv')
    
    tqdm.pandas()
    dfdate = df[(df['Time']=='16:00')|(df['Time']=='13:00')][['Date','Time','dayOpen','dayHigh','dayLow','dayClose']]
    
    def processUnit(x,df):
        a=df[(df['Date']==x[0]) & (df['Time']=='16:00')]
        if len(a)>0:
            x['dayOpen']=float(a['dayOpen'])
            x['dayHigh']=float(a['dayHigh'])
            x['dayLow']=float(a['dayLow'])
            x['dayClose']=float(a['dayClose'])
        else:
            b=df[(df['Date']==x[0]) & (df['Time']=='13:00')]
            if len(b)>0:
                x['dayOpen']=float(b['dayOpen'])
                x['dayHigh']=float(b['dayHigh'])
                x['dayLow']=float(b['dayLow'])
                x['dayClose']=float(b['dayClose'])
        return x

    df=df.progress_apply(processUnit,args=(dfdate,),axis=1)
    
    df.to_csv(file_base_name+'v2.csv',index=False)
#    df = pd.read_csv('QQQintra_processed.csv')
#    df['Datetime']=pd.to_datetime(df.Datetime)
    

def detectSplit(unadj_df_file,adj_df_day_file,file_base_name):
    unadj_df = pd.read_csv(unadj_df_file)
    unadj_df['Datetime']=pd.to_datetime(unadj_df['Datetime'])
    adj_df = pd.read_csv(adj_df_day_file)        
    for i in tqdm(range(len(adj_df.index))):
       adj_df.loc[i,'Date'] = datetime.strptime(adj_df.loc[i,'Date'],'%m/%d/%Y').strftime('%Y-%m-%d')   
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


def adjustPrice(df):
        # adjust QQQ price.
    #    for i in tqdm(range(3846)):
    #        df.loc[i,'Open']=df.loc[i,'Open']/2
    #        df.loc[i,'High']=df.loc[i,'High']/2
    #        df.loc[i,'Low']=df.loc[i,'Low']/2
    #        df.loc[i,'Close']=df.loc[i,'Close']/2    
    #        df.loc[i,'Volume']=df.loc[i,'Volume']*2    
    #        df.loc[i,'dayOpen']=df.loc[i,'dayOpen']/2
    #        df.loc[i,'dayHigh']=df.loc[i,'dayHigh']/2
    #        df.loc[i,'dayLow']=df.loc[i,'dayLow']/2
    #        df.loc[i,'dayClose']=df.loc[i,'dayClose']/2    
    
    
    qqq2=pd.read_csv('QQQall.csv')
    qqq2=calcfeatures(qqq2)
    qqq2=qqq2[['ma10','ma20','ma200','Date']]
    df_all = pd.merge(df,qqq2,how='inner',left_on='Date',right_on='Date')
    
    df=df_all
    
    divdf = pd.read_csv('qqqdiv.csv')
    divdf['Datetime']=pd.to_datetime(divdf.Date)
    for i in tqdm(range(len(divdf.index))):
        divdf.loc[i,'Date'] = divdf.loc[i,'Datetime'].strftime('%Y-%m-%d')
    divdf.drop('Datetime',axis=1,inplace=True)
    df = pd.merge(df,divdf,how='left',on='Date')
    
    def binExtended(dfintra):
        extHigh,extLow=0,999999
        for i in tqdm(range(len(dfintra))):
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
        return dfintra
    
    df2 = binExtended(df)
    dfdayext2 = df2[~np.isnan(df2.preHigh)]
    dfdayext2.reset_index(inplace=True,drop=True)
    
    #divdf = pd.read_csv('qqqdiv.csv')
    #divdf['Datetime']=pd.to_datetime(divdf.Date)
    #
    #for i in tqdm(range(len(divdf.index))):
    #    divdf.loc[i,'Date'] = divdf.loc[i,'Datetime'].strftime('%Y-%m-%d')
    #divdf.drop('Datetime',axis=1,inplace=True)
    #dfdayext3 = pd.merge(dfdayext2,divdf,how='left',on='Date')
    
    cols=['Date', 'Time', 'Volume', 'Datetime',
           'dayOpen', 'dayHigh', 'dayLow', 'dayClose', 'ma10', 'ma20', 'ma200',
           'preHigh', 'preLow', 'div']
    dfdayext3 = dfdayext2[cols]
    dfdayext3.to_csv('dfdayextdiv.csv',index=False)
    
    
    dfintra.drop(['ma10','ma20','ma200'],axis=1,inplace=True)
    dfintradict=dict()
    dates = set(dfintra.Date)
    timesets = {'11:30','13:00','14:30','10:30','12:30','9:30',
                '12:00','15:30','11:00','14:00','15:00','10:00','13:30'}
    for d in tqdm(dates):
    #    print(d)
        tmp = dfintra[(dfintra['Date']==d)&(dfintra['Time'].apply(lambda x: x in timesets))]
        tmp.reset_index(inplace=True,drop=True)
        dfintradict[d]=tmp
        
    np.save('dfintradict.npy',dfintradict)
