# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:08:20 2018

@author: ximing
"""
import numpy as np
import pandas as pd
from scipy.stats import gmean

from tqdm import tqdm
df = pd.read_csv('spyall.csv')
df['Datetime']=pd.to_datetime(df.Date)

for i in range(50,len(df)):
    df.loc[i,'vol50'] = df.loc[i,'Volume']/np.mean(df.loc[i-50:i].Volume)
    
for i in tqdm(range(20,len(df))):
    df.loc[i,'vol20'] = df.loc[i,'Volume']/np.mean(df.loc[i-20:i].Volume)
    
for i in tqdm(range(200,len(df))):
    df.loc[i,'vol200'] = df.loc[i,'Volume']/np.mean(df.loc[i-200:i].Volume)
    
for i in tqdm(range(10,len(df))):
    df.loc[i,'d1c'] = df.loc[i,'Adj Close']/df.loc[i-1,'Adj Close']
    df.loc[i,'d2c'] = df.loc[i,'Adj Close']/df.loc[i-2,'Adj Close']
    df.loc[i,'d3c'] = df.loc[i,'Adj Close']/df.loc[i-3,'Adj Close']
    df.loc[i,'d4c'] = df.loc[i,'Adj Close']/df.loc[i-4,'Adj Close']
    df.loc[i,'d5c'] = df.loc[i,'Adj Close']/df.loc[i-5,'Adj Close']
    df.loc[i,'d6c'] = df.loc[i,'Adj Close']/df.loc[i-6,'Adj Close']
    df.loc[i,'d7c'] = df.loc[i,'Adj Close']/df.loc[i-7,'Adj Close']
    df.loc[i,'d8c'] = df.loc[i,'Adj Close']/df.loc[i-8,'Adj Close']
    df.loc[i,'d9c'] = df.loc[i,'Adj Close']/df.loc[i-9,'Adj Close']
    df.loc[i,'d10c'] = df.loc[i,'Adj Close']/df.loc[i-10,'Adj Close']

prehigh = 0
for i in tqdm(range(len(df))):
    if i<200: 
        df.loc[i,'newhigh']=0
    else:
        if df.loc[i,'Adj Close']>prehigh:
            df.loc[i,'newhigh']=1
        else:
            df.loc[i,'newhigh']=0
    prehigh = max(prehigh,df.loc[i,'Adj Close'])
        

#for i in tqdm(range(len(df)-1)):
#    if df.loc[i+1,'Adj Close']/df.loc[i+1,'dayOpen']>=1.004:
#        df.loc[i,'y']=1
#    elif df.loc[i+1,'dayOpen']/df.loc[i,'Adj Close']<.996:
#        df.loc[i,'y']=-1
#    else:
#        df.loc[i,'y']=0

for i in tqdm(range(len(df)-1)):
    df.loc[i,'y']=df.loc[i+1,'dayOpen']/df.loc[i,'dayClose']

df.loc[len(df)-1,'y']=1

for i in tqdm(range(0,len(df))):
    if i==0:
        df.loc[i,'Open']=1
    else:
        df.loc[i,'Open']=df.loc[i,'dayOpen']/df.loc[i-1,'Adj Close']
    df.loc[i,'High']=df.loc[i,'dayHigh']/df.loc[i,'dayOpen']
    df.loc[i,'Low']=df.loc[i,'dayLow']/df.loc[i,'dayOpen']
    df.loc[i,'Close']=df.loc[i,'dayClose']/df.loc[i,'dayOpen']
#    if i<len(df)-1:
#        df.loc[i,'nextOpen']=df.loc[i+1,'dayOpen']/df.loc[i,'dayClose']
#    else:
#        df.loc[i,'nextOpen']=1
#
#for i in tqdm(range(len(df)-1)):
##    if df.loc[i+1,'dayOpen']/df.loc[i,'dayClose']>=1.004:
##        df.loc[i,'y']=1
#    if df.loc[i+1,'Close']>1.008:
#        df.loc[i,'y']=1
#    else:
#        df.loc[i,'y']=0


df.y.fillna(df.y.median(),inplace=True)
df_c = df.copy()
df.drop(['Date','Time'],axis=1,inplace=True)
df.drop(['Volume','Datetime'],axis=1,inplace=True)
df.drop(['dayOpen','dayHigh','dayLow','dayClose','preHigh','preLow','div'],axis=1,inplace=True)

X = df.drop(['y'],axis=1)
y = df.y

X.ma10.fillna(1,inplace=True)
X.ma20.fillna(1,inplace=True)
X.ma200.fillna(1,inplace=True)
X.vol50.fillna(1,inplace=True)
X.vol20.fillna(1,inplace=True)
X.vol200.fillna(1,inplace=True)
X.d1c.fillna(1,inplace=True)
X.d2c.fillna(1,inplace=True)
X.d3c.fillna(1,inplace=True)
X.d4c.fillna(1,inplace=True)
X.d5c.fillna(1,inplace=True)
X.d6c.fillna(1,inplace=True)
X.d7c.fillna(1,inplace=True)
X.d8c.fillna(1,inplace=True)
X.d9c.fillna(1,inplace=True)
X.d10c.fillna(1,inplace=True)


from sklearn.model_selection import train_test_split
index=[i for i in range(0,len(df))]
X_train, X_test, y_train, y_test,index_train,index_test= train_test_split(X, y,index, test_size=0.2, shuffle=True)
#X_train = X.loc[:3865]
#y_train = y.loc[:3865]
#X_test = X.loc[3865:]
#y_test = X.loc[3865:]
#X_test.reset_index(inplace=True,drop=True)
#y_test.reset_index(inplace=True,drop=True)
#y_trainlist = list(y_train)
#for i in X_train.index:
#    if y_train.loc[i]<0:
#        X_train = X_train.append(X_train.loc[i])
#        y_trainlist.append(y_train.loc[i])
        

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import datasets
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score

regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=1000,verbose=1)
regr.fit(X_train, y_train)
predictions = regr.predict(X_test)
print(explained_variance_score(predictions,y_test))

 
 
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=21)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='explained_variance')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


model = pipelines[0][1]
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(explained_variance_score(predictions,y_test))

model2 = pipelines[5][1]
model2.fit(X_train,y_train)
predictions2 = model2.predict(X_test)
print(explained_variance_score(predictions2,y_test))

model3 = GradientBoostingRegressor(random_state=21,max_depth=2,n_estimators=50)
model3.fit(X_train,y_train)
predictions3 = model3.predict(X_test)
print(explained_variance_score(predictions3,y_test))


#
#import xgboost as xgb
#gbm = xgb.XGBClassifier(max_depth=2, n_estimators=3000, learning_rate=0.01,n_jobs=4,subsample=.8).fit(X_train, y_trainlist,eval_set=[(X_train,y_trainlist),(X_test,y_test)],verbose=True,eval_metric='auc',early_stopping_rounds=1000)
## plot feature importance
#from xgboost import plot_importance
#from matplotlib import pyplot as plt
#plot_importance(gbm)
#plt.show()

#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(max_depth=2,n_estimators=100, random_state=0)

#clf.fit(X, y)
#from sklearn import svm
#clf = svm.SVC()
#clf.fit(X, y)  
y_pred = gbm.predict(X)
from sklearn.metrics import confusion_matrix

confusion_matrix(y,y_pred)

y_test_pred = gbm.predict(X_test)
confusion_matrix(y_test,y_test_pred)

profits=[]
for i in range(len(index_test)):
    if index_test[i]<4831:
        if y_test_pred[i]>=1:
            profits.append((df.loc[index_test[i]+1,'Close']-1)*3+1-1e-4)
#        if y_test_pred[i]<=-1:
#            profits.append((1-df.loc[index_test[i]+1,'Open'])*3+1-1e-4)
print((np.prod(profits)**(1/len(index_test)))**252.0)
print(gmean(profits))
print(len(profits))


profits=[]
for i in range(len(index)):
    if index[i]<4831:
        if y_pred[i]>=1:
            profits.append((df.loc[index[i]+1,'Close']-1)*3+1-1e-4)
#        if y_pred[i]<=-1:
#            profits.append((1-df.loc[index[i]+1,'Open'])*3+1-1e-4)
print((np.prod(profits)**(1/len(index)))**252.0)
print(gmean(profits))
print(len(profits))

#profits=[]
#for i in range(len(index_test)):
#    if index_test[i]<4831:
#        if y_test_pred[i]>=1:
#            profits.append((df.loc[index_test[i]+1,'Close']-1)*3+1-1e-4)
##        elif y_test_pred[i]==-1:
##            profits.append((1-df.loc[index_test[i]+1,'Close'])*3+1-1e-4)
#print((np.prod(profits)**(1/len(index_test)))**252.0)
#print(gmean(profits))
#print(len(profits))
#
#profits=[]
#for i in range(len(index)):
#    if index[i]<4831:
#        if y_pred[i]==1:
#            profits.append((df.loc[index[i]+1,'Close']-1)*3+1-1e-4)
##        elif y_pred[i]==-1:
##            profits.append((1-df.loc[index[i]+1,'Close'])*3+1-1e-4)
#print((np.prod(profits)**(1/len(index_test)))**252.0)
#print(gmean(profits))
#print(len(profits))
