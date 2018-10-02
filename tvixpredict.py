# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 20:58:56 2018

@author: ximing
"""
import pandas as pd
from scipy.stats import gmean
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


vixind = pd.read_csv('VIXind4vix.csv')
spy = pd.read_csv('SPY4vix.csv')
qqq = pd.read_csv('QQQ4vix.csv')
tvix = pd.read_csv('tvix.csv')
dia = pd.read_csv('DIA4vix.csv')

def preprocess(df):
    for i in range(len(df)-1):
        df.loc[i,'nextChange']=df.loc[i+1,'Adj Close']/df.loc[i,'Adj Close']-1
        df.loc[i,'range']=df.loc[i,'High']/df.loc[i,'Low']-1
        df.loc[i,'orange']=df.loc[i,'High']/df.loc[i,'Open']-1
    for i in range(len(df)-1):
        if i<20:
            df.loc[i,'ma20']=0
        else:
            df.loc[i,'ma20'] = df.loc[i,'Adj Close']/np.mean(df.loc[i-20:i-1,'Adj Close'])-1
    return df

def processVol(df):
    for i in range(len(df)-1):
        if i<20:
            df.loc[i,'vol20']=0
        else:
            df.loc[i,'vol20'] = df.loc[i,'Volume']/np.mean(df.loc[i-20:i-1,'Volume'])-1
    return df

vixind = preprocess(vixind)
spy = preprocess(spy)
qqq = preprocess(qqq)
tvix = preprocess(tvix)
dia = preprocess(dia)
spy = processVol(spy)
qqq = processVol(qqq)
dia = processVol(dia)


vixp = np.array(list(vixind.Close))
vixc = np.array(list(vixind.nextChange))
vixr = np.array(list(vixind.range))
spyc = np.array(spy.nextChange)
spyr = np.array(list(spy.range))
qqqc = np.array(qqq.nextChange)
qqqr = np.array(list(qqq.range))
tvixc = np.array(tvix.nextChange)
diac = np.array(dia.nextChange)
diar = np.array(list(dia.range))
vixm20 = np.array(vixind.ma20)
spym20 = np.array(spy.ma20)
qqqm20 = np.array(qqq.ma20)
diam20 = np.array(dia.ma20)
spyv20 = np.array(spy.vol20)
qqqv20 = np.array(qqq.vol20)
diav20 = np.array(dia.vol20)
vixo = np.array(vixind.orange)
spyo = np.array(spy.orange)
qqqo = np.array(qqq.orange)
diao = np.array(dia.orange)


X=np.stack((vixp,vixc,spyc,vixr,spyr,diac,diar,vixm20,spym20,diam20,spyv20,diav20,vixo,spyo,diao),axis=1)

#X=np.stack((vixp,vixc,spyc,qqqc,vixr,spyr,qqqr,diac,diar,vixm20,spym20,qqqm20,diam20,spyv20,qqqv20,diav20,vixo,spyo,qqqo,diao),axis=1)
y=np.array(tvixc)
X=X[:-1,:]
y = y[:-1]


from sklearn.model_selection import train_test_split
index=[i for i in range(0,len(X))]
X_train, X_test, y_train, y_test,index_train,index_test= train_test_split(X, y,index, test_size=0.2, shuffle=False)

import xgboost as xgb
from sklearn.metrics import explained_variance_score

xgbr = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, gamma=0, subsample=0.8,
                           colsample_bytree=1, max_depth=2)
xgbr.fit(X_train,y_train,verbose=True)
predictions = xgbr.predict(X_test)
print(explained_variance_score(predictions,y_test))


#gbm = xgb.XGBClassifier(max_depth=2, n_estimators=3000, learning_rate=0.01,n_jobs=4,subsample=.8).fit(X_train, y_train,eval_set=[(X_train,y_train),(X_test,y_test)],verbose=True,eval_metric='auc',early_stopping_rounds=1000)
# plot feature importance
from xgboost import plot_importance
from matplotlib import pyplot as plt
plot_importance(xgbr)
plt.show()

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

from sklearn.model_selection import GridSearchCV

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=np.array([10,20,30,40,50,100,200,300,400]),max_depth=np.array([2,3,4]))
model = GradientBoostingRegressor(random_state=21)
kfold = KFold(n_splits=10, random_state=21)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='explained_variance', cv=kfold)
grid_result = grid.fit(rescaledX, y_train)

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


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


predictions = model.predict(Xt)
y2=predictions+1-5e-4
tvix.loc[0,'pred']=tvix.loc[0,'Adj Close']
for i in range(1,len(tvix)):
    tvix.loc[i,'pred']=tvix.loc[i-1,'pred']*y2[i-1]
    if tvix.loc[i,'pred']!=tvix.loc[i,'Adj Close']:
#        print(i)
    


vixindmq = pd.read_csv('vixindmq.csv')
spymq = pd.read_csv('spymq.csv')
qqqall = pd.read_csv('qqqall.csv')
diamq = pd.read_csv('diamq.csv')

vixindmq = preprocess(vixindmq)
spymq = preprocess(spymq)
qqqall = preprocess(qqqall)
diamq = preprocess(diamq)
spymq = processVol(spymq)
qqqall = processVol(qqqall)
diamq = processVol(diamq)


vixp = np.array(list(vixindmq.Close))
vixc = np.array(list(vixindmq.nextChange))
vixr = np.array(list(vixindmq.range))
spyc = np.array(spymq.nextChange)
spyr = np.array(list(spymq.range))
qqqc = np.array(qqqall.nextChange)
qqqr = np.array(list(qqqall.range))
diac = np.array(diamq.nextChange)
diar = np.array(list(diamq.range))
vixm20 = np.array(vixindmq.ma20)
spym20 = np.array(spymq.ma20)
qqqm20 = np.array(qqqall.ma20)
diam20 = np.array(diamq.ma20)
spyv20 = np.array(spymq.vol20)
qqqv20 = np.array(qqqall.vol20)
diav20 = np.array(diamq.vol20)
vixo = np.array(vixindmq.orange)
spyo = np.array(spymq.orange)
qqqo = np.array(qqqall.orange)
diao = np.array(diamq.orange)



X=np.stack((vixp,vixc,spyc,vixr,spyr,diac,diar,vixm20,spym20,diam20,spyv20,diav20,vixo,spyo,diao),axis=1)
#y=np.array(tvixc)
X=X[:-1,:]
#y = y[:-1]

predictions = model.predict(X)
y2=predictions+1-4e-4
pred=[1]
for i in range(len(y2)):
    pred.append(pred[-1]*y2[i-1])
#    if df.loc[i,'pred']!=tvix.loc[i,'Adj Close']:

premin=1
premax=1
maxrange=[]
for p in pred:
    if p>premax:
        premax=p
    if p<premin:
        premin=p
    maxrange.append(p/premin)

qprice = list(qqqall['Adj Close'])
cash=1.3-0.3
stock=0.3/pred[0]
longasset = 0.3/qprice[0]
pin = pred[0]
navs=[]
for i in range(len(pred)):
    p=pred[i]
    if p<.9*pin:
        nav = cash-stock*p+longasset*qprice[i]
        tmp = nav*0.3/p
        cash = cash+(tmp-stock)*p
        stock=tmp
        pin=p
    elif p>2*pin:
        nav = cash-stock*p+longasset*qprice[i]
        ns = nav*0.3
        cash -= stock*p-ns
        stock = ns/p
        pin=p
    navs.append(cash-stock*p+longasset*qprice[i])
    
    
    
    
    