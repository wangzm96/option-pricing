# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:02:16 2019

@author: wangz
"""

import math
import numpy as np
from scipy import stats
#np.random.seed(42)
import warnings
warnings.filterwarnings("ignore")
import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import pandas as pd

S0 = 40
K = 40
T = 1.0
r = 0.06
sigma = 0.2

I = 400000
M = 250
dt = T/M
df = math.exp(-r*dt)

#S = S0*np.exp(np.cumsum((r-0.5*sigma**2)*dt+sigma*math.sqrt(dt)*np.random.standard_normal((M,I)),axis=0))

randn = np.random.randn(M,I)
randnum = np.exp((r-0.5*sigma*sigma)*dt+sigma*np.sqrt(dt)*randn)
randpaths = np.cumprod(randnum,axis=0)
S = randpaths*S0

# Inner value
hh = np.maximum(K-S,0)
payoff = hh[-1]*np.exp(-r*T)
print("European put option value(simulation) %5.4f"%(np.mean(payoff)))
print(np.std(payoff)/np.sqrt(I))



d1 = (np.log(S0 /K) + (r + 0.5 * sigma**2) * T )/(sigma * np.sqrt(T))
d2 = (np.log(S0 /K) + (r - 0.5 * sigma**2) * T )/(sigma * np.sqrt(T))
value =  K * np.exp(-r * T) * (1-stats.norm.cdf(d2, 0, 1)) - (S0 * (1-stats.norm.cdf(d1, 0, 1) ))
print("European put option value（formula) %5.4f"%value)
# PV Vector
V = hh[-1]  # 最后一天模拟的价格

coef = pd.DataFrame(columns=[1,2,3,4,5])

def polynomial(S,V,t,idx,df):
    '''
    rg = np.polyfit(S[t,idx],V[idx]*df,2)
    C = np.polyval(rg,S[t,idx])
    '''
    X  = np.matrix(np.vstack([S[t,idx],S[t,idx]**2,S[t,idx]**3,S[t,idx]**4,S[t,idx]**5]).T)
    y = np.matrix(V[idx].reshape(-1,1))
    model = LinearRegression().fit(X, y)
    #coef2.loc[len(coef2)] = model.coef_[0]
    C = model.predict(X)
    
    
    return C.flatten()

def linear(S,v,t,idx,df):
    l0 = np.exp(-S[t,idx]/2)
    l1 = np.exp(-S[t,idx]/2)*(1-S[t,idx])
    l2 = np.exp(-S[t,idx]/2)*(1-2*S[t,idx]+S[t,idx]**2/2)
    
    X  = np.matrix(np.vstack([l0,l1,l2]).T)
    y = np.matrix(V[idx].reshape(-1,1))
    C = LinearRegression().fit(X, y).predict(X)
   
    return C.flatten()

def ridge(S,v,t,idx,df):
    X  = np.matrix(np.vstack([S[t,idx],S[t,idx]**2,S[t,idx]**3,S[t,idx]**4,S[t,idx]**5]).T)
    y = np.matrix(V[idx].reshape(-1,1))
    model = Ridge(alpha=0.8).fit(X, y)
    #print(model.coef_[0])
    coef.loc[len(coef)] = model.coef_[0]
    C = model.predict(X)
    
    
    return C.flatten()

    
def elasticNet(S,v,t,idx,df):
    X  = np.matrix(np.vstack([S[t,idx],S[t,idx]**2,S[t,idx]**3,S[t,idx]**4,S[t,idx]**5]).T)
    y = np.matrix(V[idx].reshape(-1,1))
    C = Lasso( alpha=0.1).fit(X, y).predict(X)
    
    
    return C.flatten()

 
# American Option Valuation by Backwards Induction
for t in range(M-2,-1,-1): #从倒数第二天开始，到第一天
    idx = (S[t]<K)
    if len(V[idx])>0:
        
        #rg = np.polyfit(S[t,idx],V[idx]*df,5)  # 第t天的股价与继续持有的payoff拟合
        #C = np.polyval(rg,S[t,idx])  # continuation values     # 用第t天的股价猜payoff
        C = polynomial(S,V,t,idx,df)
        V[idx]  = np.where(hh[t,idx] > C,hh[t,idx],V[idx]*df)  # 如果第t天的行权payoff大于拟合的payoff，行权
        V[~idx] = V[~idx]*df
    else:
        V = V*df
    
    # exercise decision

V0 = df*np.sum(V)/I
print(np.std(df*V)/np.sqrt(I))

print("American put option value %5.4f"%V0)


