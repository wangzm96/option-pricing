# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 13:29:19 2019

@author: wangz
"""

import numpy as np
import time
import pandas as pd

def makeGrid(r,sigma,T,K,xmin,xmax,Nx,Nt):
    
    q = (2*r)/(sigma**2)
    tau = np.linspace(0,sigma**2*T/2,Nt+1).reshape(1,-1)
    dt = sigma**2*T/2/Nt
    
    x = np.linspace(xmin,xmax,Nx+1).reshape(-1,1)
    dx = (xmax-xmin)/Nx
    lamb = dt/(dx**2)
    
    Initial = np.maximum(np.exp((q-1)/2*x)-np.exp((q+1)/2*x),0)
    g = np.dot(np.maximum(np.exp((q-1)*x/2)-np.exp((q+1)*x/2),0),np.exp((q+1)**2*tau/4))
   
    
    return Initial,g,x,q,lamb

def PSOR(lamb,Nx,v,gn,RHS):
    error = 1
    #numit = 0
    vnew = np.zeros(Nx+1)
    while error>1e-10:# and numit<4000:
        for i in range(1,Nx):
            vtemp = (RHS[i-1]+lamb/2*(vnew[i-1]+v[i+1]))/(1+lamb)
            vnew[i] = np.maximum(gn[i],vtemp)
        error = np.linalg.norm(v-vnew,np.inf)
        v = vnew
    return v.reshape(-1,1)

def CrankNicolsonAmericanPut(sigma,T,K,x,q,lamb,Nx,Nt,Initial,g):
    
    
    w = Initial
    b = (1-lamb)*np.eye(Nx-1) + lamb/2*np.diag(np.ones(Nx-2),1)+lamb/2*np.diag(np.ones(Nx-2),-1)
    freeboundary = np.zeros(Nt)
    
    for n in range(Nt):
        d = np.hstack([[g[0,n]+g[0,n+1]],np.zeros(Nx-3),[g[-1,n]+g[-1,n+1]]])
        RHS = np.dot(b,w[1:-1])+(d*lamb/2).reshape(-1,1)
        gn = g[:,n]
        w = PSOR(lamb,Nx,w,gn,RHS)
        freeboundary[n] = x[np.where(w>g[:,n+1])[0][0]]
    
    
    #print(w.shape)    
    w = np.maximum(w,g[:,n+1].reshape(-1,1))
    #print((np.exp(-(q-1)/2*x-(q+1)**2/4*sigma**2*T/2)).shape)
    #print(w.shape)
    V = np.exp(-(q-1)/2*x-(q+1)**2/4*sigma**2*T/2)*w*K
    S = K*np.exp(x)
    Sf = K*np.exp(freeboundary)
    
    return V,S,Sf

if __name__ == "__main__":
    
    S0 = 40
    K = 40
    T = 1.0
    r = 0.06
    sigma = 0.2
    xmin = -10
    xmax = 10
    Nx = 1000
    Nt = 1000
    
    start = time.time()
    Initial,g,x,q,lamb = makeGrid(r,sigma,T,K,xmin,xmax,Nx,Nt)
    
    V,S,Sf = CrankNicolsonAmericanPut(sigma,T,K,x,q,lamb,Nx,Nt,Initial,g)
    end = time.time()
    print("耗时:",np.round(end-start,2),"秒")
    result = pd.DataFrame(data={"V":list(V),"S":list(S)},index=range(len(V)))
              