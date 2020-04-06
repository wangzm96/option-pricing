# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:08:44 2020

@author: wangz
"""


import numpy as np
import pandas as pd
import time
from datetime import datetime


class simulator:    
    def __init__(self,s0,sigma,coc,T,trigger,coupon,participation_rate,
                    start_date,valuation_date,trigger_valuation_date, paths,discount):
        
        self.stime = time.time()
        self.trade_days = pd.read_csv(".\\trade_day.csv") # 交易日表
        self.s0 = s0  # 今天的指数
        self.sigma = sigma 
        self.coc = coc
        self.paths = paths # 模拟路径次数
        self.start_date = start_date
        self.start_date_idx = self.trade_days[self.trade_days["date"]>=start_date].iloc[0:1,:].index[0]
        self.end_date = trigger_valuation_date[-1]  # 合约最后一天
     
        self.trigger = trigger*s0
        self.participation_rate = participation_rate
        self.coupon = coupon
       
        self.T = T # 期限
        self.dt = 1/250
        
        self.discount = discount
        # 计算关键日期的索引号
        self.trigger_valuation_date = trigger_valuation_date
        self.valuation_date_idx = []
        for d in valuation_date:
            self.valuation_date_idx.append(self.dates_idx(d))
        
        self.trigger_valuation_date_idx = []
        for d in trigger_valuation_date:
            self.trigger_valuation_date_idx.append(self.valuation_date_idx.index(self.dates_idx(d)))
          
        date_diff = self.valuation_date_idx[:-1].copy()
        date_diff.insert(0,0)
        self.date_diff = np.array(self.valuation_date_idx)-np.array(date_diff)
        
    def calculate_price(self):
        self.generate_paths()
        self.value,self.std,self.simulation_result,etime = self.payoff()
        
        print("价格:",np.round(self.value,4),"标准差:",np.round(self.std,4))
        print("耗时:",np.round(etime-self.stime,2),"秒")
    
    def calculate_greeks(self):
        ds = 0.01*self.s0
        dsigma = 0.01
        dr = 0.001
        # delta
        h  = self.date_diff[0]*self.dt
        lrdelta = self.randn[:,0]/(self.sigma*self.s0*np.sqrt(h))
        delta = np.mean(lrdelta*(self.simulation_result-self.s0))
        deltastd = np.std(self.pv(lrdelta*(self.simulation_result-self.s0)/np.sqrt(self.paths),self.trigger_valuation_date[-1]))
        delta = self.pv(delta,self.trigger_valuation_date[-1])
        print("Delta LR Method Value:",np.round(delta,4),"Stdev:",np.round(deltastd,4))
        
    
        new_paths = self.generate_paths("delta",self.s0+ds)
        v_plus,_,payoff_plus,_ = self.payoff(new_paths)
        #print(v_plus)
        new_paths = self.generate_paths("delta",self.s0-ds)
        v_minus,_,payoff_minus,_ = self.payoff(new_paths)
        #print(v_minus)
        self.delta = (v_plus-v_minus)/(2*ds)
        deltastd = np.std((payoff_plus-payoff_minus)/(2*ds))/np.sqrt(self.paths)
        print("Delta FD Method Value:",np.round(self.delta,4),"Stdev:",np.round(deltastd,4))
       
        lrgamma = (self.randn[:,0]**2-self.randn[:,0]*self.sigma*np.sqrt(h)-1)/(self.s0**2*self.sigma**2*h)
        gamma = np.mean((self.simulation_result-self.s0)*lrgamma)
        gamma = self.pv(gamma,self.trigger_valuation_date[-1])
        gammastd = np.std(self.pv((self.simulation_result-self.s0)*lrgamma/np.sqrt(self.paths),self.trigger_valuation_date[-1]))
        print("Gamma LR Method Value:",np.round(gamma,4),"Stdev:",np.round(gammastd,4))
        
        self.gamma = (v_plus+v_minus-2*self.value)/(ds*ds)
        gammastd = np.std((payoff_plus+payoff_minus-2*self.simulation_result)/(ds*ds))/np.sqrt(self.paths)
        print("Gamma FD Method Value:",np.round(self.gamma,4),"Stdev:",np.round(gammastd,4))
        
        lrvega = (s.randn**2).sum(axis=1)/self.sigma-np.sqrt(h)*self.randn.sum(axis=1)-len(self.valuation_date_idx)/self.sigma
        vega = np.mean((self.simulation_result-self.s0)*lrvega)
        vega =self.pv(vega,self.trigger_valuation_date[-1])
        vegastd = np.std(self.pv((self.simulation_result-self.s0)*lrvega,self.trigger_valuation_date[-1]))/np.sqrt(self.paths)
        print("Vega LR Method Value:",np.round(vega,4),"Stdev:",np.round(vegastd,4))
        
        new_paths = self.generate_paths("vega",sigma=self.sigma+dsigma)
        v_plus,_,payoff_plus,_ = self.payoff(new_paths)
        new_paths = self.generate_paths("vega",sigma=self.sigma-dsigma)
        v_minus,_,payoff_minus,_ = self.payoff(new_paths)
        self.vega = (v_plus-v_minus)/(2*dsigma)
        vegastd = np.std((payoff_plus-payoff_minus)/(2*dsigma))/np.sqrt(self.paths)
        print("Vega FD Method Value:",np.round(self.vega,4),"Stdev:",np.round(vegastd,4))
       
        if self.discount: time = self.T
        else: time = 0
        lrRho = -time + np.sum(np.sqrt(self.date_diff*self.dt)*self.randn/self.sigma,axis=1)
        rho = np.mean((self.simulation_result-self.s0)*lrRho)
        rho = self.pv(rho,self.trigger_valuation_date[-1])
        rhostd = np.std(self.pv((self.simulation_result-self.s0)*lrRho/np.sqrt(self.paths),self.trigger_valuation_date[-1]))
        print("Rho LR Method Value:",np.round(rho,4),"Stdev:",np.round(rhostd,4))
        
       
        new_paths = self.generate_paths("rho",r=self.coc+dr)
        v_plus,_,payoff_plus,_ = self.payoff(new_paths)
        new_paths = self.generate_paths("rho",r=self.coc-dr)
        v_minus,_,payoff_minus,_ = self.payoff(new_paths)
        self.rho = (v_plus-v_minus)/(2*dr)
        rhostd = np.std((payoff_plus-payoff_minus)/(2*dr))/np.sqrt(self.paths)
        print("Rho FD Method Value:",np.round(self.rho,4),"Stdev:",np.round(rhostd,4))
        
        
        
        
    def payoff(self,sim_paths=[]):
        if len(sim_paths) == 0 :sim_paths = self.sim_paths
        avg_obs = sim_paths.mean(axis=1) # 所有估值日之表现水平的算术平均数
        trigger = sim_paths[:,self.trigger_valuation_date_idx] >= self.trigger # 触发估值日是否高于触发水平
        
        payoff = np.zeros(self.paths)
        
        signal = (trigger[:,0]==True)
        # 触发估值日触发的次数 payoff是固定的
        for i in range(1,len(self.trigger_valuation_date)):
            signal = signal|(trigger[:,i]==True)
        trigger_1 = sim_paths[signal,:].shape[0]
        payoff_1 = self.pv((1+self.coupon)*self.s0,self.trigger_valuation_date[-1])
        payoff_1 = np.array([payoff_1]).repeat(trigger_1)
        
        payoff[signal] = payoff_1
        
        # 未触发的情况 payoff不固定
        signal = (trigger[:,0]==False)
        for i in range(1,len(self.trigger_valuation_date)):
            signal = signal&(trigger[:,i]==False)
        no_trigger_avgobs = np.maximum(avg_obs[signal]-self.s0,0)/self.s0
        payoff_2 = self.pv((1+no_trigger_avgobs*self.participation_rate)*self.s0,self.trigger_valuation_date[-1])
        
        # 合并payoff
        #payoff = np.hstack((payoff_1,payoff_2))
        payoff[signal] = payoff_2
        
        
        value = np.mean(payoff)
        std  = np.std(payoff)/np.sqrt(self.paths)
        etime = time.time()
        
        return value,std,payoff,etime
    
    def generate_paths(self,gtype=None,s0=None,sigma=None,r=None):
        
        if gtype == None:
            self.randn = np.random.randn(self.paths,len(self.valuation_date_idx))
            self.randnum = np.exp((self.coc-0.5*self.sigma**2)*(self.date_diff*self.dt)
                +self.sigma*np.sqrt(self.date_diff*self.dt)*self.randn)
            self.randnum = np.cumprod(self.randnum,axis=1)
            self.sim_paths = self.randnum*self.s0       
        elif gtype == "vega":
            randnum = np.exp((self.coc-0.5*sigma**2)*(self.date_diff*self.dt)
                +sigma*np.sqrt(self.date_diff*self.dt)*self.randn)
            randnum = np.cumprod(randnum,axis=1)
            sim_paths = randnum*self.s0  
            return sim_paths
        
        elif gtype == "delta":
            sim_paths = self.randnum*s0  
            return sim_paths
            
        elif gtype == "rho":
            randnum = np.exp((r-0.5*self.sigma**2)*(self.date_diff*self.dt)
                +self.sigma*np.sqrt(self.date_diff*self.dt)*self.randn)
            randnum = np.cumprod(randnum,axis=1)
            sim_paths = randnum*self.s0  
            return sim_paths
            
    def dates_idx(self,date):
        tmp = self.trade_days[self.trade_days["date"]>=date].iloc[0:1,:]
        return tmp.index[0]-self.start_date_idx
    
    
    def pv(self,fv,d):
        if self.discount:
        # 这里用日历日计算
            s = datetime.strptime(self.start_date,"%Y-%m-%d")
            e = datetime.strptime(d,"%Y-%m-%d")
            d = (e-s).days
        
            discount_factor = np.exp(-self.coc*d/365)
            return discount_factor*fv
        else:
            return fv
    
if __name__ == "__main__":
    s0 = 100 #
    sigma = 0.058#
    coc = 0.003
    discount = False
    T = 2
    paths = 10000000 # 模拟次数
    trigger = 1.12 # 触发水平
    coupon = 0.12 # 触发票息
    participation_rate = 1.8 # 参与率
    

    start_date = "2018-12-27" # 合约开始日期
    # 估值日
    valuation_date = ["2019-03-27","2019-06-27","2019-09-27","2019-12-27","2020-03-27","2020-06-27","2020-09-27","2020-12-27"]
    # 触发估值日
    trigger_valuation_date = ["2019-12-27","2020-12-27"]
    
    s = simulator(s0,sigma,coc,T,trigger,coupon,participation_rate,start_date,valuation_date,trigger_valuation_date,paths,discount)
    s.calculate_price()
    s.calculate_greeks()