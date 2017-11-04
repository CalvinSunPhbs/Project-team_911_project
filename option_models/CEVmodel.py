# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:22:07 2017

@author: 俞方舟
"""

from scipy.special import gamma
from scipy.stats import norm
import numpy as np
from numpy import random


# function g whcih would be used calculating Q#
def g(i,y):
    return  y**(i-1)*np.exp(-y)/gamma(i)

# function Q which would be used calculating option price#
def Q(z, v, k):

#    n = 0
    z = 0.5*z
    v = 0.5*v
    k = 0.5*k
#    gA = g(n+v,z)
#    gB = g(n,k)
#    Sg = gB
#    R = 1-gA*Sg
#    while (R-gA*Sg) > 10**-3:
#        gA = g(n+v,z)
#        gB = g(n,k)
#        Sg = Sg + gB
#        R = R - gA*Sg
#        n = n+1
#    return 1-R
    h = 1-(2/3)*(v+k)*(v+3*k)*(v+2*k)**(-2)
    P = (v+2*k)/(v+k)**2
    m = (h-1)*(1-3*h)
    return 1-norm.cdf((1-h*P*(1-h+0.5*(2-h)*m*P)-(z/v+k)**h)/(h*np.sqrt(2*P*(1+m*P))))
    



#use the analytic way to calculate the option price of CEV model#
def Price_CEV(div_rate, int_rate, sigma, beta, t_exp, spot, strike, cp_sign):
    
    k_star = 2*(int_rate-div_rate)/(sigma**2*(2-beta)*(np.exp((int_rate-div_rate)*(2-beta)*t_exp)-1))
    y = k_star * strike**(2-beta)
    x = k_star * spot**(2-beta)*np.exp((int_rate-div_rate)*(2-beta)*t_exp)
    if beta > 2:
        priceofoption = spot*np.exp(-div_rate*t_exp)*Q(2*y, 2/(beta-2), 2*x)-strike*np.exp(-int_rate*t_exp)*Q(2*x, 2/(beta-2), 2*y) 
    else:
        priceofoption = spot*np.exp(-div_rate*t_exp)*Q(2*y, 2+2/(2-beta), 2*x)-strike*np.exp(-int_rate*t_exp)*Q(2*x, 2/(2-beta), 2*y) 
    if cp_sign == 0:
        priceofoption = priceofoption - spot*np.exp(-div_rate*t_exp) + strike*np.exp(-int_rate*t_exp)
    return priceofoption

# use the Monte-Carlo methon to calculate the option price of CEV model#
def Price_MC(div_rate, int_rate, sigma, beta, t_exp, spot, strike, cp_sign, M=1000, N=1000):
    
    time_gap = t_exp/N
    price_series = np.zeros(N)
    price_series[0] = spot
    priceofoption = np.zeros(M)
    
    for p in range(M):
        
        for q in range(N-1):

            price_series[q+1] = price_series[q]+(int_rate-div_rate)*price_series[q]*time_gap+sigma*price_series[q]**(beta/2)*random.random()
        
        if cp_sign == 1:
            priceofoption[p] = np.exp(div_rate-int_rate)*max(price_series[N-1]-strike, 0)
        else:
            priceofoption[p] = np.exp(div_rate-int_rate)*max(strike-price_series[N-1],0)
            
        return np.mean(priceofoption)




class ModelCEV:
    
    def __init__(self, div_rate, int_rate, sigma, beta, t_exp, spot, strike, cp_sign):
         
        self.div_rate = div_rate
        self.int_rate = int_rate
        self.sigma = sigma
        self.beta = beta
        self.t_exp = t_exp
        self.spot = spot
        self.strike = strike
        self.cp_sign = cp_sign

    def price_analytic(self):
        return Price_CEV(self.div_rate,self.int_rate,self.sigma,self.beta,self.t_exp,self.spot,self.strike,self.cp_sign)
    
    def price_mc(self, N=1000):
        return Price_MC(self.div_rate,self.int_rate,self.sigma,self.beta,self.t_exp,self.spot,self.strike,self.cp_sign)
        

    
 
    

    