import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.ode import DifferentialEquation
from scipy.integrate import odeint
import arviz as az
import theano
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from statsmodels.stats.weightstats import DescrStatsW


class seirds_simulator:
    def __init__(self, data, delta, gamma, mu, beta0, alpha, beta_t0, betaP0, omega, epsilon, population, bounds = None, Dates = None, th_I = 0):
        self.Dates = Dates
        # Total population, N.
        self.N = population 
        # Initial number of infected and recovered individuals, I0 and R0.
        self.E0, self.I0, self.R0, self.D0 = 0, th_I, 0, 0

        # Everyone else, S0, is susceptible to infection initially.
        self.S0 = self.N - th_I
        
        # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
        
        self.delta = delta
        self.gamma = gamma
        self.mu = mu
        self.epsilon = epsilon
        self.omega = omega 
        self.beta0 = beta0
        self.alpha = alpha
        self.beta_t0 = beta_t0
        self.betaP0 = betaP0
        self.tbeta_flag = 0
        self.data = data

        self.bounds = bounds

        # A grid of time points (in days)
        self.t = np.linspace(0, 365, 366)
    
    def beta_f(self, t):
        """
        calculate beta based on some function
        """
        return self.beta0*(1-self.betaP0/(1 + np.exp(-(self.alpha * (t - self.beta_t0)))))

    # The SIERs model differential equations.
    def deriv(self, y, t):
        
        self.S = y[0]
        self.E = y[1]
        self.I = y[2]
        self.R = y[3]
        self.D = y[4]
        
        self.beta = self.beta_f(t)      
        
        self.dSdt = - self.beta * self.S * (self.I) / self.N #+ omega * self.R
        self.dEdt = self.beta * self.S * (self.I) / self.N  - self.delta * self.E
        self.dIdt = self.delta * self.E - self.gamma * self.I #+ epsilon
        self.dRdt = self.gamma * self.I * (1 - self.mu) #- omega * self.R  ### omega: rate waning immunity
        self.dDdt = self.gamma * self.I * self.mu

        return self.dSdt, self.dEdt, self.dIdt, self.dRdt, self.dDdt

    def integrate(self):
        # Initial conditions vector
        self.y0 = self.S0, self.E0, self.I0, self.R0, self.D0
        theta = [self.E0, self.delta, self.gamma, self.mu, self.beta0, self.alpha, self.beta_t0, self.betaP0]#, self.epsilon, self.omega]
        # Integrate the SEIRs equations over the time grid, t.
        ret = odeint(self.deriv, self.y0, self.t, mxstep=15000)
        self.S, self.E, self.I, self.R, self.D = ret.T
        return ret.T

    def plot_results(self, xlim = [24,200], ylim = [0,30000] ,N = 0):
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(1, 1, figsize=[17, 7])
        self.plot_update(xlim = xlim, ylim = ylim ,N = N)
    
    def plot_update(self, xlim = [24,200], ylim = [0,30000] ,N = 0):

        self.ax.plot(np.pad(self.E, (N, 0), 'constant'), '--', alpha=0.5, lw=2, label='Sim. Exposed')
        self.ax.plot(np.pad(self.I, (N, 0), 'constant'), 'b--', alpha=0.5, lw=2, label='Sim. Infected')
        self.ax.plot(np.pad(self.I + self.R + self.D, (N, 0), 'constant'), 'r--', alpha=0.5, lw=2, label='Sim. casos')
        self.ax.plot(np.pad(self.R, (N, 0), 'constant') , 'g--', alpha=0.5, lw=2, label='Sim. Recovered with immunity')
        self.ax.plot(np.pad(self.D, (N, 0), 'constant'),'--', color = 'white', alpha=0.5, lw=2, label='Sim. Death')
  

        self.ax.plot(np.pad(self.increment(self.I + self.D + self.R), (N, 0), 'constant'), 'g-', alpha=0.5, lw=2, label='Sim. Daily increment')  
        self.ax.plot(self.increment(self.data['cases']), 'g.', alpha=0.5, lw=10, label='real. Daily increment')  


        self.ax.set_xlabel('Time /days')
        self.ax.set_ylabel('# People')

        
        self.ax.scatter(self.data['time'], self.data['deaths'], label = 'real Deaths')
        self.ax.scatter(self.data['time'], self.data['recovered'], label = 'real Recovered')
        self.ax.scatter(self.data['time'], self.data['cases']-self.data['recovered']-self.data['deaths'], label = 'real Infected')
        self.ax.scatter(self.data['time'], self.data['cases'], label = 'real casos')

        
        if not (self.Dates is None):
            for key, value in Dates[self.country].items():
                ind = np.argwhere(self.data['date'] == key)[0][0]
                self.ax.plot([ind,ind],[0,30000], value[1], label = self.country+'-'+ value[0])
            
        self.ax.legend()
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

    def train(self):
        print(self.bounds)
        self.tbeta_flag = 0
        self.t = np.linspace(0, len(self.data['time']), len(self.data['time'])) 
        self.optimal = minimize(self.loss, 
            #method = 'Nelder-Mead',
            method = 'TNC',
            #method = 'L-BFGS-B',
            bounds = self.bounds,
            x0 = [self.E0, self.delta, self.gamma, self.mu, self.beta0, self.alpha, self.beta_t0, self.betaP0],
            options={'xtol': 1e-6, 'disp': True, 'maxiter': 100000})
            #options={'disp': True, 'maxcor': 100, 'ftol': 2.220446049250313e-12, 
            #         'gtol': 1e-09, 'eps': 1e-11, 'maxfun': 500000, 'maxiter': 500000, 'iprint': -1, 'maxls': 1000})#, self.alpha, self.beta_t0, self.epsilon, self.omega])
        
        print(self.optimal)
        
        print()
        print('δ: Days for symptoms to appear: '+ str(1/self.delta))
        print('1/γ: Days to recovery: '+ str(1/self.gamma))
        print('μ: proportion of cases who die: '+ str(self.mu))
        print('E0: initial exposed: '+ str(self.E0))
        print('β0: rate of infection: '+ str(self.beta0))
        print('Days before Lockdown: '+ str(self.beta_t0))
        print('Lockdown strength: '+ str(self.betaP0))
        print('Lockdown growth: '+ str(self.alpha))
    
        self.E0, self.delta, self.gamma, self.mu, self.beta0, self.alpha, self.beta_t0, self.betaP0  = self.optimal.x #, self.alpha, self.beta_t0 , self.epsilon, self.omega  = self.optimal.x
        self.t = np.linspace(0, 200, 200)
        self.prediction = self.integrate()


    def loss(self, point):
        self.E0, self.delta, self.gamma, self.mu, self.beta0, self.alpha, self.beta_t0, self.betaP0  = point

        self.integrate()
        
        return self.cal_loss()

    def cal_loss(self):

        LC = self.inv_SNR(self.I + self.R + self.D, self.data['cases'].values)**2
        LR = self.inv_SNR(self.R, self.data['recovered'].values)**2
        LD = self.inv_SNR(self.D, self.data['deaths'].values)**2

        LCM = self.inv_SNR(self.I + self.R + self.D, self.data['cases'].values, flag_ml = False)**2
        LRM = self.inv_SNR(self.R, self.data['recovered'].values, flag_ml = False)**2
        LDM = self.inv_SNR(self.D, self.data['deaths'].values, flag_ml = False)**2

        return np.sqrt(LC**2 + LR**2 + LD**2 + LCM**2 + LRM**2 + LDM**2)

    def increment(self, B):
        B = np.array(np.roll(B,-1) - B)
        B[-1] = 0
        return B

    def inv_SNR(self, A, B, flag_ml = True):
        weight = self.t
        if flag_ml:
            weight = self.increment(A)
            A = self.increment(A)
            B = self.increment(B)

        return np.sqrt(np.sum(weight**2*(A-B)**2))/np.sqrt(np.sum(weight**2*A**2))


   