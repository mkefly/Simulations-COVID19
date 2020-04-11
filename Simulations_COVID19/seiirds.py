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
    def __init__(self, data, delta, gammaR, gammaD, mu, beta0, alpha, beta_t0, omega, epsilon, population, bounds = None, Dates = None, th_I = 0):
        self.Dates = Dates
        # Total population, N.
        self.N = population 
        # Initial number of infected and recovered individuals, I0 and R0.
        self.E0, self.IR0, self.ID0, self.R0, self.D0 = 0, th_I, 0, 0, 0

        # Everyone else, S0, is susceptible to infection initially.
        self.S0 = self.N - self.E0 - self.IR0 - self.ID0 - self.R0 - self.D0
        
        # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
        
        self.delta = delta
        self.gammaR = gammaR
        self.gammaD = gammaD
        self.mu = mu
        self.epsilon = epsilon
        self.omega = omega 
        self.beta0 = beta0
        self.alpha = alpha
        self.beta_t0 = beta_t0
        self.tbeta_flag = 0
        self.data = data

        self.bounds = bounds

        # A grid of time points (in days)
        self.t = np.linspace(0, 365, 366)

    
    def deriv_theta(self, y, t, theta):
        delta = (theta[0])
        gammaR = (theta[1])
        gammaD = (theta[2])
        beta0 = (theta[3])
        alpha = (theta[4])
        mu = (theta[5])
        
        beta_t0 = self.beta_t0
        epsilon = 0
        omega = 0
        
        self.deriv(y, t, delta, gammaR, gammaD, mu, beta0, alpha, beta_t0, epsilon, omega)
        y10 = (self.dSdt, self.dEdt, self.dIRdt, self.dIDdt, self.dRdt, self.dDdt)

        return y10    
    
    def beta_f(self, t):
        """
        calculate beta based on some function
        """
        return self.beta0*(1-1/(1 + np.exp(-(self.alpha * (t - self.beta_t0)))))

    # The SIERs model differential equations.
    def deriv(self, y, t, delta, gammaR, gammaD, mu, beta0, alpha, beta_t0, epsilon, omega):
        
        self.S = y[0]
        self.E = y[1]
        self.IR = y[2]
        self.ID = y[3]
        self.R = y[4]
        self.D = y[5]
        
        #beta = beta0
        beta = self.beta_f(t)      
        
        self.dSdt = - beta * self.S * (self.IR + self.ID) / self.N #+ omega * self.R
        self.dEdt = beta * self.S * (self.IR + self.ID) / self.N  - delta * self.E
        self.dIRdt = delta * (1 - self.mu) * self.E - gammaR * self.IR #+ epsilon
        self.dIDdt = delta * self.mu * self.E - gammaD * self.ID #+ epsilon
        self.dRdt = gammaR * self.IR #- omega * self.R  ### omega: rate waning immunity
        self.dDdt = gammaD * self.ID

        #self.N -= self.D0 #### beta_f considered the evolution 
        return self.dSdt, self.dEdt, self.dIRdt, self.dIDdt, self.dRdt, self.dDdt

    def integrate(self):
        # Initial conditions vector
        self.y0 = self.S0, self.E0, self.IR0, self.ID0, self.R0, self.D0
        theta = [self.E0, self.delta, self.gammaR, self.gammaD, self.mu, self.beta0, self.alpha, self.beta_t0]#, self.epsilon, self.omega]
        # Integrate the SEIRs equations over the time grid, t.
        ret = odeint(self.deriv_theta, self.y0, self.t, args=(theta,), mxstep=15000)
        self.S, self.E, self.IR, self.ID, self.R, self.D = ret.T
        return ret.T

    def plot_results(self, xlim = [24,200], ylim = [0,20000] ,N = 0):
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(1, 1, figsize=[17, 7])
        self.plot_update(xlim = xlim, ylim = ylim ,N = N)
    
    def plot_update(self, xlim = [24,200], ylim = [0,20000] ,N = 0):
        # Plot the data on three separate curves for S(t), I(t) and R(t)
        # Create inset of width 30% and height 40% of the parent axes' bounding box
        # at the lower left corner (loc=3)
        #self.axins2 = inset_axes(self.ax, width="30%", height="40%", loc=2)
        
        self.ax.plot(np.pad(self.E, (N, 0), 'constant'), '--', alpha=0.5, lw=2, label='Sim. Exposed')
        self.ax.plot(np.pad(self.ID + self.IR, (N, 0), 'constant'), 'r--', alpha=0.5, lw=2, label='Sim. Infected')
        self.ax.plot(np.pad(self.R, (N, 0), 'constant') , 'g--', alpha=0.5, lw=2, label='Sim. Recovered with immunity')
        self.ax.plot(np.pad(self.D, (N, 0), 'constant'),'--', color = 'white', alpha=0.5, lw=2, label='Sim. Death')
  
        def increment(B):
            B = np.roll(B,-1) - B
            B = np.array(B)
            B[-1] = 0
            return B

        self.ax.plot(np.pad(increment(self.ID + self.IR + self.D + self.R), (N, 0), 'constant'), 'g-', alpha=0.5, lw=2, label='Sim. Daily increment')  
        self.ax.plot(increment(self.data['cases']), 'g.', alpha=0.5, lw=10, label='real. Daily increment')  


        self.ax.set_xlabel('Time /days')
        self.ax.set_ylabel('# People')

        #.grid(b=True, which='major', c='w', lw=2, ls='-')
        
        self.ax.scatter(self.data['time'], self.data['deaths'], label = 'real Deaths')
        self.ax.scatter(self.data['time'], self.data['recovered'], label = 'real Recovered')
        self.ax.scatter(self.data['time'], self.data['cases']-self.data['recovered']-self.data['deaths'], label = 'real Infected')
        
        if not (self.Dates is None):
            for key, value in Dates[self.country].items():
                ind = np.argwhere(self.data['date'] == key)[0][0]
                self.ax.plot([ind,ind],[0,30000], value[1], label = self.country+'-'+ value[0])
            
        self.ax.legend()
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        #fig = plt.figure(figsize=(17,7))
        #self.axins2.plot(np.pad(self.beta_f(self.t), (N, 0), 'constant', 
        #                constant_values=(self.beta_f(1), 0)),'--', color = 'white' , alpha=0.5, lw=2, label='beta_function - Day:'+str(self.beta_t0)+'Slope:'+str(self.alpha))
        #self.axins2.legend()
        
        # Turn ticklabels of insets off
        #self.axins2.tick_params(labelleft=False, labelbottom=False)

    def train(self):
        print(self.bounds)
        self.tbeta_flag = 0
        self.t = np.linspace(0, len(self.data['time']), len(self.data['time'])) 
        self.optimal = minimize(self.loss, 
            #method = 'Nelder-Mead',
            method = 'TNC',
            bounds = self.bounds, tol=1e-10,
            x0 = [self.E0, self.delta, self.gammaR, self.gammaD, self.mu, self.beta0, self.alpha, self.beta_t0],)
            #options={'disp': True, 'maxcor': 100, 'ftol': 2.220446049250313e-12, 
            #         'gtol': 1e-09, 'eps': 1e-11, 'maxfun': 500000, 'maxiter': 500000, 'iprint': -1, 'maxls': 1000})#, self.alpha, self.beta_t0, self.epsilon, self.omega])
        
        print(self.optimal)
        
        print()
        print('δ: Days for symptoms to appear: '+ str(1/self.delta))
        print('1/γR: Days to recovery: '+ str(1/self.gammaR))
        print('1/γD: Days to death: '+ str(1/self.gammaD))
        print('μ: proportion of cases who die: '+ str(self.mu))
        print('E0: initial exposed: '+ str(self.E0))
        print('β0: rate of infection: '+ str(self.beta0))
        print('Days before Lockdown: '+ str(self.beta_t0))
        print('Lockdown strength: '+ str(self.alpha))
    
        self.E0, self.delta, self.gammaR, self.gammaD, self.mu, self.beta0, self.alpha, self.beta_t0  = self.optimal.x #, self.alpha, self.beta_t0 , self.epsilon, self.omega  = self.optimal.x
        self.t = np.linspace(0, 200, 200)
        self.prediction = self.integrate()

    def loss(self, point):
        self.E0, self.delta, self.gammaR, self.gammaD, self.mu, self.beta0, self.alpha, self.beta_t0  = point

        self.integrate()
        
        Infected = self.data['cases'].values - self.data['recovered'].values - self.data['deaths'].values
                
        #l11 = np.sqrt(np.mean((self.t*((self.IR + self.ID) - (Infected)))**2))
        l11 = np.sqrt(np.mean(((self.t + self.t[::-1])*((self.IR + self.ID + self.R + self.D) - (self.data['cases'].values)))**2))
        l21 = np.sqrt(np.mean(((self.t + self.t[::-1])*((self.R) - (self.data['recovered'].values)))**2))
        l31 = np.sqrt(np.mean(((self.t + self.t[::-1])*((self.D) - (self.data['deaths'].values)))**2))   
             
        total = np.sqrt((l11)**2 + ((l21))**2 + ((l31))**2) 
        
        
        return total

"""
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
import theano.tensor as tt

def my_out():
    A Gaussian log-likelihood function for a model with parameters given in theta

    sir_model = DifferentialEquation(
            func = simulator.deriv_theta,
            times = simulator.tbay,
            n_states=6,
            n_theta=6,
            t0=0,
            )
    
    sir_curves  = sir_model(y0 = simulator.y0, theta = simulator.theta)
    
    S = sir_curves.T[0]
    E = sir_curves.T[1]
    IR = sir_curves.T[2]
    ID = sir_curves.T[3]
    R = sir_curves.T[4]
    D = sir_curves.T[5]

    print([IR + ID + R + D, R, D])     

    return tt.stack([IR + ID + R + D, R, D]).T

with pm.Model() as model:

    simulator.y0 = [simulator.S0, simulator.E0, simulator.IR0, simulator.ID0, simulator.R0, simulator.D0]

    #sigma = pm.HalfCauchy('sigma', 1, shape=3)
    #sigma = pm.HalfCauchy('sigma',1)

    # R0 is bounded below by 1 because we see an epidemic has occured
    keys_vals = ['E0', 'delta', 'gammaR', 'gammaD', 'beta0', 'alpha', 'mu']

    min_val = []
    max_val = []

    simulator.theta = []
    for i, cn in enumerate(keys_vals):
        if cn != 'E0':
            print(simulator.bounds[i][0], simulator.bounds[i][1])
            simulator.theta += [pm.Uniform(cn, simulator.bounds[i][0], simulator.bounds[i][1])]


    x = pm.Data("time", np.arange(0,len(simulator.data['time'])))

    CData = simulator.data['cases'].values
    RData = simulator.data['recovered'].values
    DData = simulator.data['deaths'].values

    simulator.tbay = np.arange(0,len(simulator.data['cases'].values))

    cases =  np.array([CData, RData, DData]).T.tolist()

    #Y = pm.Lognormal('Y', mu=pm.math.log(my_out()), sd=sigma, observed=cases)
    
    
    Y = pm.Normal('Gradient', mu=my_out(), sd=sigma, observed=cases)
    
    #simulator.prior = pm.sample_prior_predictive()
    
    simulator.trace = pm.sample(30, tune=10, target_accept=0.9, cores=1)
    
    simulator.posterior_predictive = pm.sample_posterior_predictive(simulator.trace, samples=10)
    
    simulator.data = az.from_pymc3(trace=simulator.trace, prior = simulator.prior, posterior_predictive = simulator.posterior_predictive)
    
    cases = np.array([IData,RData,DData]).T.tolist()
    
    
    sigma = pm.HalfNormal('sigma', 3, shape=3)
    
    Y = pm.Lognormal('Gradient', mu=pm.math.log(my_out()), sd=sigma, observed=cases)
    
    #Y = pm.MvNormal('model', mu=my_out(), cov=true_cov, observed=cases)

    #prior = pm.sample_prior_predictive()
    #trace = pm.sample(10,tune=10, target_accept=0.9, cores=1)
    trace =  pm.sample_smc()
    posterior_predictive = pm.sample_posterior_predictive(trace)
    
    #data = az.from_pymc3(trace=trace, prior = prior, posterior_predictive = posterior_predictive)
"""    