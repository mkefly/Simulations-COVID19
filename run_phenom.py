import sys 
import os 
import Simulations_COVID19 as SCovid19 
import numpy as np 
import pandas as pd 

''' Load and collect data for the web, fit the phenom models and save the info for the web. ''' 

samples = 300 
number_days = 200 
n_steps = 25
list_countries = ['Spain','Italy','United States of America','France','United Kingdom','Germany','Netherlands'] 

folder_html = './tables/' 
folder_images = './images/' 

for folder in [folder_html,folder_images]: 
	if not os.path.exists(folder): os.makedirs(folder) 

Dates = {'Spain':{'3/8/20':['M8','-.'],'3/14/20':['Lockdown','--']}, 'Italy':{'3/9/20':['Lockdown','--']}, 'Hubai':{'1/23/20':['Lockdown','--']}} 

####### LOAD ####### 
path='./covid19_scenarios_data/case-counts/' 
path_out = './COVID19_dash/assets/data/'
dataloader = SCovid19.data_loader() 
dataloader.collect_data_neherlab(path, get_geo_loc = False) 

phenomsirs = SCovid19.phenom_simulator(countries = list_countries, data_table = dataloader.data)
for method in ['log-model', 'gompertz-model']:
    _ = phenomsirs.sample_posterior_predictive_model(method = method, field = 'deaths', samples = samples, number_days = number_days, n_steps=n_steps)
phenomsirs.save_table(path = path_out, file = 'phenom.json')
