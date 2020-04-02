#%%
import sys 
import os 
import Simulations_COVID19 as SCovid19 
import numpy as np 
import pandas as pd 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

''' Load and collect data for the web, fit the phenom models and save the info for the web. ''' 

samples = 30 
number_days = 200 
n_steps = 25
list_countries = ['Spain','Italy','United states of america','United kingdom','Germany','Netherlands'] 

folder_html = './tables/' 
folder_images = './images/' 

for folder in [folder_html,folder_images]: 
	if not os.path.exists(folder): os.makedirs(folder) 

Dates = {'Spain':{'3/8/20':['M8','-.'],'3/14/20':['Lockdown','--']}, 'Italy':{'3/9/20':['Lockdown','--']}, 'Hubai':{'1/23/20':['Lockdown','--']}} 

####### LOAD ####### 
path='./covid19_scenarios_data/case-counts/' 
path_out = './COVID19_dash/assets/data/'
dataloader = SCovid19.data_loader() 
dataloader.collect_data_neherlab(path) 

'''
phenomsirs = SCovid19.phenom_simulator(countries = list_countries, data_table = dataloader.data)
for method in ['log-model', 'gompertz-model']:
    _ = phenomsirs.sample_posterior_predictive_model(method = method, field = 'deaths', samples = samples, number_days = number_days, n_steps=n_steps)

phenomsirs.save_table(path = path_out, file = 'phenom.json')




Run beta-SIIERS exampe
'''
"""
delta, gammaR, gammaD, mu, beta0, alpha, beta_t0, omega, epsilon  = 1/24, 1/24, 1/35, 0.48, 1.5, 1.0, 50.0, 0.0, 00.0

Dates = {'Spain':{'3/8/20':['M8','-.'],'3/13/20':['Lockdown','--']},
'Italy':{'3/9/20':['Lockdown','--']},
'Hubai':{'1/23/20':['Lockdown','--']}}

field = 'Deaths'
method = 'gompertz-model'
country = 'Spain'
flag_res = 'False'
color = 'red'
beta_t0 = {'Spain': 51,'Italy': 47}
population = { 'Spain': 46.66  * 1000000 * 10000, 'Italy': 60.48 * 1000000}

location = 'Full Country'


for country in ['Spain']:#,'Italy']:
    data = dataloader.data[dataloader.data.country == 'Spain']
    data['Country'] = dataloader.data['country'] + ', '+  dataloader.data['location']
    data = data.groupby(['Country','time']).sum().reset_index()
    bounds = [(0,2000),(1/30,1/7),(1/24,1/7), (1/34,1), (0.001,1), (0.001,60), (0.2, 1.0), (beta_t0[country],beta_t0[country])]
    simulator = SCovid19.siers_simulator(data, delta, gammaR, gammaD, mu, beta0, alpha, beta_t0[country], omega, epsilon, population[country], bounds)
    simulator.train()
    simulator.plot_results(ylim = [0,55000])
    plt.show()
"""

"""
['Brazil' 'Estonia' 'Cuba' 'Kyrgyzstan' 'Trinidad and tobago' 'Ukraine'
 'Tunisia' 'Curaçao' 'Virgin islands (british)' 'Papua new guinea'
 'Guinea' 'Rwanda' 'Faroe islands' "Côte d'ivoire" 'French polynesia'
 'Armenia' 'Honduras' 'Greenland' 'Congo, democratic republic of the'
 'Kosovo' 'Aruba' 'North macedonia' 'Algeria' 'Poland' 'Kazakhstan'
 'Argentina' 'China' 'Hungary' "Lao people's democratic republic"
 'Iceland' 'Saudi arabia' 'Liberia' 'Colombia' 'Guernsey' 'Kenya'
 'Afghanistan' 'Ghana' 'Guatemala' 'Zambia' 'Cambodia' 'Angola' 'Djibouti'
 'Namibia' 'Jamaica' 'United states of america'
 'Saint vincent and the grenadines' 'Equatorial guinea' 'Montserrat'
 'Cases on an international conveyance japan' 'Bermuda' 'Madagascar'
 'Grenada' 'Montenegro' 'Cyprus' 'El salvador' 'Gabon' 'Ecuador' 'Libya'
 'Nigeria' 'Saint kitts and nevis' 'Korea, republic of' 'Malaysia'
 'Senegal' 'Oman' 'Panama' 'Switzerland' 'Czechia'
 'Tanzania, united republic of' 'Virgin islands (u.s.)' 'Nepal'
 'Gibraltar' 'Guam' 'Australia' 'Cameroon' 'Leste' 'Turkey' 'Monaco'
 'Benin' 'Mozambique' 'New caledonia' 'Andorra' 'Greece' 'Seychelles'
 'Sweden' 'Venezuela (bolivarian republic of)' 'Central african republic'
 'Belgium' 'Taiwan, province of china' 'Costa rica' 'Haiti' 'Nicaragua'
 'Iran (islamic republic of)' 'Myanmar' 'San marino' 'Maldives' 'Somalia'
 'Israel' 'Latvia' 'Sudan' 'Uganda' 'Austria' 'Bissau' 'Serbia' 'Mali'
 'Suriname' 'Bangladesh' 'Burkina faso' 'Thailand' 'Cabo verde' 'Kuwait'
 'South africa' 'Dominica' 'France' 'Spain' 'Russian federation'
 'Holy see' 'Mauritius' 'Portugal' 'Eswatini' 'Turks and caicos islands'
 'Paraguay' 'Mongolia' 'Bahrain' 'Luxembourg' 'Guyana'
 'Syrian arab republic' 'Bolivia (plurinational state of)' 'Canada'
 'United arab emirates' 'Anguilla' 'Azerbaijan' 'Qatar' 'Viet nam'
 'Bhutan' 'Germany' 'United kingdom' 'Cayman islands' 'Puerto rico'
 'Gambia' 'Togo' 'Sri lanka' 'Bulgaria' 'Mauritania' 'Pakistan' 'Jordan'
 'Fiji' 'Croatia' 'Niger' 'Moldova, republic of' 'Bosnia and herzegovina'
 'Egypt' 'Dominican republic' 'Congo' 'Indonesia' 'Slovakia' 'Slovenia'
 'Lithuania' 'Chile' 'Romania' 'Belize' 'Morocco' 'Mexico' 'Norway'
 'Palestine, state of' 'Uruguay' 'Malta' 'Japan' 'Finland' 'Uzbekistan'
 'Ireland' 'Sint maarten (dutch part)' 'Brunei darussalam' 'Chad'
 'Eritrea' 'Jersey' 'Belarus' 'Denmark' 'Italy' 'Saint lucia' 'Zimbabwe'
 'Liechtenstein' 'Netherlands' 'Georgia' 'Antigua and barbuda' 'Ethiopia'
 'Isle of man' 'Bahamas' 'Iraq' 'Albania' 'New zealand' 'Philippines'
 'Barbados' 'India' 'Peru' 'Singapore' 'Lebanon' 'USA']
 """

# %%
