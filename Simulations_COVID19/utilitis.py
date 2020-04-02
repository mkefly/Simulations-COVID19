import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
from datetime import timedelta, datetime
#import plotly as py
#import plotly.express as px
#import plotly.graph_objs as go
#from plotly.subplots import make_subplots
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True) 
import time
from geopy.geocoders import Nominatim, options
import os
import requests
import io
from Simulations_COVID19 import utilitis
from scipy import interpolate

options.default_timeout = 20
geolocator = Nominatim(user_agent="COVID19v2")

class data_loader:
    """Loads the data from Johns Hopkins COVID19 repository, calculates the Growth-factor (GF) & rate (GR)
        can plot the data and output html content.
    """    
    def __init__(self, countries = None):
        """[Constructor]
        
        Keyword Arguments:
            countries {[list]} -- [list of countries we are interested in analysing] (default: {None})
        """     
        self.countries = countries
        self.data_read = {}

    def collect_data_neherlab(self, path='./covid19_scenarios_data/case-counts/', path_out="./COVID19_dash/assets/data/", get_geo_loc = False):
        df = pd.DataFrame()
        loc_dic = pd.read_json(path_out+'locations.json')
            
        for r, d, f in os.walk(path):
            for file in f:
                if '.tsv' in file:
                    country = os.path.basename(os.path.normpath(r)).capitalize()
                    
                    location = file[:-4].split("-")
                    location = location[len(location)-1]
                    location = location.capitalize()
                    df_temp = pd.read_csv(r+'/'+file, header=3, delimiter = '\t', na_values='')    

                    df_temp['longitude'] = None
                    df_temp['latitude'] = None
                    
                    if country == "Ecdc":
                        country = location
                        
                    df_temp['country'] = country
                    df_temp['location'] = location 
                                        
                    if location not in loc_dic.keys():
                        try:
                            locat = geolocator.geocode(location, timeout=None)   
                        except:
                            print(location,'time out') 
                        if locat: 
                            loc_dic[location] = [locat.longitude,locat.latitude]
                        else:
                            loc_dic[location] = [None,None]
                            
                    df_temp['longitude'] = loc_dic[location][0]
                    df_temp['latitude'] = loc_dic[location][1]
                    
                    if location == country:
                        df_temp['location'] = 'Full Country'
                        
                    if country == 'Unitedstates':
                        df_temp['country'] = 'USA'

                    #print(df_temp.tail(10))
                    # Ensure monotonicity and interpolate nans
                    df_temp = clean_monotonicity_interpol_nans(df_temp)   
                    #print(df_temp.tail(10))                    
                    # Eliminate compodent that are above day of precission i.e hours...
                    df_temp['time'] = df_temp['time'].str.slice(0,10)

                    if sum(df_temp.columns == 'country'):
                        df = pd.concat([df,df_temp],sort=False)
                        
        df_temp = df[df.location.isin(['Full Country'])].groupby(['country','time']).max().sort_values(['time']).reset_index().groupby(['time']).sum().reset_index()
        df_temp['country'] = 'The World'
        df_temp['location'] = 'countries in the table'     
        df_temp['longitude'] = 6.395626
        df_temp['latitude'] = 14.056159
        df_temp = clean_monotonicity_interpol_nans(df_temp) 
        
        df = pd.concat([df,df_temp],sort=False)

        if get_geo_loc:
            pd.DataFrame(loc_dic).to_json(path_out+'locations.json')
        self.data = df.reset_index().drop(['index'],axis=1)
        pd.DataFrame(self.data).to_json(path_out+'cases_world.json')
        return self.data
                    
    def recover_data(self, field):
        url = self.url
        self.data_read[field] = pd.read_csv(url, error_bad_lines=False)
        
    def compile_data(self, field, country):        
        data = pd.DataFrame()
        value = self.data_read[field][self.data_read[field]['Country/Region'].isin([country])].drop(['Province/State',
                                                    'Country/Region','Lat','Long'], axis=1).sum(axis=0)

        days = len(value.values)
        data['Days'] = np.linspace(0,days-1,days)
        data['Date'] = value.index.values
        data['Country'] = country
        data[field] = value.values
        data = self.add_growth(data, field)
        return data

    def load_data(self, country):
        data = pd.DataFrame()
        data = self.compile_data('Confirmed', country)
        for field in ['Recovered','Deaths']:
            data = pd.concat([data, self.compile_data(field, country)], axis=1)
        data = data[:-1].loc[:,~data.columns.duplicated()]
        return data
    
    def load_countries(self, flag_full_list = False):
        ### Change order
        
        self.data = pd.DataFrame()
        list_data = []
        
        if flag_full_list:
            self.countries = list(self.data_read['Confirmed']['Country/Region'].unique())
            
        for cn in self.countries:
            list_data += [self.load_data(cn).T]
        df = pd.concat(list_data, axis=1).T
        self.data = df.reset_index() 

    def add_growth(self, data, field):
        if np.sum(data.columns.str.contains(field)):
            if np.sum(data.columns.str.contains(field+'-GR')) == 0:
                N = sum(data.columns.str.contains(field) <= 10)
                Data0 = data.T[data.columns.str.contains(field)].T
                Data1 = Data0.shift(-1, axis = 0)
                Data2 = Data0.shift(-2, axis = 0) 

                epsilon =.01
                Growthr = (Data1-Data0)/(Data1 + epsilon)
                Growthr.columns = Growthr.columns.str.replace(field,field+'-GR')
                Growthr = Growthr.T.shift(1, axis = 1).T
                Growthr = Growthr.clip(0, 1)
                Growthr.iloc[0:N] = np.nan
                
                data = pd.concat([data,Growthr],axis=1)
                
        return data

    def plot_percent(self, country, i, N = 1, type_growth = 'GR', style_color = 'dark_background'):
        colors = plt.cm.rainbow(np.linspace(0, 1, N))

        data = self.data[self.data["Country"].isin([country])]
        data = data[data['Confirmed'] > 3]
        #data = data[data[field+'-'] > -1]
        plt.title(country)
        plt.plot(data['Date'],data['Confirmed'+'-'+type_growth], '-', color = colors[i] , label = 'Confirmed' + '-' + type_growth)
        plt.plot(data['Date'],data['Deaths'+'-'+type_growth], '-.', color = colors[i] , label = 'Confirmed' + '-' + type_growth)

        plt.xlabel('Day')
        plt.ylabel('Percental growth')
        plt.ylim(0,1)
        x_ticks = np.arange(0, np.max(data['Days'])-np.min(data['Days']), 5)       
        plt.xticks(x_ticks, rotation='vertical')

        plt.grid(alpha = 0.1)

    def plot_percents(self, country_list, type_growth = 'GR', style_color = 'dark_background'):
        plt.style.use(style_color)
        for i, country in enumerate(country_list):
            plt.subplot(np.ceil(len(country_list)/4), 4, i+1)
            self.plot_percent(country, i , len(country_list), type_growth, style_color)
            plt.tight_layout(w_pad = 3)
            plt.legend()
   
    def date_to_html(self, name, folder):
        html_content = '<h3>Date of the last update: '+str(self.data['Date'].iloc[-1])+'</h3>'
        str_to_html(html_content, name, folder = './')         
  
    def compile_table_to_html(self, name, folder = './', field = 'Deaths', sort_by = None):
        data_table = {}
        for country in self.data['Country'].unique():
            data = self.data[self.data['Country'] == country]
            if len(data[field]) > 2:
               data_table[country] = [data['Days'].iloc[-1],data[field].iloc[-1],str(np.round(data[field+'-'+'GR'].iloc[-1]*100, decimals=2))+" %", np.round(data[field+'-'+'GR'].iloc[-1], decimals=2)]
        list_colums = ['Days', field, field+'-'+'GR', field+'-'+'GR']
        
        if list_colums:
            field = sort_by
            
        utilitis.data_to_html(data_table, list_colums, name, folder, field)
 
    def save_table(self, path = '', file = 'phenom.json'):
        data = pd.DataFrame()

        field = 'deaths'
        vmean = []
        vmin = []
        vmax = []
        model = []
        Countries = []
        times = []
        datas = []
        for method, values in self.post_preds.items():
            for Country in values.keys():
                mean_value = np.mean(self.post_preds[method][Country][Country].T,axis=1)   
                std_value = np.std(self.post_preds[method][Country][Country].T,axis=1)
                vmean += [mean_value]
                vmin += [mean_value + std_value]
                vmax += [mean_value - std_value]
                N = len(vmax[0])
                Countries += [Country for i in range(N)]
                model += [method for i in range(N)]
                times += date_list(self.data[(self.data['Country'] == Country)].time.head(1).values[0], N)
                temp = self.data[(self.data.Country == Country)].groupby(['time']).max()[field].values
                datas += [np.pad(temp, (0, N-len(temp)), 'constant', constant_values=(0, np.nan))]
        data['vmean'] = np.concatenate(vmean)
        data['vmin'] = np.concatenate(vmin)
        data['vmax'] = np.concatenate(vmax)
        data['data'] = np.concatenate(datas)
        data['Country'] = Countries
        data['model'] = model
        data['time'] = times
        data.to_json(path+file)


def plot_to_png(name, folder = './', **kwargs):
    lgd = plt.legend(**kwargs)
    #lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fancybox=True, shadow=True, ncol=6)
    plt.savefig(folder+name+'.png', dpi=90, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

def plotly_write_html(fig, name, folder = './'):
    return fig.write_html(folder+name)

def date_list(date, N, style = '%Y-%m-%d'):
    date = datetime.strptime(date, style)
    list_results = [(date + timedelta(days=n)).strftime(style) for n in range(0, N) ]
    return list_results

def multivariateGrid(col_x, col_y, col_k, df, k_is_color=False, scatter_alpha=.5):
    plt.style.use('dark_background')
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['color'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)
            plt.grid(alpha=0.2)
        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=k_is_color[name]
        g.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color),
        )
        sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            #color='white'
        )
        
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            vertical=True,
            #color='white'
        )
        sns.set_style("whitegrid", {'axes.grid' : False})
    plt.legend(legends) 
    
def str_to_html(html_content, name, folder = './'):        
    html_file = open(folder+name+".html", "wt")
    html_file.write(html_content)
    html_file.close()

def data_to_html(data_table, list_colums, name, folder, sort_by = None):
    
    if sort_by is None:
        sort_by = list_colums[0]
        
    html_content = pd.DataFrame.from_dict(data_table, orient='index', 
                    columns = list_colums).sort_values(by=[sort_by], ascending = False).to_html().replace('\n', '')
    str_to_html(html_content, name, folder = './')   
  
def get_version_information():
    version_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), '.version')
    try:
        with open(version_file, 'r') as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")
        
        
def clean_monotonicity_interpol_nans(df_temp):
    for cn in ['deaths','cases','recovered']: #,'hospitalized','ICU']:
        A = df_temp[cn].interpolate().values
        A[-10:] = np.maximum.accumulate(A[-10:])
        df_temp[cn] =  np.minimum.accumulate(A[::-1])[::-1]
        
    return df_temp