import numpy as np
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
from datetime import timedelta, datetime
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 

import os
import requests
import io
from Simulations_COVID19 import utilitis

class data_loader:
    def __init__(self, countries = None):
        self.countries = countries
        self.data_read = {}
        for field in ['Confirmed','Recovered','Deaths']:
            self.recover_data(field)
                    
    def recover_data(self, field):
        url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-'+field+'.csv'
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
        self.data = df.reset_index() #.loc[:,~df.columns.duplicated()]

    def add_growth(self, data, field):
        if np.sum(data.columns.str.contains(field)):
            if np.sum(data.columns.str.contains(field+'-GR')) == 0:
                Dates = data.T[data.columns.str.contains("Da")].shift(-1, axis = 1).T 

                Data0 = data.T[data.columns.str.contains(field)].T
                Data1 = Data0.shift(-1, axis = 0)
                Data2 = Data0.shift(-2, axis = 0) 

                epsilon =.01
                Growthr = (Data1-Data0)/(Data1 + epsilon)
                Growthr.columns = Growthr.columns.str.replace(field,field+'-GR')
                Growthr = Growthr.T.shift(1, axis = 1).T

                Growthf = (Data1-Data0)/(Data2 - Data1 + epsilon)
                Growthf.columns = Growthf.columns.str.replace(field,field+'-GF')
                Growthf = Growthf.T.shift(2, axis = 1).T

                data = pd.concat([data,Growthr,Growthf],axis=1)#.head(-1)
        return data
    
    def show_map(self, field = 'Deaths', exept = ['None']):
        data = self.data[~self.data["Country"].isin(exept)]
        fig = px.choropleth(data, 
                            locations="Country", 
                            locationmode = "country names",
                            color=field, 
                            hover_name=field, 
                            animation_frame="Date",
                            #mapbox_style="carto-positron",
                            template="plotly_dark"
                        )

        fig.update_layout(
            title_text = 'Spread of Coronavirus',
            title_x = 0.5,
            geo=dict(
                showframe = True,
                showcoastlines = True,
            ))
            
        fig.show()
        
        return fig 
    
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
            
def plot_to_png(name, folder = './', **kwargs):
    lgd = plt.legend(**kwargs)
    #lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), fancybox=True, shadow=True, ncol=6)
    plt.savefig(folder+name+'.png', dpi=90, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

def plotly_write_html(fig, name, folder = './'):
    return fig.write_html(folder+name)

def date_list(date, N):
    date = datetime.strptime(date, '%m/%d/%Y')
    list_results = [(date + timedelta(days=n)).strftime('%m/%d/%Y') for n in range(0, N) ]
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
        