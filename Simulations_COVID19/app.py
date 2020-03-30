# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import matplotlib

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

path='./assets/data/'

###################################
# Private function
###################################

backgroundColor1 = '#252e40'
backgroundColor2 = '#252e40'
backgroundColor3 = '#252e40'
backgroundColorplots = '#252e40'

days_size_color = backgroundColor1
cases_color = backgroundColor1
recovered_color = backgroundColor1
deaths_color = backgroundColor1


family_font_figures = "HelveticaNeue"
family_font_tabs = "HelveticaNeue"

background_color_banners = "#F2F2F2"

color_font_table = '#e6e3e3'
fontWeight_table = 'Normal'
color_fort_plot = '#e6e3e3'
color_fort_map = '#e6e3e3'

def add_growth(data, field):
    if np.sum(data.columns.str.contains(field)):
      #data = data.drop([data.columns.str.contains(field+'-GR')],axis = 1)
      N = sum(data[field] <= 10)

      Data0 = data.T[data.columns.str.contains(field)].T
      Data1 = Data0.shift(-1, axis = 0)
      Data2 = Data0.shift(-2, axis = 0) 

      epsilon =.01
      Growthr = (Data1-Data0)/(Data1 + epsilon)
      Growthr.columns = Growthr.columns.str.replace(field,field+'-GR')
      Growthr = Growthr.T.shift(1, axis = 1).T
      Growthr = Growthr.clip(0, 1)*100
      Growthr.iloc[0:N] = np.nan

      data[field+'-GR'] = Growthr[field+'-GR']
    return data

def data_funct(df, location, country):
  if (country == 'Full Country') & (location == 'Full Country'):
    A = df[df['location'] == 'Full Country'].groupby(['time']).sum()
    B = df[df['location'] != 'Full Country'].groupby(['time']).sum()
    A[['hospitalized','ICU','recovered']] = B[['hospitalized','ICU','recovered']]
    df_temp = A.reset_index()
    country = 'The entire'
    location = 'World'
  else:
    if (sum(country == np.append(df[df['location'] != 'Full Country'].country.unique(),'World')))&(location == 'Full Country'):
      df_temp = df[(df.country == country)&(df.location != 'Full Country')].groupby(['time']).sum().reset_index()
    else:
      df_temp = df[(df['location'] == location)&(df['country'] == country)]
  for k in ['deaths','cases']:
    df_temp = add_growth(df_temp, k)
  return df_temp, country, location


def make_dcc_pd(country, dataframe):
    if country == "The World":
      rows_list = ['country','cases','deaths','recovered','hospitalized','ICU','time']
      dataframe = dataframe[dataframe.location == "Full Country"] 
    else:
      rows_list = ['location','cases','deaths','recovered','hospitalized','ICU','time']
      dataframe = dataframe[dataframe.country == country]
    return [dataframe.sort_values(['deaths'], axis = 0, ascending=False), rows_list]

def make_dcc_country_tab(country, dataframe):

    dataframe, rows_list = make_dcc_pd(country, dataframe)
    
    '''This is for generating tab component for country table'''
    return dcc.Tab(label=country,
            value=country,
            className='custom-tab',
            selected_className='custom-tab--selected',
            children=[dash_table.DataTable(
                    id='datatable-interact-location-{}'.format(country),
                    # Don't show coordinates
                    columns=[{"name": i, "id": i}
                        for i in rows_list],
                    # But still store coordinates in the table for interactivity
                    data=dataframe.to_dict("rows"),
                    row_selectable="single" if country else False,
                    #sort_action="native",
                    style_as_list_view=True,
                    style_cell={'font_family': family_font_tabs,
                                  'font_size': '1.2rem',
                                  'padding': '.1rem',
                                  'backgroundColor': backgroundColor3, },
                    fixed_rows={'headers': True, 'data': 0},
                    style_table={'minHeight': '300px',
                                    'height': '300px',
                                    'maxHeight': '300px'},
                    style_header={'backgroundColor': backgroundColor3,
                                    'fontWeight': 'bold'},
                    style_cell_conditional=[{'if': {'column_id': i}, 'width': str((90)/len(rows_list))+'%', 'color': color_font_table, 'textAlign': 'left', 'fontWeight': fontWeight_table} for i in rows_list]
                                    )])


################################################################################
# Functions for the plots
################################################################################

def colhex(color):
  return matplotlib.colors.to_hex(color)

def create_add_trace(fig, df_temp, list_cn, name_list, hovertext_list, N = 10, iadd = 0, backgroundColorplots = backgroundColorplots):
  colors = plt.cm.rainbow(np.linspace(0, 1, N))
  for i, cn in enumerate(list_cn):  
    fig.add_trace(go.Scatter(x=df_temp['time'], y=df_temp[cn],
                                      mode='lines+markers',
                                      line_shape='linear',
                                      name= name_list[i],
                                      line=dict(color=colhex(colors[i+iadd]), width=4),
                                      marker=dict(size=4, color='#f4f4f2',
                                                  line=dict(width=1, color=colhex(colors[i+iadd]))),
                                      text=[str(d) for d in df_temp['time']],))
                                      #hovertext=[ hovertext_list[i] + '<br>' + str(d) + 'cases<br>' for d in df_temp[cn]],
                                      #hovertemplate='<b>%{text}</b><br></br>' +
                                      #              '%{hovertext}' +
                                      #              '<extra></extra>'
  return fig

def figure_top_style(fig, tickList = None, xscale = "liner", yscale = "liner"):
  yaxis = dict(showline=False, linecolor='#272e3e',
        zeroline=False,
        # showgrid=False,
        gridcolor='rgba(203, 210, 211,.3)',
        gridwidth=.1,
        tickmode='array',
    )

  if tickList is not None:
    yaxis.update(dict(tickvals=tickList,
        # Set tick label accordingly
        ticktext=[str(i) for i in tickList]))


  # Customise layout
  fig.update_layout(
    margin=go.layout.Margin(
        l=10,
        r=10,
        b=10,
        t=5,
        pad=0
    ),
    yaxis_type=yscale,
    yaxis=yaxis,
    xaxis=dict(
        showline=False, linecolor='#272e3e',
        showgrid=False,
        gridcolor='rgba(203, 210, 211,.3)',
        gridwidth=.1,
        zeroline=False
    ),
    xaxis_type=xscale,
    xaxis_tickformat='%b %d',
    hovermode='x',
    legend_orientation="h",
    plot_bgcolor=backgroundColorplots,
    paper_bgcolor=backgroundColorplots,
    font=dict(color='#e6e3e3', size=10)
  )

  fig.update_xaxes(showline=True, linewidth=0.5, linecolor='#e6e3e3', mirror=True)
  fig.update_yaxes(showline=True, linewidth=0.5, linecolor='#e6e3e3', mirror=True)

  return fig

def figure_top_style_2(fig, country, location, backgroundColorplots = backgroundColorplots, yaxis_title="Cumulative cases numbers"):
  # Customise layout
  fig.update_layout(
      margin=go.layout.Margin(
          l=10,
          r=10,
          b=10,
          t=5,
          pad=0
      ),
      annotations=[
          dict(
              x=.5,
              y=.4,
              xref="paper",
              yref="paper",
              text=country+','+location,
              opacity=0.5,
              font=dict(family=family_font_figures,
                        size=60,
                        color="grey"),
          )
      ],
      yaxis_title=yaxis_title,
      yaxis=dict(
          showline=False, linecolor='#272e3e',
          zeroline=False,
          # showgrid=False,
          gridcolor='rgba(203, 210, 211,.3)',
          gridwidth=.1,
          tickmode='array',
      ),
      xaxis_title="Select A Location From Table",
      xaxis=dict(
          showline=False, linecolor='#272e3e',
          showgrid=False,
          gridcolor='rgba(203, 210, 211,.3)',
          gridwidth=.1,
          zeroline=False
      ),
      xaxis_tickformat='%b %d',
      hovermode='x',
      legend_orientation="h",
      legend=dict(x=.02, y=.95, bgcolor="rgba(0,0,0,0)",),
      plot_bgcolor=backgroundColorplots,
      paper_bgcolor=backgroundColorplots,
      font=dict(color='#e6e3e3', size=10)
  )
  return fig

def create_map(df_temp,  longitude= 6.395626, latitude= 14.056159, zoom = 1, backgroundColorplots = backgroundColorplots, mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNqdnBvNDMyaTAxYzkzeW5ubWdpZ2VjbmMifQ.TXcBE-xg9BFdV2ocecc_7g"):
  data = [
  go.Scattermapbox(
      lat=df_temp.latitude.values,
      lon=df_temp.longitude.values,
      mode='markers',
      marker=go.scattermapbox.Marker(
          size=8,
          color=[map_selectcolor(df_temp.loc[i], i,'out') for i in df_temp.index],
          opacity=0.3
      ),
      hoverinfo='none'
  ),
  go.Scattermapbox(
      lat=df_temp.latitude.values,
      lon=df_temp.longitude.values,
      mode='markers',
      marker=go.scattermapbox.Marker(
          size=6,
          color=[map_selectcolor(df_temp.loc[i], i,'out') for i in df_temp.index],
          opacity=0.6
      ),
      hoverinfo='none'
  ),
  go.Scattermapbox(
      lat=df_temp.latitude.values,
      lon=df_temp.longitude.values,
      mode='markers',
      marker=go.scattermapbox.Marker(
          size=3,
          color=[map_selectcolor(df_temp.loc[i], i,'inner') for i in df_temp.index],
          opacity=1
      ),
      hovertemplate= "%{hovertext}<br>" +
                      "<extra></extra>",
      hoverinfo='text',
      hovertext=['<br>Country:'+str(df_temp['country'].loc[i])+'<br>Location:'+str(df_temp['location'].loc[i])+'<br>Confirmed:'+str(df_temp['cases'].loc[i])+'<br>Death:'+str(df_temp['deaths'].loc[i])+'<br>Recovered:'+str(df_temp['recovered'].loc[i])+'<br>hospitalized:'+str(df_temp['hospitalized'].loc[i])+'<br>ICU:'+str(df_temp['ICU'].loc[i]) for i in df_temp.index]
  )]

  layout = go.Layout(
      title='World COVID-19 map',
      autosize=True,
      hovermode='closest',
      showlegend=False)
  

  fig_map = go.Figure(data=data, layout=layout)
  
  fig_map.update_layout(
      plot_bgcolor=backgroundColorplots,
      paper_bgcolor=backgroundColorplots,
      margin=go.layout.Margin(l=10, r=10, b=10, t=0, pad=40),
      hovermode='closest',
      transition={'duration': 50},
      annotations=[
      dict(
          x=.5,
          y=-.01,
          align='center',
          showarrow=False,
          text=".... description here ....",
          xref="paper",
          yref="paper",
          font=dict(size=10, color='#e6e3e3'),
      )],
      mapbox=go.layout.Mapbox(
          accesstoken=mapbox_access_token,
          style="mapbox://styles/plotlymapbox/cjvppq1jl1ips1co3j12b9hex",
          # The direction you're facing, measured clockwise as an angle from true north on a compass
          bearing=0,
          center=go.layout.mapbox.Center(
              lat=latitude,
              lon=longitude
          ),
          pitch=0,
          zoom=zoom
      )
  )

  fig_map.update_xaxes(showline=True, linewidth=0.5, linecolor='#e6e3e3', mirror=True)
  fig_map.update_yaxes(showline=True, linewidth=0.5, linecolor='#e6e3e3', mirror=True)

  return fig_map

def map_selectcolor(df, i, location):
  Mapcolor_dic = {'Full Country':'#F4F6F7','Partial':'#909497','Infected':'#F1948A','Severe':'#d7191c','Alert':'#F4D03F','Cured':'#1a9622'}
  

  count = np.nan_to_num(df.cases) - np.nan_to_num(df.deaths) - np.nan_to_num(df.recovered)

  color = Mapcolor_dic['Cured']

  if count > 0:
    color = Mapcolor_dic['Infected']
  if count > 3000:
    color =  Mapcolor_dic['Alert']
  if count > 10000:
    color =  Mapcolor_dic['Severe']    

  if location == 'inner':
    if df.location == 'Full Country':
      return color
    else:
      return Mapcolor_dic['Partial']

  return color   

def create_add_phenom_trace(fig, df_phenom, dic_phenom):
  iadd = 0
  colors = plt.cm.rainbow(np.linspace(0, 1, len(df_phenom['Country'].unique())))
  showlegend_dic = {dic_phenom[2][0]:True,dic_phenom[2][1]:False ,dic_phenom[2][2]:False }
  dash_dic = {dic_phenom[2][0]:'solid',dic_phenom[2][1]:'dash' ,dic_phenom[2][2]:'dash' } 
  width_dic = {dic_phenom[2][0]:1,dic_phenom[2][1]:0.5 ,dic_phenom[2][2]:0.5 } 
  for k, cnk in enumerate(df_phenom.Country.unique()):
    mask = (df_phenom.Country == cnk)&(df_phenom.model == dic_phenom[-2])
    for i, cn in enumerate(dic_phenom[2]): 
      fig.add_trace(go.Scatter(x=df_phenom[mask]['time'], y=df_phenom[mask][cn],
                                   mode='lines',
                                   line_shape='linear',
                                   legendgroup=cnk,
                                   showlegend=showlegend_dic[cn],
                                   name= cnk,
                                   line=dict(color=colhex(colors[k+iadd]), width=width_dic[cn], dash=dash_dic[cn]),
                                   ))
    fig.add_trace(go.Scatter(x=df_phenom[mask].dropna(axis=0)['time'], y=df_phenom[mask].dropna(axis=0)['data'],
                               mode='markers',
                               #line_shape='linear',
                               name= 'data',
                               legendgroup=cnk,                                      
                               showlegend=showlegend_dic[cn],
                               #line=dict(color=colhex(colors[k+iadd]), width=4),
                               marker=dict(size=4, color='#f4f4f2', line=dict(width=2, color=colhex(colors[k+iadd])))))
  return fig



################################################################################
# Data processing
################################################################################



df = pd.read_json(path+'cases_world.json')
loc_dic_df = pd.read_json(path+'locations.json')
df_phenom = pd.read_json(path+'phenom.json')

world, _, _ = data_funct(df, 'Full Country', 'Full Country')


df['time'] = df['time'].str.slice(0,10)
latestDate = df.sort_values('time').tail(1)['time'].values[0]
# Save numbers into variables to use in the app
FirstDate = datetime.strptime(str('12/12/2019'), '%m/%d/%Y')
daysOutbreak = (datetime.now()-FirstDate).days

Totalcases = world.cases.tail(1).values
CasesToday = np.round((world.cases.tail(1).values - world.cases.tail(2).values[0]))
CasesInPercent = np.round(CasesToday/Totalcases*100)

TotalRecovered = world.recovered.tail(1).values
RecoveredToday = np.round((world.recovered.tail(1).values - world.recovered.tail(2).values[0]))
RecoveredInPercent = np.round(RecoveredToday/TotalRecovered*100)

TotalDeaths = world.deaths.tail(1).values
DeathsToday = np.round((world.deaths.tail(1).values - world.deaths.tail(2).values[0]))
DeathsInPercent = np.round(DeathsToday/TotalDeaths*100)



list_of_extended_countries = np.append(['The World'],df[df['location'] != 'Full Country'].country.unique())
datatable_interact = ['datatable-interact-location-{}'.format(i) for i in list_of_extended_countries]
df_temp_table = df.sort_values('time').groupby(['country','location']).tail(1)
dcc_tables = [make_dcc_country_tab(i,df_temp_table) for i in list_of_extended_countries]


df_temp = df[df['time'] == df['time'].tail(1).values[0]].reset_index().drop(['index'],axis=1)
fig_map = create_map(df_temp)


country = 'Loading..'
location = ''
# Create empty figure canvas
fig_dash = go.Figure()
# Add trace to the figure
figure_top_style(fig_dash, xscale = "date", yscale = "linear")
figure_top_style_2(fig_dash, country, location)

dic_tabs = {}
dic_tabs['Cumulative Cases Linear'] = ['figure-dash',fig_dash,['deaths','cases','recovered','hospitalized','ICU'],"Cumulative cases numbers",'linear']
dic_tabs['Cumulative Cases Log'] = ['figure-dash',fig_dash,['deaths','cases','recovered','hospitalized','ICU'],"Cumulative cases numbers",'log']
dic_tabs['Rate evolution'] = ['figure-dash',fig_dash, ['deaths-GR','cases-GR'],"24h Percentage increase rate [%]",'linear']
dic_tabs['More than 500 deaths logscale'] = ['figure-dash',fig_dash]

dic_first = {}
dic_first["Days Since Outbreak"] = ['-----    -----', str(daysOutbreak), days_size_color]
dic_first["Confirmed Cases"] = [str(CasesToday[0]) +' increase of '+ str(CasesInPercent[0]) + '%', str(Totalcases[0]), cases_color]
dic_first["Recovered Cases"] = [str(RecoveredToday[0]) +' increase of '+ str(RecoveredInPercent[0]), str(TotalRecovered[0]) , recovered_color]
dic_first["Death Cases"] = [str(DeathsToday[0]) +' increase of '+ str(DeathsInPercent[0]), str(TotalDeaths[0]), deaths_color]


fig_model = go.Figure()

dic_phenom_tabs = {}
dic_phenom_tabs['Phenom logistic model linear'] = ['figure-phenom',fig_model,['vmean','vmin','vmax'],'log-model','linear']
dic_phenom_tabs['Phenom logistic model log'] = ['figure-phenom',fig_model,['vmean','vmin','vmax'],'log-model','log']
dic_phenom_tabs['Phenom Gomper model linear'] = ['figure-phenom',fig_model,['vmean','vmin','vmax'],'gompertz-model','linear']
dic_phenom_tabs['Phenom Gomper model log'] = ['figure-phenom',fig_model,['vmean','vmin','vmax'],'gompertz-model','log']


figure_top_style(fig_model, xscale = "date", yscale = "linear")
figure_top_style_2(fig_model, country, location)

text_phenom = ["Phenomenological models:","text saying select the ...."]


##################################################################################################
# Start dash app
##################################################################################################

app = dash.Dash(__name__,
                assets_folder='./assets/',
                meta_tags=[
                    {"name": "author", "content": "Miquel Oliver"},
                    {"name": "description", "content": "The coronavirus COVID-19 monitor."},
                    {"property": "og:title", "content": "Coronavirus COVID-19 Outbreak Global Cases Monitor Dashboard"},
                    {"property": "og:type", "content": "website"},
                    {"property": "og:url", "content": "https://..... our web ......com/"},
                    {"property": "og:description", "content": "The coronavirus COVID-19 monitor/dashboard provides up-to-date data and map for the global spread of coronavirus."},
                    {"name": "viewport", "content": "width=device-width, height=device-height, initial-scale=1.0"}
                ]
      )

app.title = 'Coronavirus COVID-19 Global Monitor'

# Section for Google annlytic and donation #
app.index_string = """<!DOCTYPE html>
<html>
    <head>
      <link rel="stylesheet" href="{{ "assets/css/main.css" | relative_url }}" />
  <!--[if lte IE 9]><link rel="stylesheet" href="{{ "assets/css/ie9.css" | relative_url }}" /><![endif]-->
  <!--[if lte IE 8]><link rel="stylesheet" href="{{ "assets/css/ie8.css" | relative_url }}" /><![endif]-->
  <script>var clicky_site_ids = clicky_site_ids || []; clicky_site_ids.push(101242331);</script>
  <script async src="//static.getclicky.com/js"></script>

  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$']]}});
    </script>
    <script type="text/javascript"
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
    </script>
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    <script type='text/javascript' src='https://platform-api.sharethis.com/js/sharethis.js#property=5e7f382465b3620019f35d47&product=sticky-share-buttons&cms=website' async='async'></script>
    </header>
    <body>
        {%app_entry%}

      <div style="padding-top:0em;">
        <div id="web_body"; style="padding-top:0em;">
        <div id="web_content;">
          <p>In this figure we compare the current number of COVID-19 fatalities to date shown with color dots, 
          with the multiple projections drawn from the posterior predictive distribution; these projections are 
          shown as faint solid lines (note that the more lines we have the more likely that path will be). 
          We have defined the zeroth time for each country to the day they announced their first fatality record. 
          The vertical grid lines represent important events that may have affected the growth rate such as the 
          separate lockdowns (LD) applied by China, Italy, and Spain. Note that all curves have been drawn from a 
          <a data-scroll href="#logistic">logistic model</a> and predict the # fatalities (N) for each country 
          analyzed here.</p>
        </div>
          <h1>Exponential & logistic growth</h1>

        <div id="web_content";>
          <p>A population will grow its size according to a growth rate. In the case of exponential growth, this rate stays the same regardless of the population size, inducing the population to grow faster and faster as it gets larger, without an end.</p>
          <ul>
            <li>In nature, populations can only grow exponentially during some period, 
            but inevitably the growth rate will ultimately be limited for example by 
            the resource availability.</li>
            <li>In logistic growth, the population growth rate gets smaller and smaller 
            as population size approaches a maximum. This maximum is, in essence, a 
            product of overpopulation limiting the population's resources.</li>
            <li>Exponential growth produces a J-shaped curve, while logistic growth 
            produces an S-shaped curve.</li>
            <li>When we read about bending the curve we are talking about using a l
            ogarithmic scale to plot the data, in that case, that J-shaped curve becomes 
            a straight line. The moment when this straight line bends downwards we 
            start seeing the limiting factors and we are close to the center of the 
            S-shaped curve, which in this case looks like an inverse J-shape. 
            Remember this is only a matter of how the data is plotted or shown, 
            it does not affect the data itself.</li>
          </ul>        
          
          <h5>*Text from: <a href="https://www.khanacademy.org/science/biology/ecology/population-growth-and-regulation/a/exponential-logistic-growth">khanacademy</a>*</h5> 
          
          <p>In theory, any kind of organism could take over the Earth just by reproducing. 
          For instance, imagine that we started with a single pair of male and female rabbits. 
          If these rabbits and their descendants reproduced at top speed "like bunnies"
          for 777 years, without any deaths, we would have enough rabbits to cover the 
          entire state of Rhode Island. And that's not even so impressive – if we used 
          E. coli bacteria instead, we could start with just one bacterium and have 
          enough bacteria to cover the Earth with a 111-foot layer in just 36 hours!</p>

          <p>As you've probably noticed, there isn't a 111-foot layer of bacteria covering 
          the entire Earth (at least, not at my house), nor have bunnies taken possession 
          of Rhode Island. Why, then, don't we see these populations getting as big as they 
          theoretically could? E. coli, rabbits, and all living organisms need specific resources, 
          such as nutrients and suitable environments, in order to survive and reproduce. 
          These resources aren’t unlimited, and a population can only reach a size that match 
          the availability of resources in its local environment.</p>

          <p>Population ecologists use a variety of mathematical methods to model population 
          dynamics (how populations change in size and composition over time). Some of these 
          models represent growth without environmental constraints, while others include "ceilings" 
          determined by limited resources. Mathematical models of populations can be used to accurately 
          describe changes occurring in a population and, importantly, to predict future changes.</p>    
          
          <img src="./assets/cartoon_exp_log.png" alt="" width="100%">
          
          <p>*end of the <a href="https://www.khanacademy.org/science/biology/ecology/population-growth-and-regulation/a/exponential-logistic-growth">khanacademy</a> citation*</p>

          <p>The figure above shows how the logistic and exponential models are constructed; 
          to underestand them better you can watch the video  
          
          <a data-scroll href="#bazinga">"Exponential growth and epidemics"</a> bellow.</p> 
          
          <p>After reading this text it should be obvious to us that the growth of the 
          virus cannot be exponential indefinitely but it has to flatten at some point. 
          One of these functions is the logistic model, used here to predict the number of deaths. </p>
          </div>


          <h1>The logistic function:</h1>
                  <div id="web_content"> 
          <p>If we solve the equation on the right of the previous figure, we obtain the logistic 
          function. A logistic function or logistic curve is S-shaped. This type of curve is known 
          as a sigmoid and its equation is as follows:</p>
          
          $$N(t) = K/(1 + e^{-r(t-t_0)}).$$
          
          <ul> 
            <li> $e$ = the natural logarithm base; also known as Euler's number,</li>
            <li> $t_0$ = the $t$-value of the sigmoid where the rate starts to decrease, the midpoint 
            of its evolution and the 'inflexion point' of the sigmoid's curve.</li>
            <li> $K$ =the curve's maximum value; in this case the maximum number of deaths.</li>
            <li> $r$ = the logistic growth rate or steepness of the curve</li>
          </ul>   
                </div>
          <h1>Simulating possible future scenarios:</h1>
                  <div id="web_content">
          <p>The logistic model defined above and a nonnegative binomial distribution as likelihood, 
          to obtain the posterior predictive distribution of our model; from which we will sample to 
          generate new data based on our estimated posteriors. Please do not get disturbed by this, 
          if you want to have a rough idea of the concept behind all this go lower to the video title 
          "The Bayesian Trap" by Veritasium.
          The figures show, considering this dataset and our model, the predicted evolution of the curves 
          that are expected to be observed. Note that the predictions have the uncertainty into account. 
          Meaning that in the cases where few data points are available the uncertainty grows i.e. 
          the spam of the predictions.
          In short, the figures show that given the data and our model, what evolutions are expected 
          to be observed. Note that the predictions have the uncertainty into account. This implies 
          that for the cases where few data points are available this uncertainty grows.</p>      
      </div>
          <h1>Gompertz curve model:</h1>
                    <div id="web_content">
          <p>This curve is an alternative model that could be taken at this point as upper bounds, 
          we have realized that the logistic model tends to fit the inflection point close to the end of 
          the available data, therefore giving most likely a lower bound prediction. We are not going to 
          discuss the origins but simply mention that this curve is a sigmoid function.</p>
          
          <p>Examples of uses for Gompertz curves include:</p>
          
          <ul>
            <li>Modelling of growth of tumors</li>
            <li>Modelling market impact in finance</li>
            <li>Detailing population growth in animals of prey, with regard to predator-prey relationships</li>
            <li>Examining disease spread</li>
            <li>Modelling bacterial cells within a population</li>    
          </ul>
          
          $$N(t)=N(0)e^{-e^{-a(t-c)}}$$
          
          <p>where:</p>
          
          <ul>
            <li>$N(0)$ is the initial number of cells/organisms when time is zero</li>
            <li>$a$ denotes the rate of growth</li>
            <li>$b, c$ are positive numbers</li>
            <li>$c$ denotes the displacement across in time</li>
          </ul>
      </div>


      <div class="row" style="margin-top: 2.2em"; >
        <div class="column">
          <iframe width="100%" height="100%" src="https://www.youtube.com/embed/Kas0tIxDvrg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </div>
        <div class="column">
          <h2>Exponential growth and epidemics:</h1>
          <p>While this video uses COVID-19 as a motivating example, 
          the main goal is simply a math lesson on exponentials 
          and logistic curves.</p>
          <p>by 3Blue1Brown</p>
          <p>&nbsp</p>
          <a href="https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw" class="button">Go to their channel</a>
        </div>
      </div>


      <div class="row" style="margin-bottom: 2.2em";>
          <div class="column">
            <iframe width="100%" height="100%" src="https://www.youtube.com/embed/R13BD8qKeTg" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
          </div>
          <div class="column">
            <h2>The Bayesian Trap:</h3>
            <p>Bayes' theorem explained with examples and implications for life.</p>
            <p>by Veritasium</p>
            <p>&nbsp</p>
            <a href="https://www.youtube.com/channel/UCHnyfMqiRRG1u-2MsSQLbXA" class="button">Go to their channel</a>
          </div>
      </div>


  </div>
</div>

        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

server = app.server

app.config['suppress_callback_exceptions'] = True

app.layout = html.Div(style={'backgroundColor': backgroundColor1},
    children=[
        html.Div(
            id="header",
                  style={'background-image': 'url("./assets/banner2.png")',
                  'align-items':'center',
                  'display':'flex',
                  'background-attachment':'fixed',
                  'background-position':'center',
                  'background-repeat':'no-repeat',
                  'border-bottom':' 0 !important',
                  'cursor': 'default',
                  'height': '120vh',
                  'margin-left': '-3.25em',
                  'margin-bottom': '-3.25em',
                  'max-height': '32em',
                  'min-height': '22em',
                  'position': 'relative',
                  'top': '-3.25em',
                  'opacity': '1'},
            children=[
                html.Div(
                id="banner", style={'margin-left': '6.25em'},
                children=[
                html.H1(style={'font-weight': 'bold', 'padding-top':' 1em'}, id='web_title', children="COVID-19 Statistics and Research"),
                html.Div(style={'height':'1px', 'width': '80%','display': 'block', 'margin-top':' 0.5em', 'margin-bottom':' 1.3em', 'backgroundColor': background_color_banners}),
                html.H2(style={'font_size': '1em', 'padding-top':' 1em', 'width': '75%'}, id='web_subtitle', children="This website aims to help increase the public’s understanding of the evolving pandemic outbreak."),
                html.H6(id='web_authors', children="Miquel Oliver & Xisco Jimenez Forteza."),  
                html.P(style={'font-weight': 'bold'}, id='web_update_date', children="Latest Update Date" + str(latestDate))]),
            ]),
        ######## ######## ########
        ### Principal Dashboar ###
        ######## ######## ########
        html.Div(id="number-plate",
                  style={'marginTop': '1.5%', 'marginLeft': '1.5%', 'marginRight': '1.5%', 'marginBottom': '.0%'},
                  children=[html.Div(style={'width': '24%', 'display': 'inline-block',
                          'marginRight': '.8%', 'verticalAlign': 'top',
                  'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)', 'backgroundColor': background_color_banners},
                          children=[
                                  html.P(style={'textAlign': 'center', 'font_size': '0.3em', 'color': value[2], 'padding': '.5rem'},
                                                              children=value[0]),
                                  html.H1(style={'textAlign': 'center', 'fontWeight': 'bold', 'color': value[2]},
                                                  children=[value[1]]),
                                  html.P(style={'textAlign': 'center', 'color': value[2], 'padding': '.1rem'},
                                                children=key)
                                    ]) for key, value in dic_first.items()]),

        html.Div(style={'width': '99.2%', 'marginRight': '.8%', 'display': 'inline-block', 'verticalAlign': 'top'},
                              children=[
                                  html.H5(style={'textAlign': 'center', 'backgroundColor': backgroundColor2,
                                                'color': color_fort_map, 'padding': '1rem', 'marginBottom': '0'},
                                                children=''),
                                  dcc.Graph(
                                      id='datatable-interact-map',
                                      figure=fig_map,
                                      style={'height': '500px'},),
                                  dcc.Tabs(
                                      id="tabs-plots", 
                                      value='Cumulative Cases',
                                      parent_className='custom-tabs',
                                      className='custom-tabs-container', 
                                      children=[dcc.Tab(className='custom-tab',
                                                selected_className='custom-tab--selected',
                                                label=key, 
                                                value=key) for key in dic_tabs.keys()]
                                          ),
                                  html.Div(id='tabs-content-plots'),
        ]),
        html.Div(style={'width': '99.2%', 'display': 'inline-block', 'verticalAlign': 'top'},
                  children=[
                      html.H5(style={'textAlign': 'center', 'backgroundColor': backgroundColor2,
                                    'color': color_fort_plot, 'padding': '1rem', 'marginBottom': '0'},
                                    children=' '),
                      dcc.Tabs(
                          id="tabs-table",
                          value='The World',
                          parent_className='custom-tabs',
                          className='custom-tabs-container',
                          children=dcc_tables
                      ),
        ]),

        ######## ######## ##########
        ### Contenido de modelos ###
        ######## ######## ##########
        html.Div(style={'height':'1px', 'width': '100%','display': 'block', 'margin-top':' 0.0em', 'margin-bottom':' 1.3em', 'backgroundColor': background_color_banners}),
        html.Div(style={'width': '99.2%', 'display': 'inline-block', 'verticalAlign': 'top'},
                  children=[
                      html.H1(style={'textAlign': 'left', 'backgroundColor': backgroundColor2,
                                  'color': color_fort_plot, 'padding': '1rem', 'marginBottom': '0','padding-left':' 2.3em'},
                                  children=text_phenom[0]),
                      html.Div(style={'height':'1px', 'align':'center', 'width': '45%','display': 'block', 'margin-top':' 0.0em', 'margin-bottom':' 1.3em', 'backgroundColor': background_color_banners}),
                      dcc.Tabs(id="tabs-phenom-plots", 
                              value='Phenom models',
                              parent_className='custom-tabs',
                              className='custom-tabs-container',
                              children=[dcc.Tab(className='custom-tab',
                                        selected_className='custom-tab--selected',
                                        label=key, 
                                        value=key) for key in dic_phenom_tabs.keys()]
                                  ),
                      html.Div(id='tabs-content-phenom-plots'),
                      html.P(style={'padding-left':' 2.6em','width':'95%'}, children=text_phenom[1]),
        ]),
])

##### CALLBACKS

def get_data_update(vals):

    df_temp_table = df[df['time'] == df['time'].tail(1).values[0]].reset_index().drop(['index'],axis=1)
    df_temp_table, _ = make_dcc_pd(vals[0], df_temp_table.copy())

    location = vals[0]

    zoom = 1
    if location == 'The World':
      location = 'Full Country'
      country = 'Full Country'
      longitude = 6.395626
      latitude = 14.056159 
    else:
      zoom = 3
      country = location
      location = 'Full Country'
      longitude = loc_dic_df[country][0]
      latitude = loc_dic_df[country][1]

    index_value = vals[list_tables.index(vals[0])*2+1]

    if index_value:
      country = df_temp_table.iloc[index_value[0]]['country']
      location = df_temp_table.iloc[index_value[0]]['location']
      longitude = df_temp_table.iloc[index_value[0]]['longitude']
      latitude = df_temp_table.iloc[index_value[0]]['latitude']      
      zoom = 5

    return data_funct(df, location, country)[0], location, country, zoom, longitude, latitude

list_tables = [i.partition('datatable-interact-location-')[2] for i in datatable_interact]

B = [[Input(cn,'derived_virtual_selected_rows'),Input(cn,'selected_row_ids')] for i, cn in enumerate(datatable_interact)]
flatten_inputs = sum(B, [])

@app.callback(
    Output('datatable-interact-map', 'figure'), [Input('tabs-table', 'value')]+flatten_inputs
)

def update_figures(*vals):
    df_temp, country, location, zoom, longitude, latitude = get_data_update(vals)
    df_temp = df[df['time'] == df['time'].tail(1).values[0]].reset_index().drop(['index'],axis=1)
    textList = [df.country, df.location]


    fig_map = create_map(df_temp, longitude, latitude, zoom)

    return fig_map

def update_line_plot(vals, list_cn, name_list, hovertext_list, N, iadd, typexscale, typeyscale, yaxis_title):
    df_temp, country, location, zoom, longitude, latitude = get_data_update(vals)

    # Create empty figure canvas
    fig = go.Figure()
    # Add trace to the figure
    df_temp.sort_values('deaths')
    fig = create_add_trace(fig, df_temp, list_cn = list_cn, name_list = name_list, hovertext_list = hovertext_list, N = N, iadd = iadd)
    fig = figure_top_style(fig, xscale = typexscale, yscale = typeyscale)
    fig = figure_top_style_2(fig, country = country, location = location, yaxis_title=yaxis_title)

    return fig

@app.callback(Output('tabs-content-plots', 'children'),
              [Input('tabs-plots', 'value')])
def render_content(tab):
  if tab != 'Cumulative Cases':
    figure=dic_tabs[tab][1]
    return dcc.Graph(id=dic_tabs[tab][0], style={'height': '300px'}, figure=figure,)

@app.callback(
    Output('figure-dash', 'figure'), [Input('tabs-plots', 'value')]+[Input('tabs-table', 'value')]+flatten_inputs
)
def update_logplot(*vals):
      # Create empty figure canvas
  tab = vals[0]
  vals = vals[1:]

  if tab == 'More than 500 deaths logscale':
    fig = go.Figure()
    th = 500 # threshold for countries to be shown.
    for i, cn in enumerate(df[df['deaths'] > th].reset_index()['country'].unique()):
      df_temp = df[(df['location'] == 'Full Country')&(df['country'] == cn)].groupby('time').sum()
      df_temp = df_temp[df_temp['deaths'] > th].reset_index()
      create_add_trace(fig, df_temp, list_cn = ['deaths'], name_list = [cn+' deaths'], hovertext_list = [cn+' deaths'], N = len(list_of_extended_countries)+1, iadd = i)
      figure_top_style(fig, xscale = "date", yscale = "log")
      figure_top_style_2(fig, 'Log plot', 'countries with more than 500 deaths')
  else:
    fig = update_line_plot(vals, list_cn= dic_tabs[tab][2], name_list = dic_tabs[tab][2], hovertext_list = dic_tabs[tab][2], N = len(dic_tabs[tab][2]), iadd = 0, typexscale = 'date', typeyscale = dic_tabs[tab][-1], yaxis_title = dic_tabs[tab][-2])
  return fig


@app.callback(Output('tabs-content-phenom-plots', 'children'),
              [Input('tabs-phenom-plots', 'value')])
def render_content(tab):
  if tab != 'Phenom models':
    figure=dic_phenom_tabs[tab][1]
    return dcc.Graph(id=dic_phenom_tabs[tab][0], style={'height': '300px'}, figure=figure,)

@app.callback(
    Output('figure-phenom', 'figure'), [Input('tabs-phenom-plots', 'value')]
)
def update_logplot(*vals):
      # Create empty figure canvas
  tab = vals[0]
  fig_model = go.Figure()

  fig_model = create_add_phenom_trace(fig_model, df_phenom, dic_phenom = dic_phenom_tabs[tab])  
  # Add trace to the figure

  figure_top_style(fig_model, xscale = "date", yscale = dic_phenom_tabs[tab][-1],)
  figure_top_style_2(fig_model, 'Phenom', '')
  return fig_model

if __name__ == "__main__":
    app.run_server()
