#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictions module

Developed in the scope of the project
  “MATISSE: A machine learning-based forecasting system for shellfish safety”
Funded by the Portuguese Foundation for Science and Technology (DSAIPA/DS/0026/2019).
This code is provided 'as is', with no implied or explicit guarantees.
This code is provided under a CC0 public domain license.
"""

import io
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import isfile
import datetime
#from datagen import NS_IN_DAY,read_and_clean,select_locations
from datagen import create_datasets,flatten_specs,balance_dataset
import gather
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Activation, Dropout, Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from plotly.io import write_json
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import htmlconstants as htc
from requests import get
from json import loads

MODEL_TEXT = {'MODEL':'Description'} #model descriptions

ERROR = {'date_too_early':'Date before first point'}


COLORS = {'POINT1':'#a157db'} #set colors for locations
LOC_NAMES = {'POINT1':'This is point 1'}

DATA_PARAMS = ['cooldown_days','packs','num_points',                  
               'feature_limit','sigmoid_spread','sigmoid_mean',
              'log_transform',
              'binary_time',
              'combinatory_window','include_season','future_limit']

TRAIN_PARAMS = ['layers','sizes','patience','sigmoid_output',
                'loss','batchsisze']

NS_IN_DAY = 1e9*3600*24

plt.style.use('seaborn')

def replace_all(text,replacements):
    for fin,rep in replacements:
        text = text.replace(fin,str(rep))
    return text



def read_template(filename):    
    lines = open(filename).readlines()
    parts = {}
    k = None
    for line in lines:
        if line.startswith('#'):
            continue
        if line.startswith('*'):            
            if k is not None:
                parts[k]['text']=''.join(parts[k]['text'])
            k = line[1:].strip()    
            if k.startswith('!'):
                k = k[1:]
                show = False
            else:
                show = True
            parts[k] = {'text':[],'show':show}
        elif line.startswith('>'):            
            ix = line.index(':')
            key,val = line[1:ix],line[ix+1:].strip()
            parts[k][key]=val
        else:            
            parts[k]['text'].append(line)
    if k is not None:
        parts[k]['text']=''.join(parts[k]['text'])
    return parts

def strdate_as_day(date):
    return int(pd.to_datetime(date).value/NS_IN_DAY)

def days_from_dates(dates):
    return dates.astype(np.int64).values/NS_IN_DAY

def dates_from_days(days):
    return pd.to_datetime(np.int64(days*NS_IN_DAY))
        
def dense_model(layers,input_size,sigmoid=False,output=1):    
    inputs = Input(shape=(input_size,),name='inputs')
    current = inputs
    for l in layers:
        current = Dense(l)(current)
        current = Activation(LeakyReLU())(current)
        current = Dropout(0.5)(current,training=True)
    current = Dense(output)(current)
    if sigmoid:
        current = Activation('sigmoid')(current)
    model = Model(inputs = inputs, outputs = current)
    return model


def read_toxin(filename,toxin,detection_limit):    
    df = pd.read_csv(filename)
    df = df.loc[~df[toxin].isna(),:]
    df['date'] = pd.to_datetime(df['date'],errors='coerce')    
    df['zona'] = df['zona'].str.upper()
    df.sort_values('date', inplace=True)
    df.loc[df[toxin]<detection_limit,toxin]=detection_limit
    return df


def read_and_clean(filename,toxin,detection_limit):
    data = read_toxin(filename,toxin,detection_limit)
    data.dropna(subset = ['assigned'],inplace=True)                        
    data = data.groupby(by=['date','assigned','longitude','latitude']).max()[toxin].reset_index()          
    return data


def get_data_table():
    sections = read_template('config.txt')
    toxin = eval(sections['GLOBAL']['toxin'])    
    detection_limit = eval(sections['GLOBAL']['detection_limit'])
    data_file = eval(sections['SELECTION']['data'])
    data = read_and_clean(data_file,toxin,detection_limit)    
    return data, sections


def standardize_datasets(dsets,scales=[]):
    
    for ix,ds in enumerate(dsets):        
        if len(scales)>ix:
            means,stds = scales[ix]
        else:
            means = np.mean(ds['X'],axis=0)
            stds = np.std(ds['X'],axis=0)
            scales.append( (means,stds) )        
        dsets[ix]['X_orig'] = ds['X']
        dsets[ix]['X'] = (ds['X']-means)/stds
        dsets[ix]['scale'] = (means,stds)
    return scales   

def prepare():
    global MODELS
    MODELS = {'JOINT':{} }
    data, sections = get_data_table()
    toxin = eval(sections['GLOBAL']['toxin'])
    limit = eval(sections['GLOBAL']['limit'])
    
    model_sections = [k for k in MODELS]
    
    for section_name in model_sections:
        weights = sections[section_name]['weights']    
        ds_params = {k:eval(sections[section_name][k]) for k in sections[section_name] if k in DATA_PARAMS}
        train_params = {k:eval(sections[section_name][k]) for k in sections[section_name] if k in TRAIN_PARAMS}
        ds_specs,datasets = create_datasets(data, 'assigned', toxin, limit, ds_params)        
        scale = standardize_datasets(datasets)        
        dataset = datasets[0]
        train_specs = flatten_specs(train_params,{})[0]
        layers = [train_specs['sizes']]*train_specs['layers']
        input_size = dataset['X'].shape[1]        
        model = dense_model(layers, input_size,train_specs['sigmoid_output'],dataset['y'].shape[1])        
        model.load_weights(weights)
        MODELS[section_name]['data'] = dataset
        MODELS[section_name]['model'] = model
        MODELS[section_name]['scale'] = scale[0]
        MODELS[section_name]['ds_specs'] = ds_specs[0]
        MODELS[section_name]['train_params'] = train_specs
    
    MODELS['data'] = data
     
def sigmoid(y,sigmoid_mean,sigmoid_spread):
    return 1 / (1 + np.exp((sigmoid_mean-y)/sigmoid_spread))
    
def invert_sigmoid(y,sigmoid_mean,sigmoid_spread):
    return sigmoid_mean - np.log(1/y-1)*sigmoid_spread        
     
def get_preds(preds,ds_specs):
    print(ds_specs)
    if 'sigmoid_mean' in ds_specs:                        
        mean = ds_specs['sigmoid_mean']
        spread = ds_specs['sigmoid_spread']
        eps = 0.00001
        preds[preds<eps] = eps
        preds[preds>1-eps] = 1-eps     
        preds = invert_sigmoid(preds, mean, spread)   
        preds[preds<0] = 0                            
    return preds
     
   
def plot_model(m_name,ix,file_name):
    result = {}
    
    model = MODELS[m_name]['model']
    X = MODELS[m_name]['data']['X'][[ix],:]
    X_orig = MODELS[m_name]['data']['X_orig'][ix,:]
    target = MODELS[m_name]['data']['orig_y'][ix,:]
    date = dates_from_days(MODELS[m_name]['data']['t'][ix])
    result['date'] = date.strftime("%Y-%m-%d")
    result['preds'] = {}
    preds = np.array([model.predict(X,verbose=0)[0] for _ in range(50)])
    preds = get_preds(preds,MODELS[m_name]['ds_specs'])
        
    fig = plt.figure(figsize=(10,5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    x_axis = np.linspace(0, 60, 61)
    num_points = MODELS[m_name]['ds_specs']['num_points']
    days_off = preds.shape[1]*num_points
    for col in range(preds.shape[1]):
        name = MODELS[m_name]['ds_specs']['packs'][0]['targets'][col]
        result['preds'][name] = {}
        kde = KernelDensity(bandwidth=3)
        kde.fit(preds[:,col].reshape(-1,1))
        
        result['preds'][name]['pred_mean'] = np.mean(preds[:,col])
        result['preds'][name]['pred_std'] = np.std(preds[:,col])
        
        kde_prob =  np.exp(kde.score_samples(x_axis.reshape((-1,1))))
        
        ax2.plot(x_axis,kde_prob,label = LOC_NAMES[name],color=COLORS[name])
        
        days = -X_orig[days_off+col*num_points:days_off+col*num_points+num_points]
        concs = X_orig[col*num_points:col*num_points+num_points]
        result['preds'][name]['days'] = days
        result['preds'][name]['concs'] = concs
        
        ax1.plot(days,concs,'-o',label=None,color=COLORS[name])
    
    plt.xlim(-60,60)
    ax1.set_ylim(0,200)
    plt.grid(False)
    ax1.set_ylabel('Conc. DSP')
    ax2.set_ylabel('Probabilidade por dia')
    
    plt.legend()
    plt.savefig(file_name,dpi=200,bbox_inches='tight')
    plt.close()
    return result

def plot_model_from_date(m_name,date,file_name):
    point = strdate_as_day(date)
    #print(point)
    times = MODELS[m_name]['data']['t']
    times = times[times<=point]    
    if len(times) == 0:
        return ERROR['date_too_early']
    
    index = len(times)-1
    #print(times[-1],dates_from_days(times[-1]))
    res = plot_model(m_name,index,file_name)
    return res
    

def create_plot_table(plot_data):
    print(plot_data)
    table = {'Location':[],'Time to Contamination':[]}
    for p in plot_data['preds']:        
        table['Location'].append(LOC_NAMES[p])
        p = plot_data['preds'][p]
        table['Time to Contamination'].append(f'{p["pred_mean"]:.0f}+-{p["pred_std"]:.0f} days')
    return pd.DataFrame(table).to_html(index=False)

def create_plot_div(m_name,date):
    plot_file = htc.PLOT_FOLDER+m_name+'.png'
    plot_data = plot_model_from_date(m_name,date,plot_file)
    table = create_plot_table(plot_data)
    replacements = {'MODEL':MODEL_TEXT[m_name],
                    'DATE':plot_data['date'],
                    'PLOTFILE':plot_file[len(htc.HTML_FOLDER):],
                    'TABLE':table}
    return htc.process_html(htc.MODEL_DIV_TEMPLATE,replacements)
    
    
def model_page(query):
    print(query)        
    if not 'date' in query:
        return htc.process_html(htc.HTML_MODEL,{}) 
    else:
        div = create_plot_div('JOINT',query['date'][0])
        
        div = '<section>\n'+div+'\n</section>\n'
        
        return htc.process_html(htc.HTML_MODEL,{'<!--SECTIONS-->':div}) 
    
def current_page():
    
    codes = ['L7c2','POR2','LAG']
    src = get('https://www.ipma.pt/pt/bivalves/index.jsp').text 
    src = '{'+src.split('var data = {')[1].split(';\n\n')[0]
    data = loads(src)    
    
    geoms = data['objects']['name']['geometries']
    
    table = {'Location':[],'Status':[],'Species (common)':[],'Species (scientific)':[]}
    for g in geoms:
        g = g['properties']
        if g['code'] in codes:
            # if g['code'] =='L7c2':
            #     print(g)
            #     aaa
            for st,lbl in [('open','Open'),('close','Closed')]:
                for species in g['interdictions'][st]:
                    table['Location'].append(g['name'])
                    table['Status'].append(lbl)
                    table['Species (common)'].append(species['specie_c'])
                    table['Species (scientific)'].append(species['specie_s'])
                    
            
    table = pd.DataFrame(table).to_html(index=False)
    table = table.replace('<td>Open</td>','<td class="status-open">Open</td>')
    table = table.replace('<td>Closed</td>','<td class="status-closed">Closed</td>')
    
    
    return htc.process_html(htc.HTML_CURRENT,{'<!--TABLE-->':table}) 
    
    
if __name__ == '__main__':
    
    #tests here
    
