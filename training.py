#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module implements ANN training

HTTP request handling based on examples by Doug Hellmann on
Python Module of the Week:
    http://pymotw.com/2/BaseHTTPServer/index.html
    
Developed in the scope of the project
  “MATISSE: A machine learning-based forecasting system for shellfish safety”
Funded by the Portuguese Foundation for Science and Technology (DSAIPA/DS/0026/2019).
This code is provided 'as is', with no implied or explicit guarantees.
This code is provided under a CC0 public domain license.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from latexer import read_template
from os.path import isfile
import datetime
from datagen import NS_IN_DAY,read_and_clean,select_locations
from datagen import create_datasets,flatten_specs,balance_dataset
import gather
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Activation, Dropout, Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping

CATALOG = 'catalog.pickle'

def dense_model(layers,input_size,sigmoid=False,output=1):    
    inputs = Input(shape=(input_size,),name='inputs')
    current = inputs
    for l in layers:
        current = Dense(l)(current)
        current = Activation(LeakyReLU())(current)
        current = Dropout(0.5)(current)
    current = Dense(output)(current)
    if sigmoid:
        current = Activation('sigmoid')(current)
    model = Model(inputs = inputs, outputs = current)
    return model

def get_data_table():
    sections = read_template('Report_preprocess/autotext_template.txt')
    toxin = eval(sections['GLOBAL']['toxin'])    
    detection_limit = eval(sections['GLOBAL']['detection_limit'])
    data_file = eval(sections['SELECTION']['data'])
    mintrain = eval(sections['SELECTION']['mintrain'])
    minvalid = eval(sections['SELECTION']['minvalid'])
    data = read_and_clean(data_file,toxin,detection_limit)
    locations = select_locations(data,toxin,mintrain,minvalid)
    data = data.loc[data['assigned'].isin(locations),:]
    return data, sections


def balance_datasets(datasets,y_limit):
    for ds in datasets:
        X, y, t = ds['X'], ds['y'], ds['t']     
        mask = y>y_limit
        ds['X'],ds['y'],ds['t'] = balance_dataset(X,y,mask,t)        
    return datasets

def split_datasets(datasets, valid_date,test_date):
    train = []
    valid = []
    test = []
    for ds in datasets:          
        mask = ds['t']<valid_date
        train.append({'X':ds['X'][mask,:],'y':ds['y'][mask],
                     't':ds['t'][mask],'orig_y':ds['orig_y'][mask]})
        mask = np.logical_and(ds['t']>=valid_date,ds['t']<test_date)
        valid.append({'X':ds['X'][mask,:],'y':ds['y'][mask],
                     't':ds['t'][mask],'orig_y':ds['orig_y'][mask]})
        mask = ds['t']>=test_date
        test.append({'X':ds['X'][mask,:],'y':ds['y'][mask],
                     't':ds['t'][mask],'orig_y':ds['orig_y'][mask]})            
    return train,valid,test

def standardize_datasets(dsets,scales=[]):
    
    for ix,ds in enumerate(dsets):        
        if len(scales)>ix:
            means,stds = scales[ix]
        else:
            means = np.mean(ds['X'],axis=0)
            stds = np.std(ds['X'],axis=0)
            scales.append( (means,stds) )        
        dsets[ix]['X'] = (ds['X']-means)/stds
    return scales           


        
def train_model(train,valid,layers,sizes,
                     patience=30,sigmoid_output=True,
                     loss = 'mean_squared_error',
                     batchsisze=32):
    clear_session()
    if len(train['y'].shape)==2:
        output = train['y'].shape[1]
    else:
        output = 1
    model = dense_model([sizes]*layers,train['X'].shape[1],sigmoid_output,output)
    cb = EarlyStopping(patience=patience,restore_best_weights=True)
    model.compile(Adam(learning_rate=0.0001), loss=loss)
    history = model.fit(x=train['X'],
                        y=train['y'], 
                        batch_size=batchsisze,
                        epochs = 1000,
                        callbacks=[cb],
                        validation_data = (valid['X'],valid['y']),
                        verbose=1,                        
                        )
    return history,model

      

def experiment(section_name):
       
    DATA_PARAMS = ['cooldown_days','packs','num_points',                  
                   'feature_limit','sigmoid_spread','sigmoid_mean',
                  'log_transform',
                  'binary_time',
                  'combinatory_window','include_season','future_limit']
    
    TRAIN_PARAMS = ['layers','sizes','patience','sigmoid_output',
                    'loss','batchsisze']
            
    data, sections = get_data_table()
        
    if 'lat_min' in sections[section_name]:
        data = data.loc[data['latitude']>=eval(sections[section_name]['lat_min']),:]
    if 'lat_max' in sections[section_name]:
        data = data.loc[data['latitude']<=eval(sections[section_name]['lat_max']),:]
    
    
    if 'save_model' in sections[section_name]:
        save_model = eval(sections[section_name]['save_model'])
    else:
        save_model = False
    
        
        
    toxin = eval(sections['GLOBAL']['toxin'])
    limit = eval(sections['GLOBAL']['limit'])
    
    folder = sections[section_name]['folder']
    
    ds_params = {k:eval(sections[section_name][k]) for k in sections[section_name] if k in DATA_PARAMS}
    train_params = {k:eval(sections[section_name][k]) for k in sections[section_name] if k in TRAIN_PARAMS}
    
    ds_file = folder+CATALOG
    if isfile(ds_file):
        print(f'Loading catalog from {ds_file}')
        datasets,ds_specs,train,valid,test,scale,train_specs = pickle.load(open(ds_file,'rb'))
    else:
        print(f'Creating datasets and saving to {ds_file}')
        ds_specs,datasets = create_datasets(data, 'assigned', toxin, limit, ds_params)
        valid_date = pd.to_datetime(sections[section_name]['valid_date']).value/NS_IN_DAY
        test_date = pd.to_datetime(sections[section_name]['test_date']).value/NS_IN_DAY
      
        train,valid, test = split_datasets(datasets, valid_date,test_date)
        
        scale = standardize_datasets(train)
        standardize_datasets(valid,scale)
        standardize_datasets(test,scale)
        
        if 'balance' in sections[section_name]:
            balance_datasets(train,eval(sections[section_name]['balance']))
        
        train_specs = flatten_specs(train_params,{})
        
        with open(ds_file, 'wb') as ofil: 
            pickle.dump([datasets,ds_specs,train,valid,test,scale,train_specs], ofil)
    tot_ds = len(datasets)
    tot_models = len(train_specs)
    print(f'{tot_ds} datasets')        
    
    for ds_ix in range(tot_ds):
        print(f'Dataset {ds_ix+1} of {tot_ds}')
        for tr_ix in range(tot_models):   
            f_name = folder+str(ds_ix)+'_'+str(tr_ix)+'.history'
            print(f'Model {tr_ix+1+(tot_models)*(ds_ix)} of {tot_ds*tot_models}:{f_name}')                
            if isfile(f_name):
                print('Skipped, file exists')
            else:
                res,model = train_model(train[ds_ix],
                                        valid[ds_ix],
                                        **train_specs[tr_ix])
                train_pred = model.predict(train[ds_ix]['X'])
                valid_pred = model.predict(valid[ds_ix]['X'])
                test_pred = model.predict(test[ds_ix]['X'])                
                with open(f_name, 'wb') as ofil: 
                    pickle.dump([res.history,train_pred,valid_pred,test_pred], ofil)
                if save_model:
                    model.save_weights(f_name.replace('.history','.h5'))



    
