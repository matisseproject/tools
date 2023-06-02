#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed in the scope of the project
  “MATISSE: A machine learning-based forecasting system for shellfish safety”
Funded by the Portuguese Foundation for Science and Technology (DSAIPA/DS/0026/2019).
This code is provided 'as is', with no implied or explicit guarantees.
This code is provided under a CC0 public domain license.
"""
import pandas as pd
import numpy as np
from itertools import combinations

NS_IN_DAY = 1e9*3600*24

def days_from_dates(dates):
    return dates.astype(np.int64).values/NS_IN_DAY

def dates_from_days(days):
    return pd.to_datetime(np.int64(days*NS_IN_DAY))


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

def get_points(data,toxin):
    d_array = days_from_dates(data['date'])
    t_array = data[toxin].values
    sort = np.argsort(d_array)
    d_array = d_array[sort]
    t_array = t_array[sort]
    return np.column_stack((d_array,t_array))

def get_day_windows(mat,days):    
    st,en = 0,0
    windows = []
    try:
        while mat[en,0]-mat[st,0]<days:        
            en+=1
    except IndexError:
        return windows
    en -=1
    while en<mat.shape[0]:
        if en-st>1:
            windows.append(mat[st:en+1,:])            
        while mat[en,0]-mat[st,0]>days:
            st+=1            
        en+=1
    return windows

def get_all_windows(tables,days,limit=None):
    """filters if all but last point is over limit"""
    wind_tables = {}
    for t in tables:
        winds = get_day_windows(tables[t],days)        
        if limit is not None:
            winds = [w for w in winds if np.max(w[:,1])<limit]
        wind_tables[t] = winds
    return wind_tables

def get_point_distances(mat,limit):
    ix = mat.shape[0]-1
    dist = 0
    dists = np.zeros(mat.shape[0])
    while ix>=0:
        if mat[ix,1]<limit:
            dist += 1
        else:
            dist = 0
        dists[ix] = dist
        ix-=1
    return dists

def get_day_distances(mat,limit):
    ix = mat.shape[0]-1
    last_day = mat[-1,0]+1    
    dists = np.zeros(mat.shape[0])
    while ix>=0:
        if mat[ix,1]<limit:
            dists[ix] = last_day-mat[ix,0]
        else:
            last_day = mat[ix,0]
        ix-=1
    return dists

def get_points_by_field(data,field,toxin):
    groups = data[field].unique()
    g_tables = {}        
    for g in groups:
        sel = data.loc[data[field]==g,:]        
        g_tables[g]= get_points(sel,toxin)        
    return g_tables        


def get_val_date_dists(data,field,toxin,limit):
    tables = get_points_by_field(data,field,toxin)
    d_tables ={}
    for k in tables:  
        points = tables[k]        
        sel = np.zeros((points.shape[0],4))
        sel[:,:2] = points
        sel[:,2] = get_day_distances(points,limit)        
        inv = points.copy()
        inv[:,0] = - (inv[:,0]-inv[-1,0])
        sel[:,3] = get_day_distances(inv[::-1,:],limit)[::-1]
        d_tables[k] = sel
    return d_tables


def get_val_date_dist_mats(data,field,toxin,limit):
    d_tables = get_val_date_dists(data,field,toxin,limit)
    mat = [d_tables[k] for k in d_tables]
    return np.vstack(mat)        


def get_data_stats(data,toxin,limit,days):
    locs = data['assigned'].unique()
    years = pd.DatetimeIndex(data['date']).year
    test = data.loc[years==2022,:]
    valid = data.loc[years==2021,:]
    train = data.loc[years<2021,:]
    tables = []
    for df in [train,valid,test]:
        tables.append(get_points_by_field(df,'assigned',toxin))
    windows = [get_all_windows(tab,days,limit=limit) for tab in tables]    
    stats = {k:[] for k in ['Location','Train s.','Valid s.','Test s.','Train w.','Valid w.','Test w.','Points']}
    for l in locs:
        stats['Location'].append(l)
        points = []
        for ps,ws,col in zip(tables,windows,['Train','Valid','Test']):
            if l in ps:
                stats[col+' s.'].append(ps[l].shape[0])
            else:
                stats[col+' s.'].append(0)
            if l in ws:
                stats[col+' w.'].append(len(ws[l]))
                points.extend([len(w) for w in ws[l]])
            else:
                stats[col+' w.'].append(0)                                
        if points!=[]:
            stats['Points'].append(f'{np.mean(points):.2f}')
        else:
            stats['Points'].append('-')
            
    return tables,windows,pd.DataFrame(stats)
    
def select_locations(data,toxin,mintrain,minvalid):
    
    #currently not used
    limit = 160
    days = 30
    
    samples, windows, stats = get_data_stats(data,toxin,limit,days)
    stats = stats.loc[(stats['Train s.']>=mintrain) & (stats['Valid s.']>=minvalid),:]
    return stats['Location'].values

    
def balance_dataset(X,y,mask,t=None):
    """replicate minority class"""
    classes, counts = np.unique(mask,return_counts=True)
    majority = np.max(counts)
    balanced_X = [X]
    balanced_y = [y]
    balanced_t = [t]
    for cl,cc in zip(classes,counts):
        missing = majority-cc
        if missing>0:
            sel_mask = y==cl 
            sel_X = X[sel_mask,:]
            sel_y = y[sel_mask]
            replicas = np.random.randint(0,len(sel_y),size=majority-cc)
            balanced_X.append(sel_X[replicas,:])
            balanced_y.append(sel_y[replicas])
            
            if t is not None:
                sel_t = t[sel_mask]
                balanced_t.append(sel_t[replicas])            
            
    if t is None:
        return np.concatenate(balanced_X,axis=0),np.concatenate(balanced_y,axis=0)
    else:
        return np.concatenate(balanced_X,axis=0),np.concatenate(balanced_y,axis=0),np.concatenate(balanced_t,axis=0)

def sigmoid(y,sigmoid_mean,sigmoid_spread):
    return 1 / (1 + np.exp((sigmoid_mean-y)/sigmoid_spread))
    
def invert_sigmoid(y,sigmoid_mean,sigmoid_spread):
    return sigmoid_mean - np.log(1/y-1)*sigmoid_spread

def index_points(points):
    p_ix = {}
    for p in points:        
        mat = points[p]
        p_ix[p] = {'index':{},'last':mat[-1,0],'first':mat[0,0],'last_ix':mat.shape[0]-1}
        date = mat[0,0]
        for row in range(1,mat.shape[0]):
            for d in range(int(date),int(mat[row,0])):
                p_ix[p]['index'][d]=row-1
            date = mat[row,0]
    return p_ix

def get_last_index(date,point_index):
    if date>=point_index['last']:
        return point_index['last_ix']
    elif date<point_index['first']:
        return -1
    else:
        return point_index['index'][date]
            
      
def default_pack_list(points):
    return [{'targets':[p],'features':[p]} for p in points]        
      
def create_dataset(points, point_indexes, 
                   packs= 'default', 
                   num_points=5,
                   cooldown_days=0,
                   feature_limit = None,                   
                   sigmoid_spread = None, sigmoid_mean = None,
                   log_transform = False,
                   binary_time = None,
                   combinatory_window = None,
                   include_season = False,
                   future_limit=None):
    """returns feature matrix, time to over limit, day of over limit
       receives time dist matrices, with date, toxin, days to next cont, days from last cont    
    """   
    
    def get_min_max_date(point_indexes,targets):
        firsts = [point_indexes[p]['first']-1 for p in targets]
        lasts = [point_indexes[p]['last']-1 for p in targets]
        return int(np.max(firsts)),int(np.min(lasts))
    
    def get_indexes(day,features,targets,point_indexes):
        feature_indexes = [get_last_index(day,point_indexes[f]) for f in features]
        target_indexes = [get_last_index(day,point_indexes[f])+1 for f in targets]
        return feature_indexes,target_indexes
    
    def get_matrices():        
        if np.min(feat_ixs)<num_points:
            return None,None
        feat_points = [points[f_name][f_ix-num_points:f_ix,:] for f_name,f_ix in zip(features,feat_ixs)]
        feat_mat = np.concatenate(feat_points,axis=0)        
        target_points = [points[t_name][t_ix,:] for t_name,t_ix in zip(targets,targ_ixs)]
        target_mat = np.vstack(target_points)
        
        if np.min(feat_mat[:,3])<cooldown_days:
            return None,None
        
        if feature_limit is not None and np.max(feat_mat[target_features,1])>feature_limit:
            return None,None
        
        if future_limit is not None and np.min(target_mat[:,2])>future_limit:
            return None,None
        
        return feat_mat,target_mat


    def get_packs():        
        if type(packs) is list:
            return packs
        elif packs == 'default':
            return default_pack_list(points)
        else:
            raise NotImplementedError(f'packs {packs} not implemented')
        
    
    X = []
    y = []
    days = []
    
    if combinatory_window is not None:
        raise NotImplementedError('Combinatory window not implemented')
    packs = get_packs()
    
        
    for pack in packs:
        features = pack['features']
        targets = pack['targets']
        target_features = [ix for ix,feat in enumerate(features) if feat in targets]               
        min_date,max_date = get_min_max_date(point_indexes,targets)        
        for day in range(min_date,max_date+1):
            feat_ixs,targ_ixs = get_indexes(day,features,targets,point_indexes)
            feat_mat,targ_mat =get_matrices()            
            if feat_mat is not None:
                x_row = list(feat_mat[:,1])
                x_row.extend(list(day-feat_mat[:,0]))
                if include_season:
                    x_row.append(np.sin(day/365*2*np.pi))
                    x_row.append(np.cos(day/365*2*np.pi))
                X.append(x_row)                    
                # target point is first one after day
                y.append(targ_mat[:,2]+targ_mat[:,0]-day)                
                days.append(day)                
    
    X, y, days = np.array(X),np.array(y), np.array(days)
    
    orig_y = y.copy()    
    
    if sigmoid_mean is not None:
        y = 1 / (1 + np.exp((sigmoid_mean-y)/sigmoid_spread))
    if log_transform:
        y = np.log(y)
    if binary_time is not None:
        mask = y<=binary_time        
        y[mask] = 1
        y[~mask] = 0
    return {'X':X, 'y':y, 't':days,'orig_y':orig_y}


def flatten_specs(range_specs,point_specs):
    if len(range_specs) == 0:
        return [ point_specs.copy() ]
    keys = list(range_specs.keys())
    first_key = next(iter(range_specs))
    next_specs = {k:range_specs[k] for k in keys[1:]}
    curr_spec = range_specs[first_key]
    results = []
    if type(curr_spec) is list:            
        for val in range_specs[first_key]:
            point_specs[first_key] = val
            results.extend(flatten_specs(next_specs,point_specs))
    else:
        point_specs[first_key] = range_specs[first_key]
        results.extend(flatten_specs(next_specs,point_specs))
    return results
    
def create_datasets(data, loc_field, toxin, limit, specs):    
    locs = get_val_date_dists(data,loc_field,toxin,limit)    
    day_ixs = index_points(locs)
    flat_specs = flatten_specs(specs,{})
    datasets = []
    for ix,fs in enumerate(flat_specs):
        print(f'Creating {ix+1} of {len(flat_specs)}')
        datasets.append(create_dataset(locs,day_ixs,**fs))
    return flat_specs,datasets
    
        

