#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed in the scope of the project
  “MATISSE: A machine learning-based forecasting system for shellfish safety”
Funded by the Portuguese Foundation for Science and Technology (DSAIPA/DS/0026/2019).
This code is provided 'as is', with no implied or explicit guarantees.
This code is provided under a CC0 public domain license.
"""
from requests import get
from os.path import isfile
import os


from glob import glob
#import tabula
import pandas as pd
from scipy import stats as st
import numpy as np



IPMA_URLS =  [('https://www.ipma.pt/pt/bivalves/biotox/','data/biotox_pdfs/',),
              ('https://www.ipma.pt/pt/bivalves/fito/','data/fito_pdfs/'),
              ('https://www.ipma.pt/pt/bivalves/metais/','data/metais_pdfs/'),
              ('https://www.ipma.pt/pt/bivalves/micro/','data/micro_pdfs/'),
             ]

LEGAL_LIMITS = {'DSP':160,'ASP':20,'PSP':800}

TXT_FOLDERS = ['data/biotox_txt','data/fito_txt','','']

INTERMEDIATE = ['data/biotoxin.csv','data/fito.csv','','']
FINAL =  ['data/clean_biotox.csv','data/clean_fito.csv','','']

LOC_COLS = ['Production_Area','zona','local','Sample_Point']

IPMA_LOCS = 'config/coordinates.csv'
ADDED_LOCS = 'config/added_coordinates.csv'
LOC_FORCED = 'config/loc_forced.txt'
LOC_IPMA_UPDATE = 'config/loc_ipma_update.txt'



BIOTOX_REPLACEMENTS = 'config/biotox_replacements.txt'
FITO_REPLACEMENTS = 'config/fito_replacements.txt'
TOXIN_REPORT = 'logs/toxin_report.txt'
FITO_REPORT = 'logs/fito_report.txt'


MATCHFILE = 'logs/matched.txt'
UNMATCHFILE = 'logs/unmatched.txt'

NONCOINCIDENTS = 'logs/noncoincidents.csv'

ZONESIZES='data/zonesizes.csv'
AGGREGATED = 'data/aggregated.csv'
AGGREGATE_DISTANCE = 10

BIOTOX_ASSIGNED = 'data/biotox_assigned.csv'
BIOTOX_AGGREGATED = 'data/biotox_aggregated.csv'

def retrieve(url,folder):
    main = get(url)
    parts = main.text.split("<a href='")
    links = [p.split("'")[0] for p in parts if '.pdf' in p]
    for l in links:
        fn = l.split('/')[-1]    
        if not isfile(folder+fn):
            print('Retrieving',fn)
            r = get('https://www.ipma.pt'+l)
            with open(folder+fn, 'wb') as ofil:
                ofil.write(r.content)
        else:
            print('Skipped',fn)

def retrieve_all():
    for url,folder in IPMA_URLS:
        retrieve(url,folder)


def compile_biotox():
    files = glob('biotox_csvs/*.csv')
    heads = []
    bodies = [] 
    cols = []
    bads = []
    for fil in files:
        lines = open(fil).readlines()
        heads.append('**** '+fil)
        for l in lines:
            if ' ' in l[:6]:
                l = l[:6].replace(' ',',')+l[6:]
            try:
                int(float(l.split(',')[0]))
                bodies.append(l)   
                cols.append(l.count(','))                
                if l.count(',')==5:
                    bads.append(l)
            except ValueError:
                heads.append(l)
    with open('biotox_bodies.txt','w') as ofil:
        ofil.writelines(bodies)
    with open('biotox_heads.txt','w') as ofil:
        ofil.writelines(heads)
    print(np.unique(cols,return_counts = True))
    print(bads)
        
            
    

def pdf_to_txt(in_folder,out_folder):
    """does not work well"""
    files = glob(in_folder+'/*.pdf')    
    for fil in files:        
        out_name = os.path.join(out_folder,os.path.basename(fil).split('_')[0]+'.txt')        
        if isfile(out_name):
            print(out_name,'skipped')
        else:
            os.system(f'pdftotext -layout "{fil}" "{out_name}"')      
            
        
def find_headers(lines,first_row):
    
    def get_coords(line):
        # print(line)
        # print(first_row)
        # aaa
        coords = []
        r_ix = 0
        for ix in range(len(line)):
            if line[ix:ix+len(first_row[r_ix])] == first_row[r_ix]:
                coords.append(ix + len(first_row[r_ix])//2)
                r_ix += 1
            if r_ix>=len(first_row):
                break
        return coords
    
    def closest_coord(coord):
        dist = 1e9        
        for ix,c in enumerate(coords):
            d = abs(c-coord)
            if d<dist:
                dist = d
                closest = ix
        return closest
    
    if ' Capitania' in ''.join(lines):
        headers = ['num','espécie','local','zona','capitania','data']
    else:
        headers = ['num','espécie','local','zona','data']
    toxin_headers = []
    while len(headers)+len(toxin_headers)<len(first_row):
        toxin_headers.append([])
    toxins = ['AO','AZA','YTX','AD','AE','STX','DTXs','ASP','PTX']
    header_lines = []
    
    for ix,l in enumerate(lines):
        if l.strip().startswith(first_row[0]):
            break
    line = lines[ix]
    coords = get_coords(line)[len(headers):]
    for l_ix in range(ix-8,ix):
        for t in toxins:
            if t in lines[l_ix]:
                header_lines.append(l_ix)
                p = lines[l_ix].find(t)
                col = closest_coord(p)                
                toxin_headers[col].append(t)
    header_lines =[lines[i] for i in list(set(header_lines))]
    toxin_headers = [list(set(h)) for h in toxin_headers]
    for h in toxin_headers:
        h.sort()
    toxin_headers = ['+'.join(h) for h in toxin_headers]
    return headers+toxin_headers,header_lines
    
    
            
def find_columns(lines):
    part_lens = []
    parts = []
    ixs = []
    
    for l in lines:
        ps = [s.strip() for s in l.split('  ') if s!='']
        if ' ' in ps[0]:
            t = ps[0].split(' ')
            n1 = t[0]
            n2 = ' '.join(t[1:])
            ps = [n1,n2]+ps[1:]
        if ps[0].isnumeric():
            ixs.append(ps[0])
        part_lens.append(len(ps))
        parts.append(ps)
        
    cols = st.mode(part_lens)[0][0]
    print(cols)
    rows = [ps for ps,pl in zip(parts,part_lens) if pl == cols and ps[0].isnumeric()]
    
   
    dups = []
    for row in rows:
        try:
            ixs.remove(row[0])
        except ValueError:
            dups.append(row[0])
    col_names,col_lines = find_headers(lines,rows[0])
    data = {c:[] for c in col_names}
    for row in rows:        
        for cn,val in zip(col_names,row):
            data[cn].append(val)
    for d in data:
        print(d,len(data[d]))
    data = pd.DataFrame(data)
    print(data)
    return cols,len(rows), rows,ixs,dups,col_names,col_lines,data
        

def get_replacements(repfile):
    lines = open(repfile).readlines()
    reps = {}    
    curr = None
    for l in lines:
        l = l.strip()
        if l.startswith('#'):
            curr = l.strip()[1:]
            reps[curr] = {'rep':[],'add':[]}
        elif l.startswith('ADD_COLS'):
            ps = l.split(':')
            reps[curr]['add'].append( (int(ps[1]),int(ps[2]),ps[3] ))            
        elif l!='':            
            l = l.replace('*nl*','\n')
            l = l.replace('*sp*',' ')
            reps[curr]['rep'].append(l.split('|'))
    return reps
    

def check_report(report = TOXIN_REPORT):
    text = open(report).read()
    checks = {'***VAZIO***':'At least one empty column label.',
              '***INCONSISTENTE***':'Impossible combination of toxins in a columns label.',
              'Missing:':'At least one data row could not be parsed.',
              'Duplicates:':'Repeated rows found.',
              }    
    return checks, {k:text.count(k) for k in checks}

            
def process_toxin(folder):    
    replacements = get_replacements(BIOTOX_REPLACEMENTS)
    rep = []
    fils = glob(os.path.join(folder,'*.txt'))
    frames =  []
    col_sets = {}
    for f in fils:
        f_name = os.path.basename(f)
        if f_name in replacements:
            txt = open(f).read()
            for find,replace in replacements[f_name]['rep']:
                txt = txt.replace(find,replace)
            lines = [l+'\n' for l in txt.split('\n')]
            for min_l,max_l,add in replacements[f_name]['add']:
                for ix,l in enumerate(lines):
                    if min_l<=len(l)<=max_l:
                        lines[ix] = l.strip()+add+'\n'
            # with open(os.path.join('modified',f_name),'w') as ofil:
            #     ofil.write(''.join(lines))            
        else:
            lines = open(f).readlines()
        cols, nrows, rows,missing,dups,col_names,col_lines,df = find_columns(lines)        
        
        try:
            df.drop('capitania',axis=1,inplace=True)
        except KeyError:
            pass
        frames.append(df)
        
        t_col = tuple(df.columns[5:])
        if t_col in col_sets:
            col_sets[t_col].append(f_name)
        else:
            col_sets[t_col]= [f_name]                
        
        df.insert(0,'file',f_name)
        rep.append(f_name+'\n'+f'    {cols} {nrows}')
        rep.append(''.join(col_lines))
        if '' in col_names:
            rep.append(f'    {col_names}, ***VAZIO***')
        elif 'AO+ASP+DTXs' in col_names:
            rep.append(f'    {col_names}, ***INCONSISTENTE***')
        else:
            rep.append(f'    {col_names}')
        rep.append(f'    {rows[0]}')
        if missing!=[]:
            rep.append(f'    Missing: {missing}')
        if dups!=[]:
            rep.append(f'    Duplicates: {dups}')
        
        rep.append('\n')
    with open(TOXIN_REPORT,'w') as ofil:
        ofil.write('\n'.join(rep))
    data = pd.concat(frames,axis=0)
    data.to_csv(INTERMEDIATE[0],index=False)
    
    with open('logs/col_sets.txt','w') as ofil:
        for k in col_sets:
            ofil.write(f'{k}'+'\n')
            for f in col_sets[k]:
                ofil.write(f'    {f}'+'\n')

def fito_headers(lines,first_row):
    
    def get_coords(line):
        # print(line)
        # print(first_row)
        # aaa
        coords = []
        r_ix = 0
        for ix in range(len(line)):
            if line[ix:ix+len(first_row[r_ix])] == first_row[r_ix]:
                coords.append(ix + len(first_row[r_ix])//2)
                r_ix += 1
            if r_ix>=len(first_row):
                break
        return coords
    
    def closest_coord(coord):
        dist = 1e9        
        for ix,c in enumerate(coords):
            d = abs(c-coord)
            if d<dist:
                dist = d
                closest = ix
        return closest
    
    targets = ['colheita','Zona',' de ASP', 'de DSP']
    
    # if ' Capitania' in ''.join(lines):
    #     headers = ['num','espécie','local','zona','capitania','data']
    # else:
    #     headers = ['num','espécie','local','zona','data']
    # toxin_headers = []
    # while len(headers)+len(toxin_headers)<len(first_row):
    #     toxin_headers.append([])
    # toxins = ['AO','AZA','YTX','AD','AE','STX','DTXs','ASP','PTX']
    header_lines = []
    
    for ix,l in enumerate(lines):
        if l.strip().startswith(first_row[0]):
            break
    line = lines[ix]
    coords = get_coords(line)
    for l_ix in range(ix-8,ix):
        for t in toxins:
            if t in lines[l_ix]:
                header_lines.append(l_ix)
                p = lines[l_ix].find(t)
                col = closest_coord(p)                
                toxin_headers[col].append(t)
    header_lines =[lines[i] for i in list(set(header_lines))]
    toxin_headers = [list(set(h)) for h in toxin_headers]
    for h in toxin_headers:
        h.sort()
    toxin_headers = ['+'.join(h) for h in toxin_headers]
    return headers+toxin_headers,header_lines
    
    
            
def fito_columns(lines):
    part_lens = []
    parts = []
    ixs = []
    
    for l in lines:
        ps = [s.strip() for s in l.split('  ') if s!='']
        ps[0] = ps[0].replace('L','').replace('/14','').replace('/13','')
        if ps[0].isnumeric():
            ixs.append(ps[0])
        part_lens.append(len(ps))
        parts.append(ps)
        
    cols = st.mode(part_lens)[0][0]
    print(cols)
    rows = [ps for ps,pl in zip(parts,part_lens) if pl == cols and ps[0].isnumeric()]
    dups = []
    for row in rows:
        try:
            ixs.remove(row[0])
        except ValueError:
            dups.append(row[0])
    col_names = []
    col_lines = []    
    data = None
    col_names,col_lines = fito_headers(lines,rows[0])
    # data = {c:[] for c in col_names}
    # for row in rows:        
    #     for cn,val in zip(col_names,row):
    #         data[cn].append(val)
    # for d in data:
    #     print(d,len(data[d]))
    # data = pd.DataFrame(data)
    # print(data)
    return cols,len(rows), rows,ixs,dups,col_names,col_lines,data

def process_fito(folder):    
    replacements = get_replacements(FITO_REPLACEMENTS)
    rep = []
    fils = glob(os.path.join(folder,'*.txt'))
    frames =  []
    col_sets = {}
    for f in fils:
        print(f)
        f_name = os.path.basename(f)
        if f_name in replacements:
            txt = open(f).read()
            for find,replace in replacements[f_name]['rep']:
                txt = txt.replace(find,replace)
            lines = [l+'\n' for l in txt.split('\n')]
            for min_l,max_l,add in replacements[f_name]['add']:
                for ix,l in enumerate(lines):
                    if min_l<=len(l)<=max_l:
                        lines[ix] = l.strip()+add+'\n'
            # with open(os.path.join('modified',f_name),'w') as ofil:
            #     ofil.write(''.join(lines))            
        else:
            lines = open(f).readlines()
        cols, nrows, rows,missing,dups,col_names,col_lines,df = fito_columns(lines)        
        print(f, nrows)
    
        frames.append(df)
        
        
        #df.insert(0,'file',f_name)
        rep.append(f_name+'\n'+f'    {cols} {nrows}')
        rep.append(''.join(col_lines))
        if '' in col_names:
            rep.append(f'    {col_names}, ***VAZIO***')
        elif 'AO+ASP+DTXs' in col_names:
            rep.append(f'    {col_names}, ***INCONSISTENTE***')
        else:
            rep.append(f'    {col_names}')
        rep.append(f'    {rows[0]}')
        if missing!=[]:
            rep.append(f'    Missing: {missing}')
        if dups!=[]:
            rep.append(f'    Duplicates: {dups}')
        
        rep.append('\n')
    with open(FITO_REPORT,'w') as ofil:
        ofil.write('\n'.join(rep))
    # data = pd.concat(frames,axis=0)
    # data.to_csv(INTERMEDIATE[1],index=False)
    
    with open('logs/fito_col_sets.txt','w') as ofil:
        for k in col_sets:
            ofil.write(f'{k}'+'\n')
            for f in col_sets[k]:
                ofil.write(f'    {f}'+'\n')


MONTH = {m:i+1 for i,m in enumerate(['jan','fev','mar','abr','mai','jun','jul','ago','set','out','nov','dez'])}
MONTH['dec']=12
MONTH['aug']=8
MONTH['apr']=4                

def fix_date(row):
    def get_month(s):
        if s.isnumeric():
            return int(s)
        else:
            return MONTH[s.lower()]
        
    year = int('20'+row['file'].split('.')[0][-2:])
    orig_date = row['data']
    month =MONTH[row['file'].split('.')[0][-5:-2]]
    if '/' in orig_date:
        ps = orig_date.split('/')
        day = int(ps[0])
        m1 = get_month(ps[1])        
    elif '-' in orig_date:
        ps = orig_date.split('-')
        day = int(ps[0])
        m1 = get_month(ps[1])
    
    if m1<month:
        month-=1
    if month<0:
        month = 12
        year+=1
    if month>12:
        month = 1
        year-=1
    date=f'{year}-{m1}-{day}'
    return date

def clean_values(val):
    val = str(val)
    for fi,rep in [(',','.'),('>',''),('<',''),(' ',''),
                   ('NR','nan'),('nr','nan'),('NQ','0'),
                   ('-','nan'),('#REF!','nan'),('ND','0'),
                   ('NEG','0'),('POS','nan')]:
        val = val.replace(fi,rep)
    if ('(' in val) and (')' in val):
        val = val[val.index('(')+1:val.index(')')]
    try:
        return float(val)
    except:
        print(val)
        return(val)
        
def get_dsp(row):
    if not pd.isna(row['AO']):
        return row['AO']
    else:        
        return row['AO+DTXs']

def get_asp(row):
    if not pd.isna(row['AD+AE']):
        return row['AD+AE']
    elif not pd.isna(row['AD+ASP']):
        return row['AD+ASP']
    else:
        return row['ASP']
    

def cleanup():
    df = pd.read_csv('data/biotoxin.csv')
    for c in df.columns[6:]:
        df[c]=df[c].apply(clean_values)
    orig_columns = df.shape[1]    
    df['date'] = df.apply(fix_date,axis=1)
    df['DSP'] = df.apply(get_dsp,axis=1)
    df['ASP'] = df.apply(get_asp,axis=1)
    df['PSP'] = df['STX'].copy()    
    for c in LEGAL_LIMITS:
        df[c+'_rel'] = df[c]/LEGAL_LIMITS[c]
    
    with open('uniques.txt','w') as ofil:
        for c in df.columns[6:orig_columns]:            
            ofil.write(c+'\n')
            vals = [v for v in df[c].unique() if not str(v).strip().replace('.','').isnumeric()]
            for v in vals:
                ofil.write(f'   {v}'+'\n')
                
    df.to_csv(FINAL[0],index = False)   
    return df


def clean_location(s):
    s = s.upper()
    
    for c in ['/','-','.',',',' DE ',' DO ',' DOS ',' DAS ',' DA ']:
        s = s.replace(c,' ')
    tmp = [p for p in s.split() if len(p)>1]
    parts = []
    for p in tmp:
        if p not in parts:
            parts.append(p)    
    return ' '.join(parts)

def clean_location_row(row):
    s = ''    
    for col in LOC_COLS:
        if col in row:
            s += row[col] + ' '            
    return clean_location(s)

def index_words(sentences):
    words = {}
    for ix,s in enumerate(sentences):
        parts = s.split()
        for p in parts:
            try:
                words[p].append(ix)
            except KeyError:
                words[p] = [ix]
    return words

def find_matches(sentence,w_dic):
    parts = sentence.split()
    matches = []
    for p in parts:
        if p in w_dic:
            if matches == []:
                matches = w_dic[p]
            else:
                tmp = [i for i in w_dic[p] if i in matches]                
                if len(tmp)>0:                    
                    matches = tmp
            if len(matches) == 1:
                break
    return matches

def filter_matches(matches,lats,lons):
    thresh = (5*360/40000)**2
    base_lat = lats[matches[0]]
    base_lon = lons[matches[0]]
    res = [matches[0]]
    for m in matches[1:]:
        lat = lats[m]
        lon = lons[m]
        if  ((lat-base_lat)**2+(lon-base_lon)**2)>thresh:
            res.append(m)
    return res

def fix_lat_lon(s):
    s= str(s)
    for f,r in [('"','') ,
                (' ',''),
                (',','.')]:
        s = s.replace(f,r)
    while not s[-1].isdigit():
        print(s)
        s = s[:-1]
    return float(s.replace(',','.'))

from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    From https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def check_distance(mat,maxdist):
    dist = 0
    for i in range(mat.shape[0]-1):
        for j in range(i+1,mat.shape[0]):
            d = haversine(mat[i,0],mat[i,1],mat[j,0],mat[j,1])
            if d>maxdist:
                return -1
            elif dist<d:
                dist = d
    return dist


def aggregate(locs,maxdist):
    areas = locs['Production_Area'].unique()
    aggregates = {'zona':[],'latitude':[],'longitude':[],'points':[],'maxdist':[]}
    for area in areas:
        mat = locs.loc[locs['Production_Area']==area,['Longitude','Latitude']].values
        d = check_distance(mat,maxdist)
        if d>0:
            m = np.mean(mat,axis=0)
            aggregates['zona'].append(area)
            aggregates['longitude'].append(m[0])
            aggregates['latitude'].append(m[1])
            aggregates['points'].append(mat.shape[0])
            aggregates['maxdist'].append(d)
    return pd.DataFrame(aggregates)
    
def non_coincident(table):
    
    def mindist(v1,v2):
        d = 1e9        
        for x in v1:
            for y in v2:
                if abs(x-y)<d:
                    d = abs(x-y)
        return d
    
    mask = table['assigned'].isna()
    unassigned = table['clean_loc'][mask].unique()
    assigned = table['assigned'][~mask].unique()
    u_dates = {l:table.loc[table['clean_loc']==l,'date'].astype(np.int64) /1e9 /3600/24 for l in unassigned}
    a_dates = {l:table.loc[table['assigned']==l,'date'].astype(np.int64) /1e9 /3600/24 for l in assigned}
    disjoint = []
    for l1 in unassigned:
        for l2 in assigned:
            intersect = set(u_dates[l1]).intersection(a_dates[l2])
            if len(intersect) == 0:
                support = min(len(u_dates[l1]),len(a_dates[l2]))
                d = mindist(u_dates[l1],a_dates[l2])
                disjoint.append( (l1,l2,support,d) )
                
    return disjoint
            
def check_non_coincident(table):
    noncon = {k:[] for k in ['zone','unassigned','assigned','support','mindist']}
    zones = table['zona'].unique()
    for zone in zones:
        sel = table.loc[table['zona']==zone,:]
        ncs = non_coincident(sel)
        for l1,l2,s,d in ncs:
            noncon['zone'].append(zone)
            noncon['unassigned'].append(l1)
            noncon['assigned'].append(l2)
            noncon['support'].append(s)
            noncon['mindist'].append(d)
    return pd.DataFrame(noncon)
            
    
def match_locations():
    #biotoxins
    forced_matches = {l.strip().split(':')[0]:l.strip().split(':')[1] for l in open(LOC_FORCED).readlines()}    
    ipma_update = {l.strip().split(':')[0]:l.strip().split(':')[1] for l in open(LOC_IPMA_UPDATE).readlines()}    
    
    biot = pd.read_csv(FINAL[0])
    biot['date'] = pd.to_datetime(biot['date'],errors='coerce')
    biot['zona'] = biot['zona'].str.upper()    
    all_clean_biot = biot.apply(clean_location_row, axis=1)       
    biot['clean_loc'] = all_clean_biot
        
    clean_biot = biot['clean_loc'].unique()        
    biot['assigned'] = np.nan
    biot['longitude'] = np.nan
    biot['latitude'] = np.nan    
    
    #ipma coordinates 
    locs = pd.read_csv(IPMA_LOCS)
    locs = pd.concat((locs,pd.read_csv(ADDED_LOCS)),axis=0)
    locs['Production_Area'] = locs['Production_Area'].str.upper()    
    locs['Latitude'] = locs['Latitude'].apply(fix_lat_lon)
    locs['Longitude'] = locs['Longitude'].apply(fix_lat_lon)
    locs['clean_loc'] = locs.apply(clean_location_row, axis=1)
    for upd in ipma_update:
        mask = locs['clean_loc'] == upd
        locs.loc[mask,'clean_loc'] = ipma_update[upd]
    locs.to_csv('coords_clean.csv',index=False)
    locs.drop_duplicates(subset=['clean_loc'],inplace=True)
    clean_loc = locs['clean_loc'].values    
    locs.set_index('clean_loc',inplace=True)

    words = index_words(clean_loc)
    matched = []
    unmatched = []    
    for l in clean_biot:
        found = None
        if l in forced_matches:
            found = forced_matches[l]
            zone_l = zone_f = found.split()[0]  
        else:
            matches = find_matches(l,words)
            if len(matches) == 1:
                found = clean_loc[matches[0]]
                zone_f = found.split()[0]
                zone_l = l.split()[0]
        if found is not None:            
            if zone_l == zone_f:
                matched.append( l+':'+found)
                mask = biot['clean_loc'] == l
                biot.loc[mask,'latitude'] = locs['Latitude'][found]
                biot.loc[mask,'longitude'] = locs['Longitude'][found]
                biot.loc[mask,'assigned'] = found
                biot.loc[mask,'zona'] = zone_f
            else:
                unmatched.append( l+':'+found+' (MISMATCH)')                
        else:
            possibilities = [clean_loc[i] for i in matches]
            unmatched.append( l+':'+';'.join(possibilities))
    
    count = [np.sum(all_clean_biot == u.split(':')[0]) for u in matched]
    ix = np.argsort(count)[::-1]            
    with open(MATCHFILE,'w') as ofil:
        for i in ix:
            ofil.write(f'{count[i]}'+'\t'+matched[i]+'\n')    
    count = [np.sum(all_clean_biot == u.split(':')[0]) for u in unmatched]
    ix = np.argsort(count)[::-1]        
    with open(UNMATCHFILE,'w') as ofil:
        for i in ix:
            ofil.write(f'{count[i]}'+'\t'+unmatched[i]+'\n')
    biot.to_csv(BIOTOX_ASSIGNED,index=False)
    
    nonco = check_non_coincident(biot)    
    # nonco = nonco.loc[nonco['support']>20,:]
    # nonco = nonco.loc[nonco['mindist']<15,:]
    nonco.to_csv(NONCOINCIDENTS,index=False)
    
    
    sizes = pd.read_csv(ZONESIZES)    
    aggregates = sizes.loc[sizes['length']<AGGREGATE_DISTANCE,:]
    aggregates.to_csv(AGGREGATED,index=False)
    for ix,row in aggregates.iterrows():
        mask = biot['zona'] == row['zone']
        biot.loc[mask,'latitude'] = row['latitude']
        biot.loc[mask,'longitude'] = row['longitude']
        biot.loc[mask,'assigned'] = row['zone']
    
    
    
    biot.to_csv(BIOTOX_AGGREGATED,index=False)
        
        
        
def zone_sizes():
    
    import json
    
    def get_shapes(data):
        scale = np.array(data['transform']['scale'])
        trans = np.array(data['transform']['translate'])
        shapes = []
        for s in data['arcs']:
            p = np.array(s[0])*scale+trans
            shape = [p.copy()]
            for a in s[1:]:            
                p += np.array(a)*scale            
                shape.append(p.copy())
            shapes.append(np.array(shape))        
        return shapes    
    
    def get_points(data):
        shapes = get_shapes(data)    
        geoms = data['objects']['name']['geometries']
        zones = {}
        for g in geoms:                    
            if g['type']=='Polygon':                
                arcs = g['arcs'][0]
                points = []                    
                for arc in arcs:
                    if arc>=0:
                        points.extend(shapes[arc])
                    else:
                        points.extend(shapes[-arc-1][::-1])
                zones[g['properties']['code']] = np.array(points)
            elif g['type']=='MultiPolygon':
                polys = g['arcs'][0]
                points = []
                for p in polys:                    
                    for arc in p:
                        if arc>=0:
                            points.extend(shapes[arc])
                        else:
                            points.extend(shapes[-arc-1][::-1])                    
                zones[g['properties']['code']] = np.array(points)               
        return zones
    
    ipma = json.load(open('../1_PorSemana_OF_deliverable_1/ipma_map_v2.json'))        
    zones = get_points(ipma) 
    sizes = {k:[] for k in ['zone','length','longitude','latitude']}
    for z in zones:
        points = zones[z]
        dists = [np.sum((points-p)**2,axis=1) for p in points]
        maxs =  [np.max(d) for d in dists]
        p1 = np.argmax(maxs)
        p2 = np.argmax(dists[p1])
        sizes['zone'].append(z)
        sizes['length'].append(haversine(points[p1,0],points[p1,1],points[p2,0],points[p2,1]))
        sizes['longitude'].append(np.mean(points[:,0]))
        sizes['latitude'].append(np.mean(points[:,1]))
    sizes = pd.DataFrame(sizes)
    sizes.to_csv(ZONESIZES,index=False)
    return sizes
        

def update_coords():
    
    dfs = pd.read_excel('config/sample_points_coordinates.xlsx',sheet_name= None,header=1)
    coords = None
    for d in dfs:
        df = dfs[d]
        if coords is None:
            coords = df.copy()
        else:
            coords = pd.concat((coords,df),axis=0)
    rename = {'Código':'Production_Area',
              'Local de amostragem':'Sample_Point',
              'Ponto amostragem (Latitude)':'Latitude',
              'Ponto amostragem (Longitude)':'Longitude'
              }
    coords.rename(columns=rename,inplace=True)
    sel = [rename[k] for k in rename]
    coords = coords[sel]
    coords.dropna(inplace=True)
    coords.drop_duplicates(subset = ['Production_Area','Sample_Point'],inplace=True)
    coords['Latitude'] = coords['Latitude'].apply(fix_lat_lon)
    coords['Longitude'] = coords['Longitude'].apply(fix_lat_lon)
    coords['Sample_Point'] = coords['Sample_Point'].str.strip()
    coords['Production_Area'] = coords['Production_Area'].str.strip()
    coords.to_csv(IPMA_LOCS,index=False)
    return coords

def process():
    #update_coords()
    # retrieve_all()
    # for ipma,txt_folder in zip(IPMA_URLS,TXT_FOLDERS):
    #     if txt_folder!='':
    #         pdf_to_txt(ipma[1],txt_folder)
    
    # process_toxin(TXT_FOLDERS[0])    
    process_fito(TXT_FOLDERS[1])    
    cleanup()
    #zone_sizes()
    #match_locations()

if __name__ =='__main__':
    process()    
    
    
