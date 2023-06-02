#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed in the scope of the project
  “MATISSE: A machine learning-based forecasting system for shellfish safety”
Funded by the Portuguese Foundation for Science and Technology (DSAIPA/DS/0026/2019).
This code is provided 'as is', with no implied or explicit guarantees.
This code is provided under a CC0 public domain license.

"""

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
        


            
        
        
        
        



