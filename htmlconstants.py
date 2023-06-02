"""Constants and utilities for rendering html pages

Developed in the scope of the project
  “MATISSE: A machine learning-based forecasting system for shellfish safety”
Funded by the Portuguese Foundation for Science and Technology (DSAIPA/DS/0026/2019).
This code is provided 'as is', with no implied or explicit guarantees.
This code is provided under a CC0 public domain license.
"""


SERVER_URL='http://127.0.0.1:8888'
"""str: url for the server"""

HTML_FOLDER='html/'
"""str: path for html, css, js files"""

HTML_INDEX=HTML_FOLDER+'index.html'
"""str: path for html starting page"""

HTML_MODEL=HTML_FOLDER+'model.html'
"""str: path for html model page"""

HTML_CURRENT=HTML_FOLDER+'current.html'
"""str: path for html current status page"""

HTML_PREDICT = '/predict'
"""str: url for predictions page"""

HTML_STATUS = '/monitor'
"""str: url for current status page"""


HTML_FOLDER_TAG='[HTML]'
"""str: id tag to be replaced by the html folder"""

PLOT_FOLDER = HTML_FOLDER+'images/'

MODEL_DIV_TEMPLATE = HTML_FOLDER+'model_div_template.html'




def replace_in_string(source,replacements):
    """
    replace the keys of replacements in source with
    the values of replacements (a dictionary)
    returns the replaced string plus a list of all
    keys not found in the string
    """
    outkeys=[]
    if replacements is not None:
        for key in replacements.keys():            
            if key in source:
                source=source.replace(key,replacements[key])
            else:
                outkeys.append(key)
    return source,outkeys

def create_control(control_type,name=None,label=None,options=None):
    """
    return an input control for html
    """
    if label is not None:
        res = '<label for="{1}">{0}:</label><{2} '.format(label,name,control_type)
    else:
        res = '<'+control_type+' '
    if name is not None:
        res = res + 'name="{0}" id="{0}" '.format(name)
    if options is not None:
        res = res + options
    return res + '>\n'.format(control_type,name,options)    
    

def control_dict(obj,attributes,drop_lists):
    """
    return dictionary with html input elements for listed attributes of
    object. Dictionary keys include brackets: [attr_name]
    """    
    replacements = {}        
    for a in attributes:        
        line = ''
        if len(a)==2:
            at_name,at_label = a
            attr = getattr(obj,at_name)                        
            if at_name in drop_lists.keys():            
                line = line + create_control('select',at_name,at_label)
                for option in drop_lists[at_name]:
                    if option == attr:                
                        line = line + create_control('option',None,None,'selected')
                    else:
                        line = line + create_control('option')
                    line = line[:-1] +option +'</option>\n'
                line = line + '</select></p>\n'
            elif type(attr) is bool:
                if attr:
                    check = 'checked'
                else:
                    check = None
                line = line + create_control('input type="checkbox"',at_name,at_label,check)                
            else:
                line = line + create_control('input type="text" value="{0}"'.format(attr),at_name,at_label)
        else:
            at_name,at_label,rows,cols = a
            attr = getattr(obj,at_name)
            aux = 'textarea rows="{0}" cols="{1}"'.format(rows,cols)
            line = line + create_control(aux,at_name,at_label)
            line = line + attr +'</textarea>\n'
            
        replacements['['+at_name+']']=line
    return replacements
    

def attributes_to_form(name, action,obj,attributes,
                       drop_lists={}, template='', submit = 'Submit'):
    """
    return a string with an html form for all attributes listed
    name is the name of the form
    action is the action url for the form
    obj is any object    
    attributes is a list of tuples for the attributes to include in form
        (attribute name, form label)
    drop_lists is a dictionary with attribute_name and list of strings for
      options
    template is a string where each instance of the name within square brackets
    is replaced by the control      
    """
    
    res = '<form name="{0}" action="{1}" method="post" enctype="multipart/form-data">\n'.format(name,action)
    replacements = control_dict(obj,attributes,drop_lists)
    form,outs = replace_in_string(template,replacements)
    res = res + form
    for out in outs:
        res = res + out
    res = res + '<input type="submit" value="{0}">\n</form>\n'.format(submit)
    return res

def form_to_attributes(form_data,attributes,obj):
    """updates the object attributes with the form data
    form data is a dictionary with attribute names and values
    attributes is a list of tuples for all exported attributes of the class (name, label)
    obj is the object to update.
    """

    for a in attributes:
        at_name = a[0]
        attr = getattr(obj,at_name)  
        if type(attr) is bool:
            setattr(obj, at_name, False)
        if at_name in form_data.keys():
            if type(attr) is bool:
                setattr(obj, at_name, form_data[at_name].upper=='TRUE')
            elif type(attr) is int:
                setattr(obj, at_name, int(form_data[at_name]))
            elif type(attr) is float:
                setattr(obj, at_name, float(form_data[at_name]))
            elif type(attr) is long:
                setattr(obj, at_name, long(form_data[at_name]))
            else:
                setattr(obj, at_name, form_data[at_name])
                    
            

def process_html(html_source,replacements=None):
    """reads the html source file and replaces tags
       replacements is a dictionary with tag:text_to_replace
    """
    fil = open(html_source,'rt',encoding='utf-8')
    source = fil.read()
    fil.close()
    source=source.replace(HTML_FOLDER_TAG,HTML_FOLDER)        

    source,outs = replace_in_string(source,replacements)
    return source
