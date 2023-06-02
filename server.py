"""HTTP server module

This module implements HTTP access

HTTP request handling based on examples by Doug Hellmann on
Python Module of the Week:
    http://pymotw.com/2/BaseHTTPServer/index.html
    
Developed in the scope of the project
  “MATISSE: A machine learning-based forecasting system for shellfish safety”
Funded by the Portuguese Foundation for Science and Technology (DSAIPA/DS/0026/2019).
This code is provided 'as is', with no implied or explicit guarantees.
This code is provided under a CC0 public domain license.
    
"""

from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse as urlparse
import htmlconstants as htc
HTML_ENCODING = 'UTF-8'
import json
import predictions
   
class Handler(BaseHTTPRequestHandler):
    
    def send_file(self,file_name,contents):
        """call send_header depending on file_name"""
        mimetype = None
        if file_name.endswith(".avif"):
            mimetype='image/avif'
        if file_name.endswith(".html"):
            mimetype='text/html'
        if file_name.endswith(".jpg"):
            mimetype='image/jpg'
        if file_name.endswith(".png"):
            mimetype='image/png'
        if file_name.endswith(".gif"):
            mimetype='image/gif'
        if file_name.endswith(".js"):
            mimetype='application/javascript'
        if file_name.endswith(".css"):
            mimetype='text/css'
        
        # if mimetype == 'image/png':
        #     self.send_response(304)            
        #     self.send_header('Content-type',mimetype)
        #     self.end_headers()
        #     self.wfile.write('')            
        # el
        if mimetype is not None:
            self.send_response(200)            
            self.send_header('Content-type',mimetype)
            self.end_headers()
            self.wfile.write(contents)            
        return
    
    def redirect(self,url):
        self.send_response(301)       
        self.send_header('Location',url)
        self.send_header( 'Connection', 'close' );
        self.end_headers()        
    
    def do_GET(self):
        """Process GET requests
        """
        global refresh_images      
        html='Error 404'        
        print(self.path)
        parsed_url = urlparse.urlparse(self.path)
        path = parsed_url[2]        
        file_name='dummy.html'        
        if path=='/' or path.upper()=='/INDEX.HTML':
            html = open(htc.HTML_INDEX).read()
        elif path==htc.HTML_PREDICT:    
            html = predictions.model_page(urlparse.parse_qs(parsed_url.query))     
        elif path==htc.HTML_STATUS:
            html = predictions.current_page()            
        else:            
            file_name=urlparse.unquote(path)            
            html = open(htc.HTML_FOLDER+file_name,'rb').read()
        if type(html) == str:
            html = html.encode(HTML_ENCODING)
        self.send_file(file_name,html)        
        return
        
    def post_data_as_dict(self):
        """
            Return post data as a dictionary 
        """        
        
        ctype,pdict = cgi.parse_header(self.headers.get('Content-type'))
        if ctype == 'multipart/form-data':
            postvars = cgi.parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            postvars = urlparse.parse_qs(
                        self.rfile.read(length), 
                        keep_blank_values=1)
        elif ctype == "application/json":
            data_string = self.rfile.read(int(self.headers['Content-Length']))            
            self.send_response(200)
            self.end_headers()    
            return json.loads(data_string) # return immediately to skip the trick below of [0]        
        else:
            postvars = {}
        for key in postvars.keys():
            #this is important not only to get rid of lists but
            #because of the hidden input trick for checkboxes
            postvars[key]=postvars[key][0]
        return postvars
        
            
    def handle_post_request(self):
        """Handles file upload and other stuff (WiP...)
           Source: Huang, Tao at https://gist.github.com/UniIsland/3346170
           Returns (True, session_id) if successful, or (False, error_message) otherwise
        """
        url = urlparse.urlparse(self.path)[2]  
        if url == htc.URL_DATA_POST:       
            query = self.post_data_as_dict()  
            print(query)
            data_ann.load_data_set(query)
            return (True, '/data')
        elif url == htc.URL_IMAGE_CLASSIF:       
            query = self.post_data_as_dict()  
            print(query)
            image_ann.classify_image(query)
            return (True, htc.URL_IMAGES)
        elif url == htc.QS_POST_OPTIONS:       
            query = self.post_data_as_dict()                  
            htc.form_to_attributes(query,QSGame.export,qs_game)                    
            return (True, htc.SERVER_URL+htc.URL_QSCURRENT[1:])
        return(False,'/') 

    def do_POST(self):
        (res,msg)=self.handle_post_request()
        if not res:
            # upload failed
            self.send_response(200)       
            self.end_headers()        
            self.wfile.write(f'Request failed: {msg}'.encode(HTML_ENCODING))
        else:
            #upload OK, redirecting to approapriate page
            self.redirect(msg)
            
        return
    
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """ This class allows to handle requests in separated threads.
        No further content needed, don't touch this. """

if __name__ == '__main__':    
    #For safety reasons, server is confined to local host
    #Change 'localhost' to '' to enable remote access
    #server = ThreadedHTTPServer(('localhost', 8081), Handler)
    server = HTTPServer(('localhost', 8888), Handler)    
    #server = ThreadedHTTPServer(('localhost', 8888), Handler)
    print('Loading models and data')
    predictions.prepare()
    print('Starting server. Connect with browser to 127.0.0.1:8888. Use <Ctrl-C> to stop server')
    server.serve_forever()
