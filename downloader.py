import requests
import pandas as pd
import urllib.request
import json
from concurrent.futures import ThreadPoolExecutor
import httplib2

def get_img(nmid, path): 
    h = httplib2.Http('.cache') 
    for nm in nmid:
        for i in range(1, 2):
            img = h.request("http://img1.wbstatic.net/big/new/" + str(nm - nm % 10000) + "/"+ str(nm) +"-{0}.jpg".format(i))
            if(img[0]['status']!='404'): open(path + str(nm) +"-{0}.jpg".format(i), 'wb').write(img[1])        
    
        
def get_description(nmid, path): 
    
    def get_content(nmid):
        vals = pd.Series(requests.get('https://content.wildberries.ru/api/v3/product-cache/product/nm?nmId={0}'.format(nmid)).json()['imt']['subject']['name'])
        return pd.DataFrame({'nmid': nmid, 'subject': vals}) 

    with ThreadPoolExecutor(max_workers=25) as executor:
        future = executor.map(get_content, nmid)
    
    temp = []    
    for df in future: temp.append(df)
    description = temp[0]   
    for df in temp[1:]: description = pd.concat([description, df], axis = 0)
    description.reset_index(drop = True).to_csv(path + 'description.csv', index = False)
    
def run(nmid, path):
    get_img(nmid, path)
    get_description(nmid, path)

#run(list(range(11290562, 11290562 + 10000)), path=r"C:\Users\Buc.Nikita\Desktop\wb_images\data\files_")