######## Reading Data #######

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os #importing a module

os.chdir('C:/Users/sdt144/Dropbox/SamPython')
os.getcwd()

##### reading .txt or .csv files
df1=pd.read_csv('./DataFiles/cars1.csv') #header is true by default
df1.shape
df1.head()

df1=pd.read_table('DataFiles/cars1.csv',sep=',') #header is true by default
df1.head()

pd.read_csv('DataFiles/cars1.csv',nrows=4) #reading just 4 row

pd.read_csv('DataFiles/cars1.csv',names=['dist'])

df1.to_csv('DataFiles/cars1_new.csv') # DF to csv-file

#saving csv with other delimiters
import sys
df1.to_csv(sys.stdout)
df1.to_csv(sys.stdout,sep='_')

df1.to_csv(sys.stdout,names=['dist']) # save just the first column

##### JSON files
import json
json_obj = """
{
    "id": 1,
    "name": "A green door",
    "price": 12.50,
    "tags": ["home", "green"]
}
"""

data = json.loads(json_obj)
data

df=DataFrame(data['tags'])
df

##### HTML files
from pandas import read_html
# pip install html5lib
# pip install beautifulsoup4
# pip install lxml

url='https://www.fdic.gov/bank/individual/failed/banklist.html'
DF_list=pd.io.html.read_html(url)
DF_list=read_html(url)
type(DF_list)
len(DF_list)
DF=DF_list[0]
DF
DF.columns.values

##### XLS files
# pip install xlrd
# pip install openpyxl

xlsfile = pd.ExcelFile('DataFiles/cameras.xlsx')
xlsfile
type(xlsfile)
DF = xlsfile.parse('Baltimore Fixed Speed Cameras') # name of the sheet
DF

