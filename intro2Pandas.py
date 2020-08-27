##### Introduction to Panda #####
%reset
import os
os.system('cls')

import numpy as np
import pandas as pd

from pandas import Series, DataFrame # to avoid pd.Series or pd.DataFrame

##### Series #####
obj=Series([3,6,9,12])
obj
obj.values
obj.index

ww2_cas=Series([8700000,4300000,3000000,2100000,400000],
                index=['USSR','Germany','China','Japan','USA'])
ww2_cas
ww2_cas['USA']

# check which countries had casualties greater than 4M
ww2_cas[ww2_cas>4000000]

'USSR' in ww2_cas # check membership

#convert series to dictionaries and the other way
ww2_dict=ww2_cas.to_dict() # convert series to a dictionary
ww2_dict
ww2_series=Series(ww2_dict) # convert to Series
ww2_series

countries=['China','Germany','Japan','USA','USSR','Argentina']
obj2=Series(ww2_dict,index=countries)
obj2
pd.isnull(obj2) # check for nulls
pd.notnull(obj2)

#add by index
ww2_series
obj2
ww2_series+obj2

# labeling
obj2.name='World War II Casualties'
obj2
obj2.index.name="countries"
obj2

##### Data Frames #####
import numpy as np
import pandas as pd
from pandas import Series, DataFrame 

###
import webbrowser
website='https://en.wikipedia.org/wiki/NFL_win-loss_records'
webbrowser.open(website)

nfl_df=pd.read_clipboard() #highlight from bottom-right then copy
nfl_df
nfl_df.columns # get column names

# retrieve column(s)
nfl_df.Rank # get a column
nfl_df.First Season # doesn't work
nfl_df['First Season']
DataFrame(nfl_df,columns=['Team','First Season','Total Games'])
DataFrame(nfl_df,columns=['Team','First Season','Total Games','Stadium']) # no 'stadium'

# retrieve row(s)
nfl_df.head() # first 5 rows
nfl_df.head(3)
nfl_df.tail()
nfl_df.tail(2)

nfl_df
nfl_df.ix[3] #output is a Series

# Modifying a column
nfl_df['Stadium']='Heinz Field'
nfl_df
nfl_df['Stadium']=np.arange(5)
nfl_df

stadium_series=Series(["Stadium 1","Stadium 2"],index=[0,3])
nfl_df['Stadium']=stadium_series
nfl_df

# Deleting a column
del nfl_df['Stadium']
nfl_df

# Making data frames from dictionarfies
city_dict={'City':['SF','LA','NYC'],'Population':[837000,3880000,8400000]}
city_df=DataFrame(city_dict)
city_df

##### playing with indices
ser1=Series([1,2,3,4],index=['A','B','C','D'])
my_index=ser1.index
my_index
my_index[1:]
my_index[:2]

my_index[0]='Z' # no mutable

### Creating new series/DFs by reindexing old ones
# Series
ser2=ser1.reindex(['A','B','C','D','E','F'])
ser2

ser3=ser2.reindex(['A','B','C','D','E','F','G'],fill_value=0) #fill_value
ser3

ser4 = Series(['USA','Mexico','Canada'],index = [0,5,10])
ser4
ser4.reindex(range(15),method='ffill') # reindexing series

# DFs
from numpy.random import randn
df1 = DataFrame(randn(25).reshape((5,5)),
                   index=['A','B','D','E','F'], # missed C
                   columns=['col1','col2','col3','col4','col5'])
df1
df2=df1.reindex(['A','B','C','D','E','F']) #renindexing rows of a df
df2

new_columns = ['col1','col2','col3','col4','col5','col6']
df2.reindex(columns=new_columns) #reindexing columns of a df

#reindexing with .ix is faster
df1
df1.ix[['A','B','C','D','E','F'],new_columns]

### Droping enteries
ser1=Series(np.arange(3),index=['a','b','c'])
ser1
ser1.drop('b')

df1=DataFrame(np.arange(9).reshape([3,3]),
    index=['SF','LA','NYC'],columns=['pop','size','year'])
df1
df2=df1.drop('LA')
df2

df3=df1.drop('year',axis=1) #axis=0 for row and axis=1 for column
df3

##### Selecting Enteries
### subset from series
ser1=Series(np.arange(3),index=['A','B','C'])
ser1=2*ser1
ser1
ser1['B']
ser1[1]
ser1[0:3]
ser1[['A','B']]
ser1[ser1>1]
ser1[ser1>3]=10
ser1

###subset from Data Frame
df=DataFrame(np.arange(25).reshape((5,5)),index=['LA','NYC','SF','DC','CHI'],
                columns=['A','B','C','D','E'])
df

#subset column
df['B']
df[['B','E']]
df[df['C']>8]
df>10
# subset row
df
df.ix['LA'] #with one argument, it would be row-based
df.ix[0]
df.ix[0,:]

##### Alignment
ser1 = Series([0,1,2],index=['a','b','c'])
ser2 = Series([3,4,5,6],index = ['a','b','c','d'])
ser1+ser2

df1=DataFrame(np.arange(4).reshape((2,2)),
            columns=list('AB'),index=['NY','CA']) #list is a python function
df2=DataFrame(np.arange(9).reshape((3,3)),
                    columns=list('ADC'),index=['NY','CA','SF'])
                                       
df1
df2
df1+df2
df1.add(df2,fill_value = 0) # B, SF doesn't exist in either

ser3 = df2.ix[0]
ser3
df2-ser3

##### Ranking and Sorting
#sort_index(), order()
ser1 = Series(range(3),index=['c','a','b'])
ser1
ser1.sort_index() # order by Index (asc)
ser1.order() # order by value (asc)
ser1.order(ascending=False) # order by value (asc)

#rank()
from numpy.random import randn
ser2 = Series(randn(10))
ser2
ser2.rank()
# sort put a series in order of it's item ranks

##### Summary
arr = np.array([[1,2,np.nan],[np.nan,3,4]])
arr
df1 = DataFrame(arr,index = ['a','b'],columns = ['one','two','three'])
df1

df1.sum() #default axis is 0, and Pandas will ignore the nan values
df1.sum(axis=1) #by row

df1
df1.min()
df1.min(axis=1)
df1.idxmin()
df1.idxmin(axis=1)

df1.cumsum() # accumulation sum

### unique() and value_count() methods for factor variables
ser1 = Series(['w','w','x','y','z','w','x','y','x','a'])
ser1
ser1.unique()
ser1.value_counts()

###describe method
df1
df1.describe() # similar to the summary() method in R will provide summary stat.

###covariance matrices and some visulaization
import pandas.io.data as pdweb
import datetime

prices = pdweb.get_data_yahoo(['CVX','XOM','BP'],start = datetime.datetime(2010,1,1),
                             end = datetime.datetime(2013,1,1))['Adj Close'] # stock closing price
prices.shape
prices.head()

rets = prices.pct_change() # returns= % change in the stock prices
rets.head()

rets.corr(method='pearson') #correlation matrix

#%matplotlib inline # if you want it inline
prices.plot()

#pip install seaborn #canopy command prompt, pip is the package management system
import seaborn as sns
#import matplotlib.pyplot as plt
sns.corrplot(rets,annot = False,diag_names = False)


##### Missing data
data = Series(['one','two',np.nan,'four'])
data
data.isnull()
data.dropna() #drop or remove null values

df = DataFrame([[1,2,3],[np.nan,5,6],[7,np.nan,9],[np.nan,np.nan,8]])
df
df.dropna() #row-wise
df.dropna(axis =1) # column-wise

df = DataFrame([[np.nan,2,3],[np.nan,5,6],[np.nan,np.nan,9],[np.nan,np.nan,np.nan]])
df
df.dropna(how="all")
df.dropna(how="all",axis=1)

#thresholding
npn = np.nan # to just write npn
df2 = DataFrame([[1,2,3,npn],[2,npn,5,6],[npn,7,npn,9],[1,npn,npn,npn]])
df2
df2.dropna(thresh = 2) # keep if num(non-NA)>=thersh
df2.dropna(thresh = 3)

df2
df2.fillna(1)

df2
df2.fillna({0:0,1:1,2:2,3:10}) #fill diff. columns differently col:value

df2
df2.fillna(0,inplace = True)
df2

##### Index Hierarchy
from numpy.random import randn

ser = Series(randn(6),index = [[1,1,1,2,2,2],['a','b','c','a','b','c']]) # 2 lists of indexes
ser
ser.index
ser[1]
ser[2]

ser[:,'a']

df=ser.unstack()
df

df2 = DataFrame(np.arange(16).reshape(4,4),index = [['a','a','b','b'],[1,2,1,2]],
                    columns = [['NY','NY','LA','SF'],['cold','hot','cold','hot']])
df2

#naming indices
df2.index.names = ['Index_1','Index_2']
df2.columns.names = ['Cities','Temp']
df2

df2.swaplevel('Cities','Temp',axis = 1) #interchange the indices

df2.sortlevel(1)
df2.sortlevel(0)
df2
df2.sortlevel(0,axis=1)

df2.sum(axis=1)
df2.sum(level = 'Temp',axis =1) #provide the sum for each level of the index separately