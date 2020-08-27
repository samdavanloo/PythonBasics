########## Data Manipulation ##########
%reset
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
os.system('cls')
os.chdir('C:/Users/sdt144/Dropbox/SamPython')
os.getcwd()

##### Merging
df1 = DataFrame({'key':['X','Z','Y','Z','X','X'],
                    'data_set_1':np.arange(6)})
df1
df2 = DataFrame({'key':['Q','Y','Z'],
                    'data_set_2':[1,2,3]})
df2
pd.merge(df1,df2) #note that there are two "z" in data_set_1
pd.merge(df1,df2,on='key') #specify the primary-key column

pd.merge(df1,df2,on='key',how='left')
df1
pd.merge(df1,df2,on='key',how='right')
pd.merge(df1,df2,on='key',how='outer') #union

# practice again
df3 = DataFrame({'key':['X','X','X','Y','Z','Z'],
                    'data_set_3':range(6)})
df4= DataFrame({'key':['Y','Y','X','X','Z'],
                    'data_set_4':range(5)})
df3
df4
pd.merge(df3,df4)

# two keys
df_left = DataFrame({'key1':['SF','SF','LA'],
                    'key2':['one','two','one'],
                    'left_data':[10,20,30]})
df_right = DataFrame({'key1':['SF','SF','LA','LA'],
                     'key2':['one','one','one','two'],
                     'right_data':[40,50,60,70]})
df_left
df_right
pd.merge(df_left,df_right,on=['key1','key2'])
pd.merge(df_left,df_right,on=['key1','key2'],how='outer')

df_left
df_right
pd.merge(df_left,df_right,on='key1')
pd.merge(df_left,df_right,on='key1',suffixes=('_lefty','_righty'))


### merge by index
df_left = DataFrame({'key':['X','Y','Z','X','Y'],'data':range(5)})
df_left
df_right = DataFrame({'group_data':[10,20]},index=['X','Y'])
df_right
pd.merge(df_left,df_right,left_on='key',right_index=True)

df_left_hr = DataFrame({'key1':['SF','SF','SF','LA','LA'],
                       'key2':[10,20,30,20,30],
                       'data_set':np.arange(5)})
df_left_hr
df_right_hr = DataFrame(np.arange(10).reshape(5,2),
                        index=[['LA','LA','SF','SF','SF'],
                              [20,10,10,10,20]],
                        columns=['col_1','col_2'])
df_right_hr
pd.merge(df_left_hr,df_right_hr,left_on=['key1','key2'],right_index=True)

### another method
df_left
df_right
df_left.join(df_right)

##### Concatenate
#arrays
arr1 = np.arange(9).reshape(3,3)
arr1
np.concatenate([arr1,arr1]) # along axis one
np.concatenate([arr1,arr1],axis=1)

#series
ser1 = Series([0,1,2],index=['T','U','V'])
ser2 = Series([3,4],index=['X','Y'])
ser1
ser2
pd.concat([ser1,ser2]) # alone axis one, output is a series
pd.concat([ser1,ser2],axis=1) #output is a Data Frame
pd.concat([ser1,ser2],keys=['cat1','cat2']) # doubly indexed series, with keys as the outer index

# DFs
df1 = DataFrame(np.random.randn(4,3),columns=['X','Y','Z'])
df2 = DataFrame(np.random.randn(3,3),columns=['Y','Q','X'])
df1
df2
pd.concat([df1,df2])
pd.concat([df1,df2],ignore_index=True) # ignores the index

##### Combining Series and DFs
#Series
ser1 = Series([2,np.nan,4,np.nan,6,np.nan],index=['Q','R','S','T','U','V'])
ser1
ser2 = Series(np.arange(len(ser1)),dtype=np.float64,index = ['Q','R','S','T','U','V'])
ser2
Series(np.where(pd.isnull(ser1),ser2,ser1),index=ser1.index)

ser1.combine_first(ser2) # uses ser1 unless its null values which are replaced by ser2

#DFs
nan=np.nan
df_odds = DataFrame({'X':[1.,nan,3.,nan],'Y':[nan,5,nan,7],'Z':[nan,9,nan,11.]})
df_evens = DataFrame({'X':[2.,4,nan,6.,8.],'Y':[nan,10.,12.,14.,16.]})
df_odds
df_evens
df_odds.combine_first(df_evens)

##### Reshaping
df1= DataFrame(np.arange(8).reshape(2,4),
                  index=pd.Index(['LA','SF'],name='city'),
                  columns=pd.Index(['A','B','C','D'],name='letter'))
                  # pd.index() allows you to name your index
df1
df_st = df1.stack() # output is a series
df_st
df_st.unstack() # back to DF
df_st.unstack('letter')
df_st.unstack('city') #similar to transpose

ser1 = Series([0,1,2],index=['Q','X','Y'])
ser2 = Series([4,5,6],index=['X','Y','Z'])
df = pd.concat([ser1,ser2],keys=['Alpha','Beta']) # remember keys would be outer indices
df
df.unstack()
df.unstack().stack() # no null value

df=df.unstack()
df
df.stack(dropna=False)

##### Pivoting
df=pd.read_csv('./DataFiles/data4pivot.csv')
df
df_piv = df.pivot('date','variable','value')
df_piv

##### Duplicates in DFs
df = DataFrame({'key1':['A']*2 + ['B']*3,
                   'key2':[2,2,2,3,3]})
df
df.duplicated() # read from top to bottom

df.drop_duplicates()
df.drop_duplicates(['key1']) #based on a specific column

df
df.drop_duplicates(['key1'],take_last=True)

##### Mapping
df = DataFrame({'city':['Alma','Brian Head','Fox Park'],'altitude':[3158,3000,2762]})
df
state_map = {'Alma':'Colorado','Brian Head':'Utah','Fox Park':'Wyoming'}
df['state'] = df['city'].map(state_map) # map based on a dict.
df

##### Replace
ser1 = Series([1,2,3,4,1,2,3,4])
ser1
ser1.replace(1,np.nan)
ser1.replace([1,4],[100,400])
ser1.replace({4:np.nan}) #using dictionaries

##### Rename Index
df = DataFrame(np.arange(12).reshape(3,4),
                  index = ['NY','LA','SF'],
                  columns= ['A','B','C','D'])
df
df.index.map(str.lower)
df.index=df.index.map(str.lower)
df

df.rename(index=str.title,columns=str.lower) # changing index and columns at the same time

df.rename(index={'ny':'NEW YORK'},columns = {'A':'ALPHA'}) # renaming using dictionaries

df
df.rename(index={'ny':'NEW YORK'},inplace=True) #permanent
df

##### Binning
years = [1990,1991,1992,2008,2012,2015,1987,1969,2013,2008,1999]
decade_bins = [1960,1970,1980,1990,2000,2010,2020]
decade_cat = pd.cut(years,decade_bins)
decade_cat.categories
pd.value_counts(decade_cat)
pd.cut(years,2) # number of bins

##### Outliers
np.random.seed(123)
df = DataFrame(np.random.randn(1000,4))
df.head()
df.tail()
df.describe()

col=df[0]
col.head()
col[np.abs(col)>3]

df[(np.abs(df)>3).any(1)]

df[np.abs(df)>3] = np.sign(df)*3 # cap outliers in the data at 3
df.describe()

##### Permutations
df = DataFrame(np.arange(16).reshape(4,4))
df
blender = np.random.permutation(4) #without replacement
blender
df.take(blender) #permutate the rows 

box = np.array([1,2,3]) #with replacement 
shaker=np.random.randint(0,len(box),size=10)
shaker
box.take(shaker)


##### Groupby on DFs
df = DataFrame({'k1':['X','X','Y','Y','Z'],
                    'k2':['alpha','beta','alpha','beta','alpha'],
                    'dataset1':np.random.randn(5),
                    'dataset2':np.random.randn(5)})
df
group1 = df['dataset1'].groupby(df['k1'])
group1
group1.mean()

cities = np.array(['NY','LA','LA','NY','NY'])
month = np.array(['JAN','FEB','JAN','FEB','JAN'])
df
df['dataset1'].groupby([cities,month]).mean() # not necessarily in the same DF

df.groupby('k1').mean() #groupby a column
df.groupby(['k1','k2']).mean() #groupby two columns

dataset2_group = df.groupby(['k1','k2'])[['dataset2']] # just dataset2
dataset2_group.mean()

df.groupby(['k1']).size() # find sizes of the groups

for name,group in df.groupby('k1'):
    print "This is the %s group" %name
    print group
    print '\n'

for (k1,k2) , group in df.groupby(['k1','k2']):
    print "Key1 = %s Key2 = %s" %(k1,k2)
    print group
    print '\n'

group_dict = dict(list(df.groupby('k1')))# group names are the keys to dictionary
group_dict['X']
group_dict['Y']

list(df.groupby('k1'))

##### More on Groupby of DFs using Dics and Series
animals = DataFrame(np.arange(16).reshape(4, 4),
                   columns=['W', 'X', 'Y', 'Z'],
                   index=['Dog', 'Cat', 'Bird', 'Mouse'])
animals
animals.ix[1, ['W', 'Y']] = np.nan #making some null values
animals
behavior_map = {'W': 'good', 'X': 'bad', 'Y': 'good','Z': 'bad'}
animal_col = animals.groupby(behavior_map, axis=1)
animals
animal_col.sum()

behave_series = Series(behavior_map)
behave_series
animals
animals.groupby(behave_series, axis=1).count() #counts number of instances ignore nans

animals
animals.groupby(len).sum() # groupby the length of DF's index

#side note on creating hierarcical columns in DFs
hier_col = pd.MultiIndex.from_arrays([['NY','NY','NY','SF','SF'],
                [1,2,3,1,2]],names=['City','Neighborhood'])
hier_col
dframe_hr = DataFrame(np.arange(25).reshape(5,5),columns=hier_col)
dframe_hr = dframe_hr*100
dframe_hr

##### Data Aggregation
os.chdir('C:/Users/sdt144/Dropbox/SamPython')
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/'
# Save winequality-red.csv
df_wine = pd.read_csv('DataFiles/winequality_red.csv',sep=';')
df_wine.shape
df_wine.head()

df_wine['alcohol'].mean()

def max_to_min(arr):
    return arr.max() - arr.min()

wino = df_wine.groupby('quality')
wino.describe()

wino.agg('mean') # aggregate function
wino.agg(max_to_min)

df_wine.head()
df_wine['qual/alc ratio'] = df_wine['quality']/df_wine['alcohol']
df_wine.head()

df_wine.pivot_table(index=['quality']) # groupby using pivote_table (showing the means)

df_wine.plot(kind='scatter',x='quality',y='alcohol')

##### Split, Apply, Combine
# we want to see the wines with the highest alcohol per each quality category
df_wine = pd.read_csv('DataFiles/winequality_red.csv',sep=';')

def ranker(df):
    df['alc_content_rank'] = np.arange(len(df)) + 1
    return df
    
df_wine.sort('alcohol',ascending=False, inplace=True)
df_wine.head()
df_wine = df_wine.groupby('quality').apply(ranker) #split and apply (ranker is applied to each group separately)
df_wine.head(20)

def abc(df):
    df['res'] = df['quality']+2
    return df
    
df_wine.apply(abc)


num_of_qual = df_wine['quality'].value_counts()
num_of_qual

len(num_of_qual)
df_wine[df_wine.alc_content_rank == 1].head(len(num_of_qual)) #combine

##### Cross-Tabulation (for frequnency tables)
from StringIO import StringIO

data ="""\
Sample   Animal   Intelligence
1 Dog Smart
2 Dog Smart
3 Cat Dumb
4 Cat Dumb
5 Dog Dumb
6 Cat Smart"""

df = pd.read_table(StringIO(data),sep='\s+')
df

pd.crosstab(df.Animal,df.Intelligence,margins=True)
