########## Data Vizualization ##########
%reset
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
os.chdir('C:/Users/sdt144/Dropbox/SamPython')
os.getcwd()
# install seaborn

from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import randn

##### Histograms
dataset1 = randn(100)
plt.hist(dataset1) # 10 equally-spaced bins

dataset2 = randn(80)
plt.hist(dataset2,color='indianred')

#overlaid hists
plt.hist(dataset1,normed=True,color='indianred',alpha=0.5,bins=20)
plt.hist(dataset2,normed=True,alpha=0.5,bins=20)

data1 = randn(1000)
data2 = randn(1000)
sns.jointplot(data1,data2) # scatter plot with hists

sns.jointplot(data1,data2,kind='hex')


##### Kernel Density Estimation
dataset = randn(25)
sns.rugplot(dataset)

plt.hist(dataset,alpha=0.3)
sns.rugplot(dataset)

sns.rugplot(dataset)

x_min = dataset.min() - 2
x_max = dataset.max() + 2
x_axis = np.linspace(x_min,x_max,100)
bandwidth = ((4*dataset.std()**5)/(3*len(dataset)))**.2 # see wikipedia

kernel_list = []
for data_point in dataset:
    
    # Create a kernel for each point and append to list
    kernel = stats.norm(data_point,bandwidth).pdf(x_axis)
    kernel_list.append(kernel)
    
    #Scale for plotting
    kernel = kernel / kernel.max()
    kernel = kernel * .4
    
    plt.plot(x_axis,kernel,color = 'grey',alpha=0.5)

plt.ylim(0,10)

sum_of_kde = np.sum(kernel_list,axis=0)
fig = plt.plot(x_axis,sum_of_kde,color='indianred')
sns.rugplot(dataset,c = 'indianred')
#plt.yticks([])
plt.suptitle("Sum of the Basis Functions")

#easy way using seaborn
sns.kdeplot(dataset)

# see the effect of changed bandwidth
sns.rugplot(dataset,color='black')
for bw in np.arange(0.5,2,0.25):
    sns.kdeplot(dataset,bw=bw,lw=1.8,label=bw)

# check different kernel functions
# see https://en.wikipedia.org/wiki/Kernel_(statistics)
# need to install 'statsmodels' using:
# 'easy_install statsmodels' from the shell

kernel_options = ["biw", "cos", "epa", "gau", "tri", "triw"]
for kern in kernel_options:
    sns.kdeplot(dataset,kernel=kern,label=kern)

for kern in kernel_options:
    sns.kdeplot(dataset,kernel=kern,label=kern,shade=True,alpha=0.5) #shade

sns.kdeplot(dataset,vertical=True) #Vertical

sns.kdeplot(dataset,cumulative=True) #CDF

#Multivariate
mean = [0,0]
cov = [[1,0],[0,100]]
dataset2 = np.random.multivariate_normal(mean,cov,1000)
df = pd.DataFrame(dataset2,columns=['X','Y'])
sns.kdeplot(df) #2D kernel as contour plot
sns.kdeplot(df.X,df.Y,shade=True)

sns.kdeplot(df,bw='silverman') #method to select band width (bw)
sns.kdeplot(df,bw=1)

sns.jointplot('X','Y',df,kind='kde') #kernel+mraginal pdf

### Combining plot styles
dataset = randn(100)
sns.distplot(dataset,bins=25) #kernel + hist

sns.distplot(dataset,rug=True,hist=False)

sns.distplot(dataset,bins=25,
             kde_kws={'color':'indianred','label':'KDE PLOT'},
             hist_kws={'color':'blue','label':"HISTOGRAM"})

ser1 = Series(dataset,name='My_DATA') # also work with Series
sns.distplot(ser1,bins=25)

##### Box and Violin Plots
seed=123
data1 = randn(100)
data2 = randn(100)

sns.boxplot(data=[data1,data2])
sns.boxplot(data=[data1,data2],whis=np.inf) #extending whiskers
sns.boxplot(data1, vert = False)

# violin
%matplotlib inline
data1 = stats.norm(0,5).rvs(100)
data2 = np.concatenate([stats.gamma(5).rvs(50)-1,-1*stats.gamma(5).rvs(50)])
sns.boxplot(data=[data1,data2],whis=np.inf)
sns.violinplot(data=[data1,data2])

sns.violinplot(data2,bw=0.01)

sns.violinplot(data1,inner='stick') #adding rugs


##### Regression plots
tips = sns.load_dataset('tips')
tips.head()

sns.lmplot('total_bill','tip',tips) #scatter+linear fit

sns.lmplot('total_bill','tip',tips,
          scatter_kws={'marker':'o','color':'indianred'},
          line_kws={'linewidth':1,'color':'blue'})

sns.lmplot('total_bill','tip',tips,order=4,   # change order of the fit
          scatter_kws={'marker':'o','color':'indianred'},
          line_kws={'linewidth':1,'color':'blue'})

sns.lmplot('total_bill','tip',tips,fit_reg=False) # no fit

tips.head()
tips['tip_perct']=100*(tips['tip']/tips['total_bill'])
tips.head()

sns.lmplot('size','tip_perct',tips) # discrete predictor

sns.lmplot('size','tip_perct',tips,x_estimator=np.mean) #adding mean and confidence interval

sns.lmplot('total_bill','tip_perct',tips,hue='sex',markers=['x','o']) #conditioning
sns.lmplot('total_bill','tip_perct',tips,hue='day')

#locally-weighted scatterplot smoothing (LOESS)
sns.lmplot('total_bill','tip_perct',tips,lowess=True,line_kws={'color':'black'})

sns.regplot('total_bill','tip_perct',tips) #lower-level function that lmplot() uses

#h Subplots
fig, (axis1,axis2) = plt.subplots(1,2,sharey=True)
sns.regplot('total_bill','tip_perct',tips,ax=axis1)
sns.violinplot(x='size',y='tip_perct',data=tips.sort('size'),color='Red',ax=axis2)

########## Heatmaps and clustered matrices
flight_df = sns.load_dataset('flights')
flight_df.head()

flight_df = flight_df.pivot('month','year','passengers')
flight_df

sns.heatmap(flight_df)

sns.heatmap(flight_df,annot=True,fmt='d') # add numbers too

sns.heatmap(flight_df,center=flight_df.loc['January',1955],
           annot=True,fmt='d') #chose your own center

##
yearly_flights = flight_df.sum() #sum along columns

years = pd.Series(yearly_flights.index.values)
years = pd.DataFrame(years)
flights = pd.Series(yearly_flights.values) 
flights = pd.DataFrame(flights)
year_df = pd.concat((years,flights),axis=1)
year_df.columns = ['Year','Flights']
year_df

f, (axis1,axis2) = plt.subplots(2,1)
sns.barplot(x='Year',y='Flights',data=year_df,ax = axis1)
sns.heatmap(flight_df,cmap='Blues',ax=axis2,cbar_kws={"orientation": "horizontal"})

### cluster maps
%matplotlib qt
sns.clustermap(flight_df)

sns.clustermap(flight_df,col_cluster=False) #uncluster the columns

sns.clustermap(flight_df,standard_scale=1) # by columns - extract the column mean
sns.clustermap(flight_df,standard_scale=0) # by row

sns.clustermap(flight_df,z_score=1) # standardizing

