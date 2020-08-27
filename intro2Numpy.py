##### Intrroduction to Numpy #####
%reset
import os
os.system('cls')


import numpy as np
my_list1=[1,2,3,4]
my_array1=np.array(my_list1) 
my_array1
my_list2=[11,22,33,44]
my_list3=[my_list1,my_list2]
my_array3=np.array(my_list3)
my_array3

my_array3.shape
my_array3.dtype

my_array4=np.zeros(5)
my_array4
my_array4.dtype

my_array5=np.ones([5,5])
my_array5

np.eye(5)
np.arange(5)
np.arange(5,12,2)

### array operations
arr1=np.array([[1,2,3,4],[8,10,12,14]])
arr2=np.array([[0,2,0,4],[18,20,22,44]])
arr1+arr2
arr1*arr2 #elementwise multiplication 
1/arr1
from __future__ import division
1/arr1
arr1**2

### indexing
arr=np.arange(0,11)
arr
arr[1:4]
arr[0:4]=100
arr
arr=np.arange(0,11)
slice_arr=arr[0:4]
slice_arr
slice_arr[:]=99
slice_arr
arr
arr_copy=arr.copy()

## 2D arrays
arr_2d=np.array([[5,10,15],[33,43,53],[70,80,90]])
arr_2d
arr_2d[1] #only one index will refer to row
arr_2d[0,1] 
arr_2d[0][1] # both work
arr_2d
arr_2d[:2,1:]

## fancy
arr2d=np.zeros([10,10])
arr2d
arr_len=arr2d.shape[1]
for i in range(arr_len):
    arr2d[i]=i
arr2d
arr2d[1,1]
arr2d[[1,3]] # rows two and four
arr2d
arr2d[:,[0,1,3]] # all rows of columns 1, 2, and 4

### transpose
arr=np.array([[1,2,3,4]])
arr
arr.T
np.dot(arr,arr.T) #for vectors, np.dot() performs inner product no matter
                    # the shape of the vector row, or column
np.dot(arr.T,arr)
np.outer(arr,arr.T) # outer product of vectors

arr=np.arange(50).reshape(10,5)
arr
arr.T
np.dot(arr.T,arr) # for matrices, np.dot() performs matrix multiplication
                    #arr.T*arr
np.dot(arr,arr.T) # arr*arr.T
arr.T.dot(arr) # same

### matrix
x = np.matrix( [[2,3], [3, 5]] )
x
y = np.matrix( [[1,2], [5, -1]] )
y
np.dot(x,y)
x*y    # same


### 3D array
arr=np.arange(50).reshape(2,5,5)
arr

### Universal array functions
arr=np.arange(0,11)
arr
np.sqrt(arr)
np.exp(arr)

A=np.random.randn(5)
A
B=np.random.rand(5)
B
np.outer(A,B)
np.add(A.T,B)
np.maximum(A,B) # element by element max. between the two arrays
np.minimum(A,B)

### some vizualization
import matplotlib.pyplot as plt
%matplotlib inline

points=np.arange(-5,5,0.1)
dx,dy=np.meshgrid(points,points)
dx
dy
z=np.sin(dx)+np.sin(dy)
z.shape
plt.imshow(z)

plt.imshow(z) # run simultaneously
plt.colorbar()
plt.title('z=sin(x)+sin(y)')

### array processing
A=np.array([1,2,3,4])
B=np.array([100,200,300,400])
Condition=np.array([False,False,True,True])
ans=[(A_val if Cond else B_val) for A_val,B_val,Cond in zip(A,B,Condition) ]
ans

ans2=np.where(Condition,A,B) #simpler
ans2
from numpy.random import randn
arr=randn(5,5)
arr
np.where(arr<0,0,arr)

arr=np.array([[1,2,3],[4,5,6],[7,8,9]])
arr
arr.sum()
arr.sum(0) #sum along the columns
arr.sum(1) #sum along the rows

arr.mean()
arr.mean(0)
arr.std()
arr.var()

arr=randn(5)
arr
arr.sort()
arr

### some boolean functions
bool_arr=np.array([True,False,True,False])
bool_arr.any()
bool_arr.all()

### some set operators
countries=np.array(['France','Germany','USA','Russia','USA','Mexico','Germany'])
np.unique(countries) # eliminate replications

np.in1d(['France','USA','Sweden'],countries) # check membership

### Saving numpy data file
import os
os.chdir('C:/Users/sdt144/Dropbox/SamPython')
os.getcwd()

#save to .npy file
arr=np.arange(5)
arr
np.save('myarrayfile',arr)
arr=np.arange(10)
np.load('myarrayfile.npy') # extension is need to load

#save to .npz file
arr1=np.arange(5)
arr2=np.arange(10)
np.savez('myarrays.npz',x=arr1,y=arr2)

os.listdir(os.getcwd()) # see the content of the current directory

array_archive=np.load('myarrays.npz')
array_archive['x']

#save as .txt
arr=np.array([[1,2,3],[4,5,6]])
arr
np.savetxt('mytextarray.txt',arr,delimiter=',')
os.listdir(os.getcwd())
np.loadtxt('mytextarray.txt',delimiter=',')

