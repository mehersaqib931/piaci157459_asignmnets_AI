#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[2]:


import numpy as np
arr=np.arange(10).reshape(2,5)
print(arr)


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[3]:


arr_1=np.ones(10).reshape(2,5)
arr=np.arange(10).reshape(2,5)
np.vstack((arr,arr_1))


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[5]:


np.hstack((arr,arr_1))


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[6]:


arr.flatten()


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[8]:


arr.ravel()


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[9]:


arr1=np.arange(15).reshape(5,3)
print(arr1)


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[10]:


arr2=np.arange(25).reshape(5,5)
np.square(arr2)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[11]:


arr3=np.arange(30).reshape(5,6)
np.mean(arr3)


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[12]:


np.std(arr3)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[13]:


np.median(arr3)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[14]:


arr3.T


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[15]:


arr4=np.arange(16).reshape(4,4)
np.trace(arr4)


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[16]:


np.linalg.det(arr4)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[17]:


print(np.percentile(arr4,5))
print(np.percentile(arr4,95))


# ## Question:15

# ### How to find if a given array has any null values?

# In[18]:


arr4==0


# In[ ]:




