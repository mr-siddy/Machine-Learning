#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# In[3]:


a = np.array([1,2,3])
b = np.array([4,5,6,7])

a,b = np.meshgrid(a,b)

print(a)

print(b)


# In[15]:


# making a 3D surface  ---> Surface Plot


 # <--- This is important for 3d plotting 

#your code

fig = plt.figure()
axes = fig.gca(projection='3d')

axes.plot_surface(a,b,a**2+b**2,cmap = 'rainbow')


# In[20]:


a1 = np.arange(-1,1,0.02)
b1 = a1

a1,b1 = np.meshgrid(a1,b1)

#print(a1)

#print(b1)

new_fig = plt.figure()
new_ax = new_fig.gca(projection = '3d')
new_ax.plot_surface(a1,b1,a1**2+b1**2, cmap = 'rainbow')


# In[24]:


# countour plot
   
fig = plt.figure()
axes = fig.gca(projection='3d')

axes.contour(a1,b1,a1**2+b1**2, cmap = 'rainbow')
plt.title('Contour plot')
plt.show()


# In[ ]:




