#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np


# In[9]:


#creating some random data to test
x = np.random.randn(100000)
y = 3*x + 2


# In[10]:


#loss function with numpy
def loss_1(a,b,x,y):
    J = 1/2 * sum((a*x + b)-y)**2
    return J

#loss function with float
def loss_2(a,b,x,y):
    J = 0
    for i in range(len(x)):
        J += ((a * float(x[i]) + b) - float(y[i]))**2
    return J/2


# In[16]:


#learning rate is intuitively found related to the scale of the data
def grad_des(x, y,cost=loss_1, a=0, b=0, learning_rate = 0.000001):
    #initialize some parameters
    djda = 0
    djdb = 0
    losses = [] # to see the change in losses per iteration   
    # iterating till convergence
    for i in range(400):
        a = a - learning_rate*djda
        b = b - learning_rate*djdb
        losses.append(cost(a,b,x,y))
        djda = float(sum(((a*x+b)-y)*x))
        djdb = float(sum((a*x + b)-y))
        print(f"a:{a}, b:{b}, loss:{cost(a,b,x,y)}")  
    return cost(a,b,x,y),a,b


# In[17]:


grad_des(x,y) #with the loss_1 function


# In[18]:


grad_des(x,y,loss_2)


# In[ ]:




