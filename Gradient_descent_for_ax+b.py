#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np
import matplotlib.pyplot as plt


# In[133]:


#creating some random data to test
x = np.random.randn(10)
y = 3*x + 2


# In[135]:


#loss function
def loss(a,b,x,y):
    J = 1/2 * sum((a*x + b)-y)**2
    return J


# In[139]:


def grad_des(a,b,x,y,learning_rate = 0.001):
    #initialize some parameters
    a = 0
    b = 0
    djda = 0
    djdb = 0
    losses = [] # to see the change in losses per iteration   
    # iterating till convergence
    for i in range(4000):
        a = a - learning_rate*djda
        b = b - learning_rate*djdb
        losses.append(loss(a,b,x,y))
        djda = sum(((a*x+b)-y)*x)
        djdb = sum((a*x + b)-y)
    print(f"a:{a}, b:{b}, loss:{loss(a,b,x,y)}")  
    return loss(a,b,x,y),a,b,losses

