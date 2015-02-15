'''
Created on 08-Feb-2015

@author: adarsh
'''
import numpy as np

m = 150
d = 75

X = np.random.rand(m, d)
theta = np.hstack((np.array([10.0]*10),np.zeros(65)))

y = X.dot(theta) + 0.1 * np.random.randn(m)

np.savetxt('X.txt', X)
np.savetxt('y.txt', y.T)
