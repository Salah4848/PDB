from codefiles import *

n=40
theta = np.arange(n*n).reshape(n, n)
mu = np.arange(n)

f = lambda x : x*x
w = lambda x,y : x+y/2

visualize(w,f,theta,mu)