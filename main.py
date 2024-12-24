from codefiles import *

# Parameters
n = 500 # Number of nodes
k= 5

w,f =random_step_graphon_signal(k,aligned=True)


A,X,theta,mu,xi = sample_from_graphon_signal(w,f,n)

method2 = lambda m,v: FANS(m,v,lamb=0)
method3 = lambda m,v: VEMbasedV(m,v,k,sort=False,cluster=True)
method4 = lambda m,v:  (m,v,"empirical") #empiricl esitmate
method5 = lambda m,v: VEMICL(m,v,sort=False)
method6 = lambda m,v: VEMref(m,v,int(np.sqrt(m.shape[0])))
method7 = lambda m,v: ir_ls(m,v[:,None],int(np.sqrt(m.shape[0])))


methods = [method7, method6, method2,method5]


f_vect = blockify_signal(f,n)
w_mat = blockify_graphon(w,n)

pairs = [("True",f_vect,w_mat)]


benchmark_error(A,X,theta,mu,methods)

for method in methods:
    theta_hat,mu_hat,name = method(A,X)
    #theta_hat = np.flipud(theta_hat)
    theta_hat,mu_hat = align_graphon_signal(theta_hat,mu_hat,w_mat,f_vect,xi,usegraphon=False,uselatents=True)
    pairs.append((name,mu_hat,theta_hat))

plot_arrays(pairs)


## List of graphon functions to test:
# np.cos(10*x)*np.sin(10*x) + np.sin(10*y)*np.cos(10*y) bad performance
# abs(x-y) good performance
# 0.25*(x**2+y**2+np.sqrt(x)+np.sqrt(y)) good performance
# np.cos(10*(x+y)) bad performance
# no.sin(5*np.pi*(x + y - 1) + 1) /2 + 0.5