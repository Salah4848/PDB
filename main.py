from codefiles import *

# Parameters
n = 500 # Number of nodes
k= 10

w,f =random_step_graphon_signal(k,aligned=True)
'''f = lambda x: 2*np.cos(10*x)+16
w = make_dist_graphon(f, g=lambda x: x/21)'''

A,X,theta,mu,xi = sample_from_graphon_signal(w,f,n)

method2 = lambda m,v: FANS(m,v,lamb=0)
method3 = lambda m,v: VEMbasedV(m,v,k,sort=False,cluster=True)
method4 = lambda m,v:  (m,v,"empirical") #empiricl esitmate
method5 = lambda m,v: VEMICL(m,v,sort=False)
method6 = lambda m,v: VEMref(m,v,k)
method7 = lambda m,v: ir_ls(m,v[:,None],k)


methods = [method3,method7] #[method7, method6, method2,method5]


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


""" n=100

w_1 = lambda x,y: 0.5*(x+y)
f_1 = lambda x: 5*x+10

w_2 = lambda x,y: abs(x-y)
f_2 = make_diff_signal(w_2,initconstant=2,sequence=lambda x: 10*x)

w_3 = lambda x,y: np.sin(5*np.pi*(x + y - 1) + 1) /2 + 0.5
temp = make_diff_signal(w_3,initconstant=2,sequence=lambda x: x)
f_3 = lambda x: 10+temp(x)

f_4 = lambda x: 2*np.cos(10*x)+16
w_4 = make_dist_graphon(f_4, g=lambda x: x/21)

funcs = [(w_1,f_1,"pair 1"),(w_2,f_2,"pair 2"),(w_3,f_3,"pair 3"),(w_4,f_4,"pair 4")] """