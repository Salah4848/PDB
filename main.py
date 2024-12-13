from codefiles import *

# Parameters
n = 300  # Number of nodes
k=3
method = lambda m,v: EMbased(m,v,k,blockoutput=True)
#method = lambda m,v: FANSbased(m,v)

w,f =random_step_graphon_signal(k,aligned=True)


A,X,theta,mu = sample_from_graphon_signal(w,f,n)

theta_hat, mu_hat  = method(A, X)

'''w_matrix = blockify_graphon(w,n)
theta_hat = align_graphon(theta_hat,w_matrix,diagonly=True)
f_vector = blockify_signal(f,n)
mu_hat = align_signal(mu_hat,f_vector)'''

visualize(w,f,theta_hat,mu_hat)
#benchmark_error(A,X,theta,mu,w,f,method)
