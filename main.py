from codefiles import *

# Parameters
n = 100  # Number of nodes
k=3
method = lambda m,v: EMbased(m,v,k)

w,f =random_step_graphon_signal(k)

A,X,theta,mu = sample_from_graphon_signal(w,f,n)

theta_hat, mu_hat  = method(A, X)

w_matrix = blockify_graphon(w,n)
theta_hat_aligned = align_graphon(theta_hat,w_matrix)
f_vector = blockify_signal(f,n)
mu_hat_aligned = align_signal(mu_hat,f_vector)

visualize(w,f,theta_hat_aligned,mu_hat_aligned)
#benchmark_error(A,X,theta,mu,w,f,method)