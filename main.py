from codefiles import *

n=40

f = lambda x : 10*x*x
w = lambda x,y : (x+y)/2

A,X = sample_from_graphon_signal(w,f,n)

theta_hat, mu_hat = estimate_graphon_signal(A,X)

w_matrix = blockify_graphon(w,n)
f_matrix = blockify_signal(f,n)

aligned_theta_hat = align_graphon(theta_hat, w_matrix)
aligned_mu_hat = align_signal(mu_hat, f_matrix)

visualize(w,f,aligned_theta_hat,aligned_mu_hat)