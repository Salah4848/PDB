from codefiles import *

# Parameters
n = 200 # Number of nodes
k=20

w,f =random_step_graphon_signal(k,aligned=True)
w = lambda x,y: np.cos(10*x)*np.sin(10*x) + np.sin(10*y)*np.cos(10*y)
f = make_diff_signal(w)


A,X,theta,mu,xi = sample_from_graphon_signal(w,f,n)

method2 = lambda m,v: FANSbased(m,v)
method3 = lambda m,v: VEMbasedV(m,v,k)


methods =[method3]

f_vect = blockify_signal(f,n)
w_mat = blockify_graphon(w,n)
perm = np.argsort(xi)
X_aligned = X[perm]
A_aligned = A[perm,:]
A_aligned = A_aligned[:,perm]
pairs = [("True",f_vect,w_mat),("Empirical",X_aligned,A_aligned)]

for method in methods:
    theta_hat,mu_hat,name = method(A,X)
    #theta_hat = np.flipud(theta_hat)
    theta_hat,mu_hat = align_graphon_signal(theta_hat,mu_hat,w_mat,f_vect)
    pairs.append((name,mu_hat,theta_hat))

plot_arrays(pairs)
#benchmark_error(A,X,theta,mu,w,f,method[0])