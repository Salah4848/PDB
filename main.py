from codefiles import *

# Parameters
n = 100  # Number of nodes
k=5

w,f =random_step_graphon_signal(k,aligned=True)
w = lambda x,y: np.cos(10*(x+y))
f = lambda x: 10*np.cos(10*(x+x))
A,X,theta,mu,xi = sample_from_graphon_signal(w,f,n)

method1 = lambda m,v: CVEMbased(m,v,k)
method2 = lambda m,v: FANSbased(m,v)
method3 = lambda m,v: VEMbasedV(m,v,k,False)
method4 = lambda m,v: FANSbased_vectorized(m,v)

methods =[method4,method3]

f_vect = blockify_signal(f,n)
w_mat = blockify_graphon(w,n)
perm = np.argsort(xi)
X_aligned = X[perm]
A_aligned = A[perm,:]
A_aligned = A_aligned[:,perm]
pairs = [("True",f_vect,w_mat),("Empirical",X_aligned,A_aligned)]

for method in methods:
    theta,mu,name = method(A_aligned,X_aligned)
    #theta,mu = align_graphon_signal(theta,mu,w_mat,f_vect)
    pairs.append((name,mu,theta))

plot_arrays(pairs)