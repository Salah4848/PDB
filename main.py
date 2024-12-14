from codefiles import *

# Parameters
n = 100  # Number of nodes
k=4
method1 = lambda m,v: CVEMbased(m,v,k)
method2 = lambda m,v: FANSbased(m,v)
method3 = lambda m,v: VEMbased(m,v,k)


methods =[method1,method2,method3]

w,f =random_step_graphon_signal(k,aligned=True)

A,X,theta,mu = sample_from_graphon_signal(w,f,n)

pairs = [("true",blockify_signal(f,n),blockify_graphon(w,n))]

for method in methods:
    theta,mu,name = method(A,X)
    pairs.append((name,mu,theta))

plot_arrays(pairs)