import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.interpolate import BSpline
from sklearn.cluster import KMeans
from .utility import *

def VEMbased(A,X, K, max_iter=100, tol=1e-6):
    """
    Variational EM for joint Stochastic Block Model
    inputs:
        A: Adjacency matrix (n x n).
        X: Node signals (n x 1).
        k: Number of blocks.
    outputs:
        theta_hat,mu_hat,name
    """
    n = A.shape[0]
    
    #Initialize variational parameters and model parameters
    tau = np.random.dirichlet(np.ones(K), size=n)
    pi = np.ones(K) / K
    Q = np.random.uniform(0, 1, (K, K))
    M = np.array([X[np.random.choice(n)] for _ in range(K)])
    for _ in range(max_iter):
        tau_prev = tau.copy()
        #E-step: Update variational parameters
        for i in range(n):
            log_weights = np.log(pi)
            for k in range(K):
                for j in range(n):
                    if i != j:
                        if A[i,j] == 1:
                            log_weights[k] += np.sum(tau[j,l] * np.log(Q[k,l]) for l in range(K))
                        else:
                            log_weights[k] += np.sum(tau[j,l] * np.log(1 - Q[k,l]) for l in range(K))
                log_weights[k] -= 0.5 * ((X[i] - M[k])**2)
            tau[i] = np.exp(log_weights)
            tau[i] /= tau[i].sum()
        
        # M-step: Update model parameters
        #Update block weights
        pi = tau.mean(axis=0)
        
        #Update block probability matrix
        for k in range(K):
            for l in range(K):
                num = den = 0
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            expected_edge = tau[i,k] * tau[j,l]
                            num += expected_edge * A[i,j]
                            den += expected_edge
                
                Q[k,l] = num / den if den > 0 else 0
        
        #Update block means
        for k in range(K):
            M[k] = np.sum(tau[:,k] * X) / tau[:,k].sum()
        
        #Check convergence
        if np.max(np.abs(tau - tau_prev)) < tol:
            break
    perm = np.argsort(np.diagonal(Q))
    Q = Q[perm,:]
    Q = Q[:,perm]
    pi = pi[perm]
    M = M[perm]
    thresholds = np.cumsum(pi)
    est_graphon = make_step_graphon(thresholds[:-1],Q)
    est_signal = make_step_signal(thresholds[:-1],M)
    theta = blockify_graphon(est_graphon,n)
    mu = blockify_signal(est_signal,n)
    return theta,mu,"VEM"

def VEMbasedV(A, X, K, sort=True, cluster=False, max_iter=100, tol=1e-6, fixed_point_iter=50):
    """
    Vectorized Variational EM for joint Stochastic Block Model. In this implemetation we ignore that j!=i for certain calculation.
    inputs:
        A: Adjacency matrix (n x n).
        X: Node signals (n x 1).
        K: Number of blocks.
    outputs:
        theta_hat, mu_hat, name
    """
    n = A.shape[0]

    tau = np.random.dirichlet(np.ones(K), size=n)  # (n, K)
    pi = np.ones(K) / K  # (K,)
    Q = np.random.uniform(0.1, 0.9, (K, K))  # (K, K)
    M = np.array([X[np.random.choice(n)] for _ in range(K)])  # (K,)

    for _ in range(max_iter):
        tau_prev = tau.copy()
        #E-step: Update variational parameters
        log_pi = np.log(pi+1e-10)
        log_Q = np.log(Q+1e-10)
        log_1_minus_Q = np.log(1 - Q+1e-10)
        for _ in range(fixed_point_iter):
            tau_prevfix = tau.copy()

            adjacency_term = A @ (tau @ log_Q.T) + (1 - A) @ (tau @ log_1_minus_Q.T) #ignored i!=j

            signal_term = -0.5 * ((X[:, None] - M[None, :]) ** 2)  

            log_weights = log_pi + adjacency_term + signal_term  
            log_weights -= np.nan_to_num(log_weights.max(axis=1, keepdims=True)) #log_weights -= log_weights.max(axis=1, keepdims=True) #To avoid numerical errors, the adjustement is compensated when normalizing
            tau = np.exp(log_weights)
            tau /= tau.sum(axis=1, keepdims=True)+1e-10
            if np.isnan(tau).any() or np.isinf(tau).any():
                raise ValueError("Numerical instability in tau update")
            if np.max(np.abs(tau - tau_prevfix)) < tol:
                break

        #M-step: Update model parameters
        pi = tau.mean(axis=0)

        numerator = tau.T @ A @ tau
        denominator = tau.T @ np.ones((n, n)) @ tau
        Q = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0) # here too ignored i!=j

        M = np.sum(tau * X[:, None], axis=0) / (tau.sum(axis=0)+0.01)

        if np.max(np.abs(tau - tau_prev)) < tol:
            break
    if cluster:
        z = np.argmax(tau, axis = 1)
        z_1,z_2 = np.meshgrid(z, z)
        mu = M[z]
        theta = Q[z_1,z_2]
        return theta, mu, "ClustVEM"

    if sort:
        perm = np.argsort(np.diagonal(Q))
        Q = Q[perm][:, perm]
        pi = pi[perm]
        M = M[perm]
    thresholds = np.cumsum(pi)
    est_graphon = make_step_graphon(thresholds[:-1], Q)
    est_signal = make_step_signal(thresholds[:-1], M)
    theta = blockify_graphon(est_graphon, n)
    mu = blockify_signal(est_signal, n)

    return theta, mu, "VEM"

def FANSbased(A, X):
    '''
    Uses FANS algorithm (vectorized).
    Inputs:
        X : 1xn array, observed signal data
        A : nxn array, observed adjacency matrix
    Outputs:
        theta_hat, mu_hat, name
    '''
    n = len(X)

    # Compute dg and df using broadcasting
    dg = np.zeros((n, n))
    df = np.zeros((n, n))

    for i in range(n):
        temp1 = np.delete(A, i, axis=1)
        temp2 = np.delete(X, i, axis=0)
        dg[i] = np.max(np.abs((A[:, i][:, np.newaxis] - A).dot(temp1)), axis=1) / n
        df[i] = np.max(np.abs((X[i] - X)[:, np.newaxis] * temp2), axis=1)

    # Symmetrize dg and df
    dg = (dg + dg.T) / 2
    df = (df + df.T) / 2

    # Combined distance
    lamb = 0
    c = 1
    d = dg + lamb * df
    h = c * np.sqrt(np.log(n) / n)

    # Initialize outputs
    theta_hat = np.zeros_like(A)
    mu_hat = np.zeros_like(X)

    # Compute theta_hat and mu_hat
    for i in range(n):
        # Find neighbors for graphon
        temp = np.delete(d[i], i)
        q = np.quantile(temp, h)
        neighbors = np.where(temp <= q)[0]
        neighbors = neighbors[neighbors != i]
        size = len(neighbors)

        # Estimate mu_hat[i]
        if size > 0:
            mu_hat[i] = np.mean(X[neighbors])  # Average over neighbors
        else:
            mu_hat[i] = X[i]  # Fallback to self if no neighbors

        for j in range(i + 1):
            if size > 0:
                # Estimate theta_hat[i, j]
                theta_hat[i, j] = (np.sum(A[neighbors, j]) / size +
                                   np.sum(A[i, neighbors]) / size) / 2
            else:
                theta_hat[i, j] = A[i, j]  # Fallback to original value
            theta_hat[j, i] = theta_hat[i, j]  # Ensure symmetry

    return theta_hat, mu_hat, "FANS"

