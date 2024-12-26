import numpy as np
from scipy.linalg import eigh
from sklearn.mixture import GaussianMixture
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

def VEMbasedV(A, X, K, sort=True, cluster=False, outputall = False, giveperm = False,max_iter=100, tol=1e-6, fixed_point_iter=100):
    """
    Vectorized Variational EM for joint Stochastic Block Model. In this implemetation we ignore that j!=i for certain calculation. It is also highly numerically unstable, so must be run multiple times.
    For faster (pytorch) more proffessional code see: https://github.com/bastienlc/SBM-EM.git
    inputs:
        A: Adjacency matrix (n x n).
        X: Node signals (n x 1).
        K: Number of blocks.
    outputs:
        theta_hat, mu_hat, name
    """
    n = A.shape[0]

    tau = np.random.dirichlet(np.ones(K), size=n)  # (n, K)
    pi = tau.mean(axis=0)
    numerator = tau.T @ A @ tau
    denominator = tau.T @ np.ones((n, n)) @ tau
    Q = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0) # here too ignored i!=j
    M = np.sum(tau * X[:, None], axis=0) / (tau.sum(axis=0)+0.01)

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
            log_weights -= log_weights.max(axis=1, keepdims=True) #To avoid numerical errors, the adjustement is compensated when normalizing
            tau = 0.8*tau + 0.2*np.exp(log_weights) #Slowing down convergence for stability
            tau /= (tau.sum(axis=1, keepdims=True)+1e-10)
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
        if giveperm:
            perm = np.argsort(z)
            return theta,mu,"VEM",perm
        if outputall:
            return Q,M,pi,z
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

def compute_ll(A,X,Q,M,pi,z):
    '''
    computes the complete data likelihood with given params
    '''
    log_likelihood = np.sum(
        A * np.log(Q[z[:, None], z]+1e-10) +
        (1 - A) * np.log(1 - Q[z[:, None], z]+1e-10)-0.5*(X-M[z])**2
    )

    log_prior_Z = np.sum(np.log(pi[z]+1e-10))

    return log_likelihood + log_prior_Z

def compute_icl(A, X, k, Q, M, pi,z):
    """
    Computes the ICL value for a given number of blocks.
    """
    n = A.shape[0]

    ll = compute_ll(A,X,Q,M,pi,z)

    # Penalize model complexity
    penalty = 0.5 * (k * (k + 3) / 2) * np.log(n * (n + 1) / 2) + \
              0.5 * (k - 1) * np.log(n)

    return ll- penalty

def select_num_blocks(A, X, method, max_blocks):
    """
    Determines the optimal number of blocks using the ICL criterion.
    inputs:
        A : nxn array adjacency
        X : n array signals
        method: must take as params (A,X,k) and output Q,M,pi,z
        maxblocks: lowerbound on the number of blocks
    ouputs:
        best_num_blocks : optimal number of blocks in given range according to ICL
    """
    n = A.shape[0]
    icl_values = []

    for k in range(1, max_blocks + 1):
        Q,M,pi,z = method(A, X, k)
        icl_value = compute_icl(A, X, k, Q, M, pi, z)
        icl_values.append(icl_value)

    best_num_blocks = np.argmax(icl_values) + 1
    return best_num_blocks

def select_best_iter(A, X, method,num_iter = 10):
    '''
    Runs whatever method you feed it multiple times and takes the result with the best likelihood
    inputs:
        method: must take params (A,X) and output Q,M,pi,z
    outputs:
        theta_hat,mu_hat
    '''
    lls = []
    params = []
    for i in range(num_iter):
        Q,M,pi,z = method(A,X)
        lls.append(compute_ll(A,X,Q,M,pi,z))
        params.append((Q,M,pi,z))
    best = np.argmax(lls)
    Q,M,pi,z = params[best]
    
    return Q,M,pi,z 

def VEMICL(A,X,sort=False,bound = 20):
    '''
    Uses ICL criterion to choose number of blocks
    '''
    method = lambda m,v,blocks: VEMref(m,v,blocks,outputall=True,num_iter=2)

    k = select_num_blocks(A,X,method,bound)
    print(k)

    if sort:
        theta_hat,mu_hat,_ = VEMbasedV(A,X,k,sort=True,cluster=False,outputall=False)
        return theta_hat,mu_hat,"VEM+ICL"

    theta_hat,mu_hat,_ = VEMbasedV(A,X,k,cluster=True,outputall=False)
    return theta_hat,mu_hat,"VEM+ICL"

def VEMref(A,X,k,outputall=False,num_iter=10):
    '''
    It runs the VEM algo multiple times and uses the ll to choose the best answer
    outputs:
        theta_hat,mu_hat,name
    '''
    method = lambda m,v: VEMbasedV(m,v,k,sort=False,cluster=True,outputall=True)

    Q,M,pi,z = select_best_iter(A,X,method,num_iter)
    if outputall: return  Q,M,pi,z
    z_1,z_2 = np.meshgrid(z, z)
    mu_hat = M[z]
    theta_hat = Q[z_1,z_2]

    return theta_hat,mu_hat,"VEM"

def FANS(A, X, lamb = 0,h_quantile=0.1):
    '''
    Adaptation of the FANS method itself adpated from neighborhood smoothing. I have attempted to vectorize it but the memory it takes makes the code even slower.
    '''
    n = A.shape[0]
    h_quantile = min(h_quantile, np.sqrt(np.log(n)/n))
    

    theta_hat = np.zeros_like(A, dtype=float)
    mu_hat = np.zeros(n)
    
    for i in range(n):
        #Compute differences between row i and all other rows
        diff = A[i] - A

        d_tilde = np.zeros(n)
        
        #Create mask for valid k indices (excluding i)
        k_mask = np.ones(n, dtype=bool)
        k_mask[i] = False
        
        #Compute dot products for all i' at once
        dots = diff @ A / n
        
        for i_prime in range(n):
            if i != i_prime:
                k_mask[i_prime] = False
                d_tilde[i_prime] = np.sqrt(np.max(np.abs(dots[i_prime, k_mask])) + lamb*(X[i]-X[i_prime])**2)
                k_mask[i_prime] = True
        
        nonzero_distances = d_tilde#[d_tilde > 0]
        if len(nonzero_distances) > 0:
            threshold = np.quantile(nonzero_distances, h_quantile)
            neighborhood = d_tilde <= threshold
            neighborhood[i] = True
            
            theta_hat[i] = np.mean(A[neighborhood], axis=0)
            mu_hat[i] = np.mean(X[neighborhood])
        else:
            theta_hat[i] = A[i]
            mu_hat[i]=X[i]
    
    theta_hat = (theta_hat + theta_hat.T) / 2
    
    return theta_hat, mu_hat, f"FANS_lb={lamb}"



def ir_ls(A, X, K, T=10, sigma=1, init_method='spectral'):
    '''
    Iterative method with least squares. Original code: https://github.com/glmbraun/CSBM
    '''
    n, d = X.shape

    def initialize_partition(A, X, K, method):
        if method == 'spectral':
            eigenvalues, eigenvectors = eigh(A, subset_by_index=[A.shape[0] - K, A.shape[0] - 1])
            features = np.hstack((eigenvectors, X))
            gmm = GaussianMixture(n_components=K, random_state=0).fit(features)
            labels = gmm.predict(features)
        else:  # Random initialization
            labels = np.random.randint(0, K, size=n)
        Z = np.zeros((n, K))
        Z[np.arange(n), labels] = 1
        return Z

    # Step 1: Initialize membership matrix
    Z = initialize_partition(A, X, K, init_method)

    for t in range(T):
        # Step 2: Estimate parameters
        n_k = Z.sum(axis=0)
        W = Z / np.maximum(n_k, 1e-10)  # Normalize columns
        Q = W.T @ A @ W
        M = W.T @ X

        # Compute Sigma_k for graph distance
        Sigma = np.zeros((K, K))
        for k in range(K):
            for k_prime in range(K):
                #if Q[k, k_prime] > 0:
                Sigma[k, k_prime] = n_k[k_prime] / (Q[k, k_prime]+1e-10)

        # Step 3: Partition refinement
        Z_new = np.zeros_like(Z)
        for i in range(n):
            distances = np.zeros(K)
            for k in range(K):
                graph_dist = np.linalg.norm((A[i, :] @ W - Q[k, :]) @ np.sqrt(np.diag(Sigma[k, :])))
                cov_dist = np.linalg.norm((X[i, :] - M[k, :]) / sigma)
                distances[k] = graph_dist**2 + cov_dist**2
            Z_new[i, np.argmin(distances)] = 1

        # Check convergence
        if np.allclose(Z, Z_new):
            break
        Z = Z_new
    
    z = np.argmax(Z, axis=1)
    z_1,z_2 = np.meshgrid(z,z)
    theta_hat = Q[z_1,z_2]
    mu_hat = M.T[0][z]
    return theta_hat,mu_hat,"IR-LS"
