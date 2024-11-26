import matplotlib.pyplot as plt
import numpy as np
from .algorithms import *

def simplify(matrix, k):
    """
    Convert an n x n matrix into a k x k block matrix by averaging blocks using vectorized operations. https://stackoverflow.com/questions/29435842/simplify-matrix-by-averaging-multiple-cells
    If k does not divide n then it discards the last rows and columns of the matrix to make it work.
    inputs:
    - matrix: The input n x n matrix
    - k: The number of blocks per dimension
    
    outputs:
    - block_matrix: The resulting k x k block matrix
    """

    n = matrix.shape[0]

    if n%k!=0:
        n = n - (n%k)
        matrix = matrix[:n,:n]
    
    block_size = n // k
    
    reshaped = matrix.reshape(k, block_size, k, block_size)

    transposed = reshaped.transpose(0, 2, 1, 3)
    
    block_matrix = transposed.mean(axis=(2, 3))
    
    return block_matrix

def visualize(graphon, signal, est_graphon, est_signal, n_points=100):
    '''
    Plots a nice 4x4 grid.
    inputs:
        graphon: must be function R^2 to R.
        signal: must be a function R to R.
        est_graphon: must be a square 2D array.
        est_signal: must be a 1D array
    '''
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)
    w = graphon(X,Y)

    t = np.linspace(0,1,n_points)
    f = signal(t)
    tstep = np.linspace(0,1,len(est_signal)+1)
    est_signal=np.append(est_signal,[est_signal[-1]])

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.title("Estimated Graphon (theta Hat)")
    plt.imshow(est_graphon, cmap='viridis',vmin=0,vmax=1)
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("True Graphon")
    plt.imshow(w, cmap='viridis',vmin=0,vmax=1)
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("Estimated Signal (mu_hat)")
    plt.plot(tstep, est_signal, drawstyle='steps-post')

    plt.subplot(2, 2, 4)
    plt.title("True Signal")
    plt.plot(t,f)


    plt.tight_layout()
    plt.show()

def sample_from_graphon_signal(w, f, n, symmetric=True, self_loops=False):
    '''
    Samples an nxn adjacency matrix from the graphon W and signal values from signal f with added standard gaussian noise.
    outputs:
        (A,X,theta,mu)
    '''
    #Sampling the ξ
    vertices = np.random.uniform(0, 1, n)

    #Graph sampling
    u, v = np.meshgrid(vertices, vertices)
    theta = w(u, v)
    A = np.random.random((n, n)) < theta

    #Signal sampling
    mu = f(vertices)
    epsilon = np.random.normal(0,1,n)
    X = mu + epsilon

    if symmetric:
        upper = np.triu(A, k=1)
        A = upper + upper.T
    
    if not self_loops:
        np.fill_diagonal(A, 0)
        
    return (A.astype(int).astype(float), X, theta, mu)

def blockify_graphon(graphon, n):
    '''
    Turn our graphon into an n-block function. Used for alignement
    outputs:
        graphon_matrix: nxn 2D array
    '''
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    u, v = np.meshgrid(x, y)
    graphon_matrix = graphon(u,v)
    return graphon_matrix
    
def blockify_signal(signal,n):
    '''
    Turn our signal into an n-step function. Used for alignement
    outputs:
        signal_matrix: nxn 2D array
    '''
    x = np.linspace(0, 1, n)
    signal_matrix = signal(x)
    return signal_matrix

def squared_norm_matrix(M):
    return np.sum(np.square(M))

def squared_norm_vector(v):
    return np.dot(v,v)

def benchmark_error(A, X, theta, mu, graphon, signal, graph_it=True):
    '''
    Graphs the error(s) against the amount of samples used. Used to check rates.
    '''
    N = len(X)
    error_equiv_graphon = np.zeros(N-3)
    error_equiv_signal = np.zeros(N-3)
    error_prob_matrix = np.zeros(N-3)
    error_mean_vector = np.zeros(N-3)

    for n in range(3,N):
        Xn = X[:n]
        An = A[:n,:n]
        thetan = theta[:n,:n]
        mun = mu[:n]

        theta_hat, mu_hat = estimate_graphon_signal(An, Xn)

        w_matrix = blockify_graphon(graphon, n)
        f_matrix = blockify_signal(signal, n)
        aligned_theta_hat = align_graphon(theta_hat, w_matrix)
        aligned_mu_hat = align_signal(mu_hat, f_matrix)

        error_equiv_graphon[n-3] = squared_norm_matrix(aligned_theta_hat-w_matrix)/(n*n)
        error_equiv_signal[n-3] = squared_norm_vector(aligned_mu_hat-f_matrix)/n
        error_prob_matrix[n-3] = squared_norm_matrix(theta_hat-thetan)/(n*n)
        error_mean_vector[n-3] = squared_norm_vector(mu_hat-mun)/n

    if graph_it:
        t = np.arange(3,N)
        
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        plt.title("Graphon L2 error modulo sigma")
        plt.plot(t,error_equiv_graphon)

        plt.subplot(2, 2, 2)
        plt.title("Signal L2 error modulo sigma")
        plt.plot(t,error_equiv_signal)

        plt.subplot(2, 2, 3)
        plt.title("Error theta_hat-theta squared")
        plt.plot(t,error_prob_matrix)

        plt.subplot(2, 2, 4)
        plt.title("Error mu_hat-mu squared")
        plt.plot(t,error_mean_vector)


        plt.tight_layout()
        plt.show()

