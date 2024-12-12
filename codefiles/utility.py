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

def visualize(graphon, signal, est_graphon, est_signal, n_points=200):
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

    plt.close()
    plt.figure()

    plt.subplot(223)
    plt.title("Estimated Graphon (theta Hat)")
    plt.imshow(est_graphon, cmap='viridis',vmin=0,vmax=1)
    plt.colorbar()

    plt.subplot(224)
    plt.title("True Graphon")
    plt.imshow(w, cmap='viridis',vmin=0,vmax=1)
    plt.colorbar()

    plt.subplot(211)
    plt.title("Estimated Signal (mu_hat)")
    plt.plot(tstep, est_signal, drawstyle='steps-post',label="estimate")
    plt.plot(t,f, label="True")
    plt.legend()

    


    plt.tight_layout()
    plt.show()

def sample_from_graphon_signal(w, f, n, symmetric=True, self_loops=False):
    '''
    Samples an nxn adjacency matrix from the graphon W and signal M from signal f with added standard gaussian noise.
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

def benchmark_error(A, X, theta, mu, graphon, signal,method, graph_it=True):
    '''
    Graphs the error(s) against the amount of samples used. Used to check rates. min samplesize is 20
    '''
    minn = 20
    N = len(X)
    error_equiv_graphon = np.zeros(N-minn)
    error_equiv_signal = np.zeros(N-minn)
    error_prob_matrix = np.zeros(N-minn)
    error_mean_vector = np.zeros(N-minn)

    for n in range(minn,N):
        Xn = X[:n]
        An = A[:n,:n]
        thetan = theta[:n,:n]
        mun = mu[:n]

        theta_hat, mu_hat = method(An, Xn)

        w_matrix = blockify_graphon(graphon, n)
        f_matrix = blockify_signal(signal, n)
        aligned_theta_hat = align_graphon(theta_hat, w_matrix)
        aligned_mu_hat = align_signal(mu_hat, f_matrix)

        error_equiv_graphon[n-minn] = squared_norm_matrix(aligned_theta_hat-w_matrix)/(n*n)
        error_equiv_signal[n-minn] = squared_norm_vector(aligned_mu_hat-f_matrix)/n
        error_prob_matrix[n-minn] = squared_norm_matrix(theta_hat-thetan)/(n*n)
        error_mean_vector[n-minn] = squared_norm_vector(mu_hat-mun)/n

    if graph_it:
        t = np.arange(minn,N)
        
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


def random_step_signal(k, value_range=(-10,10), threshold_range=(0, 1)):
    '''
    Generates random step signal
    '''

    thresholds = np.sort(np.random.uniform(low=threshold_range[0], high=threshold_range[1], size=k-1))
    
    values = np.sort(np.random.uniform(low=value_range[0], high=value_range[1], size=k))
    
    def signal(x):

        x = np.asarray(x)
        result = np.empty_like(x, dtype=np.float64)
        
        x_regions = np.digitize(x, thresholds, right=False)
        result = values[x_regions]

        
        return result
    
    return signal, thresholds, values

def random_step_graphon(k, value_range=(0, 1), threshold_range=(0, 1)):
    """
    Generate a random step graphon.
    
    """
    thresholds = np.sort(np.random.uniform(low=threshold_range[0], high=threshold_range[1], size=k-1))
    
    values = np.random.uniform(low=value_range[0], high=value_range[1], size=(k, k))
    values = (values + values.T) / 2
    
    def graphon(x, y):

        x = np.asarray(x)
        y = np.asarray(y)
        
        x_regions = np.digitize(x, thresholds, right=False)
        y_regions = np.digitize(y, thresholds, right=False)
        
        result = values[x_regions, y_regions]
        return result
    
    return graphon, thresholds, values

def random_step_graphon_signal(k, graphon_range=(0, 1),signal_range=(-10,10), threshold_range=(0, 1)):
    """
    Generate a random step graphon-signal.
    
    """
    thresholds = np.sort(np.random.uniform(low=threshold_range[0], high=threshold_range[1], size=k-1))
    
    gvalues = np.random.uniform(low=graphon_range[0], high=graphon_range[1], size=(k, k))
    gvalues = (gvalues + gvalues.T) / 2
    
    svalues = np.sort(np.random.uniform(low=signal_range[0], high=signal_range[1], size=k))

    def graphon(x, y):

        x = np.asarray(x)
        y = np.asarray(y)
        
        x_regions = np.digitize(x, thresholds, right=False)
        y_regions = np.digitize(y, thresholds, right=False)
        
        result = gvalues[x_regions, y_regions]
        return result
    
    def signal(x):

        x = np.asarray(x)
        result = np.empty_like(x, dtype=np.float64)
        
        x_regions = np.digitize(x, thresholds, right=False)
        result = svalues[x_regions]

        
        return result
    
    return graphon, signal