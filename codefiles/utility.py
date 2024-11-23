import matplotlib.pyplot as plt
import numpy as np 

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
    est_signal=np.append(est_signal,[0])

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.title("Estimated Graphon (Theta Hat)")
    plt.imshow(est_graphon, cmap='viridis')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("True Graphon")
    plt.imshow(w, cmap='viridis')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("Estimated Signal")
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
        (A,X)
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
        
    return (A.astype(int).astype(float),X)

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