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
        
    return A.astype(float), X, theta, mu, vertices

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

def align_graphon(theta_hat, true_graphon, diagonly=False):
    """
    Align the estimated graphon to the true graphon. We use a sorting method : https://stackoverflow.com/questions/54041397/given-two-arrays-find-the-permutations-that-give-closest-distance-between-two-a
    Can feed it k-block versions of the matrices.
    inputs:
    - theta_hat: Estimated probability matrix (kxk).
    - true_graphon: True graphon matrix (kxk).

    outputs:
    - aligned_graphon: Aligned estimated graphon.
    """
    n = theta_hat.shape[0]

    if diagonly:
        #Extract diagonal
        diaghat = np.diagonal(theta_hat)
        diagtrue = np.diagonal(true_graphon)
        #Get the increasing sorting for theta and its inverse for the graphon
        maphat = np.argsort(diaghat)
        inversemaptrue =np.argsort(np.argsort(diagtrue))
        #First sort for increaing diag values
        result = theta_hat[maphat,:]
        result = result[:,maphat]
        #Then apply the inverse sort of the true graphon
        result = result[inversemaptrue,:]
        result = result[:,inversemaptrue]
        return result

    
    #Extract the upper triangular part (excluding diagonal for symmetry)
    triu_indices = np.triu_indices(n, k=0)
    theta_hat_flat = theta_hat[triu_indices]
    true_graphon_flat = true_graphon[triu_indices]
    
    #Sort the flattened arrays
    true_sort_indices = np.argsort(true_graphon_flat)
    estimated_sort_indices = np.argsort(theta_hat_flat)
    
    #Inverse the true sorting
    inverse_permutation = np.argsort(true_sort_indices)
    
    #Apply the inverse permutation to the sorted estimate
    aligned_flat = np.zeros_like(theta_hat_flat)
    aligned_flat = theta_hat_flat[estimated_sort_indices]
    aligned_flat = aligned_flat[inverse_permutation]
    
    #Reconstruct the aligned graphon matrix
    aligned_graphon = np.zeros_like(theta_hat)
    aligned_graphon[triu_indices] = aligned_flat
    temp = np.copy(aligned_graphon)
    np.fill_diagonal(temp,0)
    aligned_graphon = temp + aligned_graphon.T  #Symmetrize
    
    return aligned_graphon

def align_signal(mu_hat, true_signal):
    '''
    Aligns the estimated signal to the true signal. Same procedure as for the graphon.
    Can feed it k-block versions
    inputs:
        mu_hat: estimated mean vector (1xk)
        true_signal: true mean vector (1xk)
    '''
    
    #Sort arrays
    true_sort_indices = np.argsort(true_signal)
    estimated_sort_indices = np.argsort(mu_hat)
    
    #Inverse the true sorting
    inverse_permutation = np.argsort(true_sort_indices)
    
    #Apply the inverse permutation to the sorted estimate
    aligned = np.zeros_like(mu_hat)
    aligned = mu_hat[estimated_sort_indices]
    aligned = aligned[inverse_permutation]

    
    return aligned

def random_step_signal(k, value_range=(-10,10), threshold_range=(0, 1),sort=False):
    '''
    Generates random step signal
    '''

    thresholds = np.sort(np.random.uniform(low=threshold_range[0], high=threshold_range[1], size=k-1))
    
    
    values = np.random.uniform(low=value_range[0], high=value_range[1], size=k)

    if sort:
        values = np.sort(values)
    
    signal =make_step_signal(thresholds,values)
    
    return signal, thresholds, values

def random_step_graphon(k, value_range=(0, 1), threshold_range=(0, 1),sort=False):
    """
    Generate a random step graphon.
    
    """
    thresholds = np.sort(np.random.uniform(low=threshold_range[0], high=threshold_range[1], size=k-1))
    
    values = np.random.uniform(low=value_range[0], high=value_range[1], size=(k, k))
    values = (values + values.T) / 2
    if sort:
        diag = np.diagonal(values)
        perm = np.argsort(diag)
        values = values[perm,:]
        values = values[:,perm]
    
    graphon = make_step_graphon(thresholds,values)
    
    return graphon, thresholds, values

def random_step_graphon_signal(k, graphon_range=(0, 1),signal_range=(-10,10), threshold_range=(0, 1),aligned=False):
    """
    Generate a random step graphon-signal.
    
    """
    dirichlet_probs = np.random.dirichlet([20] * (k))
    thresholds = np.cumsum(dirichlet_probs)[:-1]  # Cumulative sum ensures sorted values
    thresholds = threshold_range[0] + thresholds * (threshold_range[1] - threshold_range[0])  # Scale to range
    
    gvalues = np.random.uniform(low=graphon_range[0], high=graphon_range[1], size=(k, k))
    gvalues = (gvalues + gvalues.T) / 2
    
    svalues = np.sort(np.random.uniform(low=signal_range[0], high=signal_range[1], size=k))

    if aligned:
        diag = np.diagonal(gvalues)
        perm = np.argsort(diag)
        gvalues = gvalues[perm,:]
        gvalues = gvalues[:,perm]

        svalues= np.sort(svalues)


    graphon = make_step_graphon(thresholds,gvalues)
    signal = make_step_signal(thresholds,svalues)
    
    return graphon, signal

def make_step_signal(thresholds, values):
    def signal(x):

        x = np.asarray(x)
        result = np.empty_like(x, dtype=np.float64)
        
        x_regions = np.digitize(x, thresholds, right=False)
        result = values[x_regions]

        
        return result
    return signal

def make_step_graphon(thresholds,values):
    def graphon(x, y):

        x = np.asarray(x)
        y = np.asarray(y)
        
        x_regions = np.digitize(x, thresholds, right=False)
        y_regions = np.digitize(y, thresholds, right=False)
        
        result = values[x_regions, y_regions]
        return result
    return graphon