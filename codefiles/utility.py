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
        signal_matrix: nx1 1D array
    '''
    x = np.linspace(0, 1, n)
    signal_matrix = signal(x)
    return signal_matrix

def squared_norm_matrix(M):
    return np.sum(np.square(M))

def squared_norm_vector(v):
    return np.dot(v,v)

def align_graphon(theta_hat, true_graphon):
    """
    Align the estimated graphon to the true graphon. Uses node degrees for approximate optimal alignment when the marignal varies enough.
    inputs:
    - theta_hat: Estimated probability matrix (nxn).
    - true_graphon: True graphon matrix (nxn).

    outputs:
    - aligned_graphon: Aligned estimated graphon.
    """
    n = theta_hat.shape[0]

    #get the node degress
    degrees_hat = np.sum(theta_hat, axis = 0)
    degrees_true = np.sum(true_graphon, axis =0)

    #Get the permutations
    perm_hat = np.argsort(degrees_hat)
    perm_true = np.argsort(degrees_true)

    #inverse the true sorting
    inverse_perm_true = np.argsort(perm_true)

    #apply permutations
    theta_hat = theta_hat[perm_hat,:]
    theta_hat = theta_hat[:,perm_hat]
    theta_hat = theta_hat[inverse_perm_true,:]
    theta_hat = theta_hat[:,inverse_perm_true]
    
    return theta_hat

def align_signal(mu_hat, true_signal):
    '''
    Aligns the estimated signal to the true signal. It uses sorting, which provides an exact optimal alignment.
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
    mu_hat = mu_hat[estimated_sort_indices]
    mu_hat = mu_hat[inverse_permutation]

    
    return mu_hat

def random_step_signal(k, value_range=(-10,10), threshold_range=(0, 1),sort=False):
    '''
    Generates random step signal
    '''

    thresholds = np.sort(np.random.uniform(low=threshold_range[0], high=threshold_range[1], size=k-1))
    
    
    values = np.random.uniform(low=value_range[0], high=value_range[1], size=k)

    if sort:
        values = np.sort(values)
    
    signal =make_step_signal(thresholds,values)
    
    return signal

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
    
    return graphon

def random_step_graphon_signal(k, graphon_range=(0, 1),signal_range=(-5,5), threshold_range=(0, 1),aligned=False):
    """
    Generate a random step graphon-signal.
    
    """
    dirichlet_probs = np.random.dirichlet([10] * (k))
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

def mat_vect_error(M1,v1,M2,v2):
    '''
    Matrices must be shape nxn and vectors must be length n
    '''
    n=M1.shape[0]

    error = squared_norm_vector(v1-v2)/n + squared_norm_matrix(M1-M2)/(n*n)
    return error

def align_graphon_signal(theta_hat,mu_hat, true_graphon, true_signal, vertices=None, usegraphon=False,uselatents=False):
    '''
    This function aligns our estimates to the ground truth.
    outputs:
    (theta_aligned,mu_aligned)
    '''
    n= theta_hat.shape[0]
    def degreemethod(mat,vect,tmat,tvect):

        #get the node degress
        degrees_hat = np.sum(mat, axis = 0)
        degrees_true = np.sum(tmat, axis =0)

        #Get the permutations
        perm_hat = np.argsort(degrees_hat)
        perm_true = np.argsort(degrees_true)

        #inverse the true sorting
        inverse_perm_true = np.argsort(perm_true)

        #apply permutations
        mat = mat[perm_hat,:]
        mat = mat[:,perm_hat]
        mat = mat[inverse_perm_true,:]
        mat = mat[:,inverse_perm_true]
        vect = vect[perm_hat]
        vect = vect[inverse_perm_true]
        return mat,vect
    
    def sigmethod(mat,vect,tmat,tvect):
        if vertices is None: return mat, vect
        perm = np.argsort(tvect)
        perm_hat = np.argsort(vect)

        inverse_perm = np.argsort(perm)

        
        mat = mat[perm_hat,:]
        mat = mat[:,perm_hat]
        mat = mat[inverse_perm,:]
        mat = mat[:,inverse_perm]
        vect = vect[perm_hat]
        vect = vect[inverse_perm]
        return mat, vect
    
    def diagmethod(mat,vect,tmat,tvect):

        perm = np.argsort(np.diagonal(tmat))
        perm_hat = np.argsort(np.diagonal(mat))

        inverse_perm = np.argsort(perm)

        mat = mat[perm_hat,:]
        mat = mat[:,perm_hat]
        mat = mat[inverse_perm,:]
        mat = mat[:,inverse_perm]
        vect = vect[perm_hat]
        vect = vect[inverse_perm]
        return mat, vect
    
    def latentmethod(mat,vect,tmat,tvect):

        perm = np.argsort(vertices)

        mat = mat[perm,:]
        mat = mat[:,perm]
        vect = vect[perm]
        return mat, vect



    if uselatents : return latentmethod(theta_hat,mu_hat,true_graphon,true_signal)

    methods = [degreemethod,sigmethod,diagmethod,latentmethod]
    aligned = []
    prev = np.inf
    index = 0
    best = 0
    for method in methods:
        aligned_mat,aligned_vect = method(theta_hat,mu_hat,true_graphon,true_signal)
        aligned.append((aligned_mat,aligned_vect))
        if usegraphon:
            error = squared_norm_matrix(aligned_mat-true_graphon)/(n**2)
        else:
            error = mat_vect_error(aligned_mat,aligned_vect,true_graphon,true_signal)
        if error<prev:
            best = index
            prev = error
        index+=1
    print(best)
    return aligned[best]

def make_diff_signal(graphon_func, initconstant = 1, diffusion_num=500, sequence=lambda n: 10*n,precision=300):
    '''
    Makes a diffused signal fucntion form the input graphon:
    inputs:
        graphon_funct: must be the graphon function
        diffusion_num: amount of times to run a diffusion
        sequence: the sequence which scales each diffusion
    outputs:
        signal_func
    '''
    g_mat = blockify_graphon(graphon_func,precision)

    n = g_mat.shape[0]
    f0 = initconstant*np.ones(n)
    f =  sequence(0)*f0 # Initialize with the term for l = 0
    
    S_power = np.eye(n)  # Initialize A^0 as the identity matrix
    for l in range(1, diffusion_num + 1):
        S_power = np.dot(S_power, g_mat)/precision
        f += sequence(l) * np.dot(S_power, f0)

    thresholds = np.linspace(0,1,100)
    signal_func = make_step_signal(thresholds[:-1],f)

    return signal_func

def make_dist_graphon(signal_func, g = None,dist_metric=lambda a,b: abs(a-b), inversprop=True):
    '''
    Makes graphon from a signal using a distance/inverse distance relation
    '''
    f_vect = blockify_signal(signal_func,100)
    
    if g is None: g = lambda x: np.cos(x)*np.cos(x)

    f_1,f_2 = np.meshgrid(f_vect,f_vect)

    w_mat = g(dist_metric(f_1,f_2))
    w_mat = w_mat/np.max(w_mat)

    thresholds = np.linspace(0,1,100)
    graphon_func = make_step_graphon(thresholds[:-1], w_mat)

    return graphon_func