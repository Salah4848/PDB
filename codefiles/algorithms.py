import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans



def FANSbased(A,X):
    '''
    Uses FANS algorithm.
    inputs:
        X : 1xn array, obsrved signal data
        A : nxn array, observed adjacensy matrix
    outputs:
        (theta_hat,mu_hat)
    '''
    n=len(X)
    dg = np.zeros_like(A)
    df = np.zeros_like(A)
    #Compute the distances for each ij
    for i in range(n):
        for j in range(i+1):
            temp1 = np.delete(A,[i,j],1)
            temp2 = np.delete(X,[i,j],0)
            dg[i,j] = abs(np.max(np.dot(A[:,i]-A[:,j],temp1)))/n
            df[i,j] = abs(np.max((X[i]-X[j])*temp2))
            dg[j,i] = dg[i,j]
            df[j,i] = df[i,j]

    
    lamb = 1
    c=1
    d = dg + lamb*df
    h = c*np.sqrt(np.log(n)/n) #c*np.sqrt(np.log(n)/n)
    theta_hat = np.zeros_like(A)
    mu_hat = np.zeros_like(X)

    for i in range(n):
        #find neighbors for graphon
        temp = np.delete(d[i],i)
        q = np.quantile(temp,h)
        neighbors = np.where(temp<=q)
        size = len(neighbors[0])
        #estimate mu_i
        mu_hat[i] = np.sum(X[neighbors]) / size
        for j in range(i+1):
            #estimate theta_ij
            theta_hat[i,j] = (np.sum(A[neighbors, j]) / size +
                           np.sum(A[i, neighbors]) / size) / 2
            theta_hat[j,i] =theta_hat[i,j]

    return (theta_hat,mu_hat)


def EMbased(A, X, k, max_iter=100):
    '''
    An EM based method for blockmodels. https://stephens999.github.io/fiveMinuteStats/intro_to_em.html
    inputs:
        A: Adjacency matrix (n x n).
        X: Node signals (n x 1).
        k: Number of blocks.
    outputs:
        theta_hat,mu_hat
    '''
    n = A.shape[0]

    #Use KMeans on X for initial block assignments
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X.reshape(-1, 1))
    z_est = kmeans.labels_

    #Initialize Q and M based on initial assignments
    Q_est = np.zeros((k, k))
    M_est = np.zeros(k)
    
    for a in range(k):
        for b in range(k):
            Q_est[a, b] = np.mean(A[np.ix_(z_est == a, z_est == b)])
        M_est[a] = np.mean(X[z_est == a])


    for iteration in range(max_iter):
        #E-Step: Update block assignments
        z_new = np.zeros(n, dtype=int)
        for i in range(n):
            scores = []
            for a in range(k):
                #Compute likelihood
                signal_likelihood = -0.5 * ((X[i] - M_est[a])**2)
                graphon_likelihood = 0# Find smthing to put here
                scores.append(signal_likelihood + graphon_likelihood)
            z_new[i] = np.argmax(scores)

        #M-Step: Update Q and M
        for a in range(k):
            for b in range(k):
                Q_est[a, b] = np.mean(A[np.ix_(z_new == a, z_new == b)])
            M_est[a] = np.mean(X[z_new == a])

        #Convergence check
        if np.all(z_est == z_new):
            break

        z_est = z_new

    mu_est = M_est[z_est]
    z_mat = np.meshgrid(z_est, z_est)
    theta_est = Q_est[z_mat] 
    return theta_est, mu_est


def align_graphon(theta_hat, true_graphon):
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
    n = len(mu_hat)
    
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