import numpy as np

def estimate_graphon_signal(X,A):
    '''
    Returns estimates of the probabiliti matrix theta and the mean vector mu
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
            df[j,i] = dg[i,j]
        
    d = dg + df #should add a lambda param here maybe
    h = int(np.ceil(np.sqrt(np.log(n) / n)))  #neighborhood size parameter
    theta_hat = np.zeros_like(A)
    mu_hat = np.zeros_like(X)

    for i in range(n):
        #find neighbors
        neighbors = np.argsort(d[i])[:h]

        #estimate mu_i
        mu_hat[i] = np.sum(X[neighbors]) / len(neighbors)
        for j in range(n):
            #estimate theta_ij
            theta_hat[i, j] = (np.sum(A[neighbors, j]) / len(neighbors) +
                           np.sum(A[i, neighbors]) / len(neighbors)) / 2

    return (theta_hat,mu_hat)

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
    triu_indices = np.triu_indices(n, k=1)
    theta_hat_flat = theta_hat[triu_indices]
    true_graphon_flat = true_graphon[triu_indices]
    
    #Sort the flattened arrays
    true_sort_indices = np.argsort(true_graphon_flat)
    estimated_sort_indices = np.argsort(theta_hat_flat)
    
    #Inverse the true sorting
    inverse_permutation = np.argsort(true_sort_indices)
    
    #Apply the inverse permutation to the sorted estimate
    aligned_flat = np.zeros_like(theta_hat_flat)
    aligned_flat[inverse_permutation] = theta_hat_flat[estimated_sort_indices]
    
    #Reconstruct the aligned graphon matrix
    aligned_graphon = np.zeros_like(theta_hat)
    aligned_graphon[triu_indices] = aligned_flat
    aligned_graphon = aligned_graphon + aligned_graphon.T  #Symmetrize
    
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
    aligned[inverse_permutation] = mu_hat[estimated_sort_indices]

    
    return aligned

