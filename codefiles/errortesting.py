from .algorithms import *
from .utility import *

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


def plot_arrays(array_pairs):
    """
    Plots a list of (title, 1D array, 2D array) pairs in one figure.
    All the 1D arrays are plotted on the same axis and each 2D array is plotted as a heatmap in separate subplots, arranged horizontally.

    Args:
        array_pairs (list of tuples): A list where each element is a tuple (title, 1D array, 2D array).
    """
    num_pairs = len(array_pairs)

    # Create a figure with 2 rows: 1 for the 1D plot and 1 for the 2D heatmaps arranged horizontally
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1]})
    
    # Plot all 1D arrays on the same axis (first row)
    ax1 = axes[0]  # First row (single plot for all 1D arrays)
    for i, (title, arr_1d, _) in enumerate(array_pairs):
        ax1.plot(arr_1d, label=f'{title}')
    ax1.set_title('Signal plot')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)

    # Plot each 2D array as a heatmap in separate subplots, arranged horizontally (second row)

    for i, (title, _, arr_2d) in enumerate(array_pairs):
        ax = fig.add_subplot(2, num_pairs, num_pairs + i + 1)  # Second row for 2D heatmaps
        cax = ax.imshow(arr_2d, cmap='Greys', interpolation='nearest',vmin=0,vmax=1)
        ax.set_title(f'{title}')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_facecolor('none')
        if i==num_pairs-1:
            fig.colorbar(cax, ax=ax)

    ax=axes.flat[1]
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Adjust layout to avoid overlap, making sure the 1D plot doesn't get stretched
    plt.tight_layout()  
    plt.show()



