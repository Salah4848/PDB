from .algorithms import *
from .utility import *

def benchmark_error(A, X, theta, mu, methods, graph_it=True):
    '''
    Graphs the error(s) against the amount of samples used. Used to check rates. min samplesize is 20
    '''
    
    N = len(X)

    minn = 50
    intervals = N//10

    error_prob_matrix = []
    error_mean_vector = []
    error_joint=[]
    names = []
    for n in range(minn,N,intervals):
        Xn = X[:n]
        An = A[:n,:n]
        thetan = theta[:n,:n]
        mun = mu[:n]

        errsprob = []
        errsmean = []
        errsjoint = []
        for method in methods:
            theta_hat, mu_hat, name = method(An, Xn)
            if n==minn:
                names.append(name)
            errth = squared_norm_matrix(theta_hat-thetan)/(n*n)
            errmu = squared_norm_vector(mu_hat-mun)/n
            errsprob.append(errth)
            errsmean.append(errmu)
            errsjoint.append(errth+errmu)

        error_prob_matrix.append(errsprob)
        error_mean_vector.append(errsmean)
        error_joint.append(errsjoint)

    if graph_it:
        error_prob_matrix = np.array(error_prob_matrix)
        error_mean_vector = np.array(error_mean_vector)
        error_joint = np.array(error_joint)
        print("THETA ERROR \n",error_prob_matrix)
        print("MU ERROR \n",error_mean_vector)
        print("JOINT ERROR \n",error_joint)
        
        t = np.arange(minn,N,intervals)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1]})

        ax1 = axes[0]

        for i in range(len(methods)):
            ax1.plot(t,error_joint[:,i], label = names[i])
        ax1.legend()
        ax1.set_title('Joint error')

        ax2 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2,2,4)

        for i in range(len(methods)):
            ax2.plot(t,error_prob_matrix[:,i], label = names[i])
            ax3.plot(t,error_mean_vector[:,i], label = names[i])
        ax2.legend()
        ax2.set_title('Graphon error')
        ax3.legend()
        ax3.set_title('Signal error')

        ax=axes.flat[1]
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

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
    ax1.set_ylabel('Value')
    ax1.set_xticks([])
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



