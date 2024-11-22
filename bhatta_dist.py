import numpy as np

def bhattacharyya_distance_multivariate(x, y):
    """
    Compute the Bhattacharyya distance between two multivariate numpy arrays x and y, assuming they come from Gaussian distributions.
    
    :param x: numpy array of shape (n_samples, n_features)
    :param y: numpy array of shape (n_samples, n_features)
    :return: Bhattacharyya distance
    """
    # Compute means
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)
    
    # Compute covariances
    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)
    
    # Compute the mean difference
    mean_diff = mean_x - mean_y
    
    # Compute the average covariance matrix
    cov_avg = (cov_x + cov_y) / 2
    
    # Compute the Bhattacharyya distance
    term1 = 0.125 * np.dot(np.dot(mean_diff.T, np.linalg.inv(cov_avg)), mean_diff)
    term2 = 0.5 * np.log(np.linalg.det(cov_avg) / np.sqrt(np.linalg.det(cov_x) * np.linalg.det(cov_y)))
    
    bd = term1 + term2
    
    return bd

def bhattacharyya_coeff(x, y):
    return np.exp(-bhattacharyya_distance_multivariate(x,y))