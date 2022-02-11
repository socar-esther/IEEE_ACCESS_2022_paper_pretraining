
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

num_cca_trials = 5

def positivedef_matrix_sqrt(array):
    w, v = np.linalg.eigh(array)  # Return the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or a real symmetric matrix.
    #  A - np.dot(v, np.dot(np.diag(w), v.T))
    wsqrt = np.sqrt(w)  # w is eigenvalues, v is eigenvectors
    sqrtarray = np.dot(v, np.dot(np.diag(wsqrt), np.conj(v).T))  # np.conj(v) return the complex conjugate, element-wise.
    return sqrtarray


def remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon):
    x_diag = np.abs(np.diagonal(sigma_xx))  # np.diagonal get the diagonal value of the matrix
    y_diag = np.abs(np.diagonal(sigma_yy))
    x_idxs = (x_diag >= epsilon)  # epsilon=0
    y_idxs = (y_diag >= epsilon)

    sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
    sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
    sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
    sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]

    return (sigma_xx_crop, sigma_xy_crop, sigma_yx_crop, sigma_yy_crop,x_idxs, y_idxs)


def compute_ccas(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon,
                 verbose=True): # epsilon = 0
    (sigma_xx, sigma_xy, sigma_yx, sigma_yy, x_idxs, y_idxs) = remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, epsilon)
    numx = sigma_xx.shape[0]  # 100
    numy = sigma_yy.shape[0]  # 50

    if numx == 0 or numy == 0:
        return ([0, 0, 0], [0, 0, 0], np.zeros_like(sigma_xx),np.zeros_like(sigma_yy), x_idxs, y_idxs)

    if verbose:   # verbose = True
        print("adding eps to diagonal and taking inverse")
    sigma_xx += epsilon * np.eye(numx)
    sigma_yy += epsilon * np.eye(numy)
    inv_xx = np.linalg.pinv(sigma_xx)  # compute the pseudo-inverse of a matrix
    inv_yy = np.linalg.pinv(sigma_yy)

    if verbose:
        print("taking square root")
    invsqrt_xx = positivedef_matrix_sqrt(inv_xx)
    invsqrt_yy = positivedef_matrix_sqrt(inv_yy)

    if verbose:
        print("dot products...")
    arr = np.dot(invsqrt_xx, np.dot(sigma_xy, invsqrt_yy))

    if verbose:
        print("trying to take final svd")
    u, s, v = np.linalg.svd(arr)

    if verbose:
        print("computed everything!")

    return [u, np.abs(s), v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs


def sum_threshold(array, threshold):
    assert (threshold >= 0) and (threshold <= 1), "print incorrect threshold"

    for i in range(len(array)):
        if np.sum(array[:i])/np.sum(array) >= threshold:
            return i


def create_zero_dict(compute_dirns, dimension):
    return_dict = {}
    return_dict["mean"] = (np.asarray(0), np.asarray(0))
    return_dict["sum"] = (np.asarray(0), np.asarray(0))
    return_dict["cca_coef1"] = np.asarray(0)
    return_dict["cca_coef2"] = np.asarray(0)
    return_dict["idx1"] = 0
    return_dict["idx2"] = 0

    if compute_dirns:
        return_dict["cca_dirns1"] = np.zeros((1, dimension))
        return_dict["cca_dirns2"] = np.zeros((1, dimension))

    return return_dict


def get_cca_similarity(acts1, acts2, epsilon=0., threshold=0.98,
                       compute_coefs=True,
                       compute_dirns=False,
                       verbose=True):
    # assert dimensionality equal
    assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"  # acts1.shape[1] == acts2.shape[1]=1000, 1000 data points
    # check that acts1, acts2 are transposition
    assert acts1.shape[0] < acts1.shape[1], ("input must be number of neurons"
                                           "by datapoints")
    return_dict = {}

    # compute covariance with numpy function for extra stability
    numx = acts1.shape[0]  # numx=100
    numy = acts2.shape[0]  # numx=50

    covariance = np.cov(acts1, acts2)  # [150, 150]
    sigmaxx = covariance[:numx, :numx]  # [0:100, 0:100]
    sigmaxy = covariance[:numx, numx:]  # [0:100, 100:150]
    sigmayx = covariance[numx:, :numx]  # [100:150, 0:100]
    sigmayy = covariance[numx:, numx:]  # [100:150, 100:150]

    # rescale covariance to make cca computation more stable
    xmax = np.max(np.abs(sigmaxx))
    ymax = np.max(np.abs(sigmayy))
    sigmaxx /= xmax
    sigmayy /= ymax
    sigmaxy /= np.sqrt(xmax * ymax)
    sigmayx /= np.sqrt(xmax * ymax)

    ([u, s, v], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs) = compute_ccas(sigmaxx, sigmaxy, sigmayx,
                                                                     sigmayy,
                                                                     epsilon=epsilon,
                                                                     verbose=verbose)

    # if x_idxs or y_idxs is all false, return_dict has zero entries
    if (not np.any(x_idxs)) or (not np.any(y_idxs)):
        return create_zero_dict(compute_dirns, acts1.shape[1])

    if compute_coefs:
    
        # also compute full coefficients over all neurons  # x_idx = [True,...True] len(x_idx)=100,  y_idx = [True,...True] len(y_idx)=50
        x_mask = np.dot(x_idxs.reshape((-1, 1)), x_idxs.reshape((1, -1))) # x_mask: 100x100, all True
        y_mask = np.dot(y_idxs.reshape((-1, 1)), y_idxs.reshape((1, -1))) # y_mask: 50x50, all True

        return_dict["coef_x"] = u.T  # shape: [100, 100]
        return_dict["invsqrt_xx"] = invsqrt_xx
        return_dict["full_coef_x"] = np.zeros((numx, numx))
        np.place(return_dict["full_coef_x"], x_mask,
                 return_dict["coef_x"])
        return_dict["full_invsqrt_xx"] = np.zeros((numx, numx))
        np.place(return_dict["full_invsqrt_xx"], x_mask,
                 return_dict["invsqrt_xx"])

        return_dict["coef_y"] = v
        return_dict["invsqrt_yy"] = invsqrt_yy
        return_dict["full_coef_y"] = np.zeros((numy, numy))
        np.place(return_dict["full_coef_y"], y_mask,
                 return_dict["coef_y"])
        return_dict["full_invsqrt_yy"] = np.zeros((numy, numy))
        np.place(return_dict["full_invsqrt_yy"], y_mask,
                 return_dict["invsqrt_yy"])

        # compute means
        neuron_means1 = np.mean(acts1, axis=1, keepdims=True)
        neuron_means2 = np.mean(acts2, axis=1, keepdims=True)
        return_dict["neuron_means1"] = neuron_means1
        return_dict["neuron_means2"] = neuron_means2

    if compute_dirns: # false here
        # orthonormal directions that are CCA directions
        cca_dirns1 = np.dot(np.dot(return_dict["full_coef_x"],
                                   return_dict["full_invsqrt_xx"]),
                            (acts1 - neuron_means1)) + neuron_means1
        cca_dirns2 = np.dot(np.dot(return_dict["full_coef_y"],
                                   return_dict["full_invsqrt_yy"]),
                            (acts2 - neuron_means2)) + neuron_means2

    # get rid of trailing zeros in the cca coefficients
    idx1 = sum_threshold(s, threshold)
    idx2 = sum_threshold(s, threshold)

    return_dict["cca_coef1"] = s
    return_dict["cca_coef2"] = s
    return_dict["x_idxs"] = x_idxs
    return_dict["y_idxs"] = y_idxs
    # summary statistics
    return_dict["mean"] = (np.mean(s[:idx1]), np.mean(s[:idx2]))
    return_dict["sum"] = (np.sum(s), np.sum(s))

    if compute_dirns:
        return_dict["cca_dirns1"] = cca_dirns1
        return_dict["cca_dirns2"] = cca_dirns2

    return return_dict


def robust_cca_similarity(acts1, acts2, threshold=0.98, epsilon=1e-6, compute_dirns=True):
    for trial in range(num_cca_trials):
        try:
            return_dict = get_cca_similarity(acts1, acts2, threshold, compute_dirns)
        except np.LinAlgError:
            acts1 = acts1*1e-1 + np.random.normal(size=acts1.shape)*epsilon
            acts2 = acts2*1e-1 + np.random.normal(size=acts1.shape)*epsilon
        if trial + 1 == num_cca_trials:
            raise

    return return_dict
