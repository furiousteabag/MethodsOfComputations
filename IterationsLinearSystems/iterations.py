import numpy as np
from numpy.linalg import solve, inv, norm

A = np.array([[7.35272,  0.88255,  -2.270052],
              [0.88255,  5.58351,  0.528167],
              [-2.27005, 0.528167, 4.430329]])

b = np.array([[1.],
              [0.],
              [0.]])

def residual(A, b, x):
    """Calculating solution error.

    Returns:
        res_ver (ndarray): b - Ax
    """
    res_vec = b - np.dot(A, x)
    return res_vec

def gaussian_elimination(A, b, eps=1e-5):
    """Solving linear system using gaussian elimination.

    Args:
        A (ndarray<ndarray, ndarray>): matrix of coefficents.
        b (ndarray): vector of values.
        eps (float): all values below eps equivalent to zero.

    Returns:
        x (ndarray): solution.
    """

    # Getting matrix shape.
    n = A.shape[0]

    # Merging coefficents with values.
    Ab = np.concatenate((A, b), axis=1)

    # Making upper triangular matrix.
    for k in range(0, n):

        # Dividing all row alements after
        # diagonal element on diagonal
        # element.
        tmp = Ab[k][k]
        if np.abs(tmp) < eps:
            print("\nElement Ab[{}][{}]={} smaller than eps={}.".format(
                k, k, tmp, eps))
        for j in range(k, n + 1):
            Ab[k][j] = Ab[k][j] / tmp

        # Substracting top element multiplied
        # by 1st element in row from each
        # element.
        for i in range(k + 1, n):
            tmp = Ab[i][k]
            for j in range(k, n + 1):
                Ab[i][j] = Ab[i][j] - Ab[k][j] * tmp

    # Solve equation for an upper
    # triangular matrix Ab.
    x = np.zeros((3, 1))
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i][n] / Ab[i][i]
        for k in range(i - 1, -1, -1):
            Ab[k][n] -= Ab[k][i] * x[i]

    # Calculating error.
    res_vec = residual(A, b, x)

    return x


def iterations_prepare(A, b, verbose=True):
    """prepares linear system to x = h_d * x + g_d form.

    Args:
        A (ndarray<ndarray, ndarray>): matrix of coefficents.
        b (ndarray): vector of values.

    D is diagonal matrix with coefficents from A diagonal. 

    Returns:
        H_D (ndarray<ndarray, ndarray>): E - D^(-1) * A.
        g_D (ndarray): D^(-1) * b.
    """

    verboseprint = print if verbose else lambda *a, **k: None

    verboseprint("\nA: \n{}".format(A))
    verboseprint("\nb: \n{}".format(b))

    D = np.diag(np.diag(A))
    verboseprint("\nD: \n{}".format(D))

    H_D = np.identity(A.shape[0]) - np.dot(inv(D), A)
    verboseprint("\nH_D: \n{}".format(H_D))

    g_D = np.dot(inv(D), b)
    verboseprint("\ng_D: \n{}".format(g_D))

    H_D_norm = norm(H_D, np.inf)
    verboseprint("\n||H_D||_inf: {}".format(H_D_norm))

    return H_D, g_D

def apriori_estimation(H, g, verbose=True, eps=1e-5, x_0=[[0.],
                                                          [0.],
                                                          [0.]]):
    """Apriori estimating number of iterations.

    Find such k, that difference between
    answer and solution will be less then eps.

    Args:
        H (ndarray<ndarray, ndarray>): E - D^(-1) * A.
        g (ndarray): D^(-1) * b.
        eps (float): solution error rate.
        x_0 (ndarray<ndarray, ndarray>): starting vector.

    Returns:
        k (int): estimated number of iterations.
    """

    verboseprint = print if verbose else lambda *a, **k: None

    k = 0
    while True:
        error = (norm(H, np.inf)**k) * norm(x_0, np.inf) + \
                (norm(H, np.inf)**k) * norm(g, np.inf) / \
                (1 - norm(H, np.inf))
        if (error < eps):
            break
        k += 1

    verboseprint("\nEstimated number of iterations: {}".format(k))

    return k


def iterations_method(A, b, k, verbose=True, x_0=[[0.],
                                                  [0.],
                                                  [0.]]):
    """Solving linear system using iterations method.

    Args:
        A (ndarray<ndarray, ndarray>): matrix of coefficents.
        b (ndarray): vector of values.
        k (int): number of iterations.
        x_0 (ndarray<ndarray, ndarray>): starting vector.

    Returns:
        x (ndarray): solution.
    """

    verboseprint = print if verbose else lambda *a, **k: None

    H, g = iterations_prepare(A, b, verbose=verbose)

    for i in range(k):
        x = np.dot(H, x_0) + g
        x_0 = x

    verboseprint("\nIterations solution: \n{}".format(x))

    return x



# H, g = iterations_prepare(A, b)
# k = apriori_estimation(H, g)

gauss_solution = gaussian_elimination(A, b) 
print("\nGauss solution: \n{}".format(gauss_solution))

iterations_solution = iterations_method(A, b, 30) 
# print("\nIterations solution: \n{}".format(iterations_solution))
