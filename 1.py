import numpy as np

def svd(A):
    AT_A = A.T @ A
    eigenvalues, eigvector = np.linalg.eigh(AT_A)

    sorted_eigval  = np.argsort(eigenvalues)[::-1]
    sigma = np.sqrt(np.abs(eigenvalues[sorted_eigval]))

    eigvector = eigvector[:, sorted_eigval]

    non_zero = sigma > 0
    sigma_positive = sigma[non_zero]
    eigvector_positive = eigvector[:, non_zero]

    U = (A @ eigvector_positive) / sigma_positive

    S = np.diag(sigma_positive)
    Vt = eigvector_positive.T

    return U, S, Vt

A = np.array([[0, 1, 1],
              [1, 2, 2]])

U, S, VT = svd(A)

print("U:\n", U)
print("\nS:\n", S)
print("\nV^T:\n", VT)

print("\nПеревірка з початковою матрицею")
print(np.allclose(A, U @ S @ VT))
