import numpy as np

def svd(A):
    A = np.array(A, dtype=float)
    m, n = A.shape

    ATA = A.T @ A
    eigvalues, eigvectors = np.linalg.eigh(ATA)

    idx = np.argsort(eigvalues)[::-1]
    eigvalues = eigvalues[idx]
    eigvectors = eigvectors[:, idx]

    sing_vals = np.sqrt(np.clip(eigvalues, 0, None))
    sing_vals = np.round(sing_vals, 12)

    AAT = A @ A.T
    eigvals_U, U = np.linalg.eigh(AAT)
    idxU = np.argsort(eigvals_U)[::-1]
    U = U[:, idxU]

    Sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        Sigma[i, i] = sing_vals[i]

    Vt = eigvectors.T

    return U, Sigma, Vt


A = np.array([[1, -1, 1],
              [-2,  2, -2]], dtype=float)

U, S, Vt = svd(A)
A_rec = U @ S @ Vt

np.set_printoptions(precision=6, suppress=True)

print("U:")
print(U)
print("\nSigma:")
print(S)
print("\nVt:")
print(Vt)
print("\nReconstructed A:")
print(A_rec)
