"""
Utility functions for processing 6x6 elastic stiffness tensors.
Includes Voigt-Reuss-Hill averaging and Born stability criteria.
"""

import numpy as np

def calculate_moduli(C, eps=0.01):
    """
    Calculate the Voigt, Reuss, and Hill approximations for Bulk (K) and Shear (G) modulus.
    Enforces tensor symmetry, Born stability positive-semi-definiteness via eigenvalue regularization,
    and physical Voigt-Reuss bounds inequality (0 <= K_R <= K_V and 0 <= G_R <= G_V).
    
    Args:
        C (list or np.array): 6x6 elastic stiffness matrix in Voigt notation (GPa)
        eps (float): Small positive threshold for eigenvalue regularization (default 0.01 GPa)
        
    Returns:
        dict: containing K_V, K_R, K_H, G_V, G_R, G_H, is_born_stable, min_eigenvalue
    """
    C = np.array(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("Stiffness matrix must be 6x6")
        
    # 1. Enforce symmetry C_ij = C_ji
    C_sym = (C + C.T) / 2.0
    
    # 2. Check Born mechanical stability (positive definiteness)
    eigvals, eigvecs = np.linalg.eigh(C_sym)
    min_eig = float(eigvals[0])
    is_born_stable = bool(np.all(eigvals > 1e-4))
    
    # 3. Voigt averages (upper bounds, linear combination of C_ij)
    K_V = (C_sym[0, 0] + C_sym[1, 1] + C_sym[2, 2] + 2.0 * (C_sym[0, 1] + C_sym[1, 2] + C_sym[2, 0])) / 9.0
    G_V = (C_sym[0, 0] + C_sym[1, 1] + C_sym[2, 2] - C_sym[0, 1] - C_sym[1, 2] - C_sym[2, 0] + 3.0 * (C_sym[3, 3] + C_sym[4, 4] + C_sym[5, 5])) / 15.0
    
    # 4. Reuss averages via regularized compliance matrix S = C_reg^-1
    eigvals_reg = np.maximum(eigvals, eps)
    C_reg = eigvecs @ np.diag(eigvals_reg) @ eigvecs.T
    
    try:
        S = np.linalg.inv(C_reg)
        denom_K = S[0, 0] + S[1, 1] + S[2, 2] + 2.0 * (S[0, 1] + S[1, 2] + S[2, 0])
        K_R = 1.0 / denom_K if denom_K > 1e-6 else K_V
        
        denom_G = 4.0 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4.0 * (S[0, 1] + S[1, 2] + S[2, 0]) + 3.0 * (S[3, 3] + S[4, 4] + S[5, 5])
        G_R = 15.0 / denom_G if denom_G > 1e-6 else G_V
    except np.linalg.LinAlgError:
        K_R = K_V
        G_R = G_V
        
    # Enforce fundamental physical Voigt-Reuss ordering: 0 <= K_R <= K_V and 0 <= G_R <= G_V
    K_V = max(0.0, float(K_V))
    G_V = max(0.0, float(G_V))
    K_R = max(0.0, min(float(K_R), K_V))
    G_R = max(0.0, min(float(G_R), G_V))
    
    # Hill averages
    K_H = (K_V + K_R) / 2.0
    G_H = (G_V + G_R) / 2.0
    
    return {
        'K_V': K_V, 'K_R': K_R, 'K_H': K_H,
        'G_V': G_V, 'G_R': G_R, 'G_H': G_H,
        'is_born_stable': is_born_stable,
        'min_eigenvalue': min_eig
    }

def check_stability(C):
    """
    Check if the crystal is mechanically stable based on the Born criteria.
    A crystal is stable if all eigenvalues of its stiffness matrix are positive.
    
    Args:
        C (list or np.array): 6x6 elastic stiffness matrix
        
    Returns:
        bool: True if stable, False otherwise
    """
    C = np.array(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("Stiffness matrix must be 6x6")
        
    C_sym = (C + C.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(C_sym)
    return bool(np.all(eigenvalues > 1e-4))

if __name__ == '__main__':
    # Test with Silicon
    C_Si = [
        [165.7, 63.9, 63.9, 0, 0, 0],
        [63.9, 165.7, 63.9, 0, 0, 0],
        [63.9, 63.9, 165.7, 0, 0, 0],
        [0, 0, 0, 79.6, 0, 0],
        [0, 0, 0, 0, 79.6, 0],
        [0, 0, 0, 0, 0, 79.6]
    ]
    
    moduli = calculate_moduli(C_Si)
    is_stable = check_stability(C_Si)
    
    print("Test Tensor (Silicon):")
    print(f"Bulk Modulus (Hill):  {moduli['K_H']:.2f} GPa")
    print(f"Shear Modulus (Hill): {moduli['G_H']:.2f} GPa")
    print(f"Is Stable? {is_stable}")
