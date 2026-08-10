"""
Utility functions for processing 6x6 elastic stiffness tensors.
Includes Voigt-Reuss-Hill averaging and Born stability criteria.
"""

import numpy as np

def calculate_moduli(C):
    """
    Calculate the Voigt, Reuss, and Hill approximations for Bulk (K) and Shear (G) modulus.
    
    Args:
        C (list or np.array): 6x6 elastic stiffness matrix in Voigt notation (GPa)
        
    Returns:
        dict: containing K_V, K_R, K_H, G_V, G_R, G_H
    """
    C = np.array(C, dtype=float)
    if C.shape != (6, 6):
        raise ValueError("Stiffness matrix must be 6x6")
        
    # Invert C to get compliance matrix S
    try:
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        # If matrix is singular, return NaNs
        return {k: float('nan') for k in ['K_V', 'K_R', 'K_H', 'G_V', 'G_R', 'G_H']}
    
    # Voigt averages
    K_V = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[1, 2] + C[2, 0])) / 9.0
    G_V = (C[0, 0] + C[1, 1] + C[2, 2] - C[0, 1] - C[1, 2] - C[2, 0] + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0
    
    # Reuss averages
    try:
        K_R = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[1, 2] + S[2, 0]))
    except ZeroDivisionError:
        K_R = float('nan')
        
    try:
        G_R = 15.0 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[1, 2] + S[2, 0]) + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    except ZeroDivisionError:
        G_R = float('nan')
        
    # Hill averages
    K_H = (K_V + K_R) / 2.0
    G_H = (G_V + G_R) / 2.0
    
    return {
        'K_V': K_V, 'K_R': K_R, 'K_H': K_H,
        'G_V': G_V, 'G_R': G_R, 'G_H': G_H
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
        
    # Ensure matrix is symmetric (it should be physically)
    C_sym = (C + C.T) / 2.0
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(C_sym)
    
    # Check if all eigenvalues are strictly positive
    # Using a small threshold to account for numerical precision issues
    return np.all(eigenvalues > 1e-5)

if __name__ == '__main__':
    # Test with a known tensor (approximate values for Silicon)
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
