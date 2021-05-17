import numpy as np

#Constructing the array responses for AoA candidates
"""
************** Input
phi_min: Lower-bound of AoAs
phi_max: Upper-bound of AoAs
N: # Antennas
delta_inv: # AoA Candidates 
************** Ouput 
phi: AoA Candidates   
A_BS: Collection of array responses for AoA Candidates
"""
def func_codedesign(delta_inv,phi_min,phi_max,N):
    phi = np.linspace(start=phi_min,stop=phi_max,num=delta_inv)
    from0toN = np.float32(list(range(0, N)))
    A_BS = np.zeros([N,delta_inv],dtype=np.complex64)
    for i in range(delta_inv):
        a_phi = np.exp(1j*np.pi*from0toN*np.sin(phi[i]))
        A_BS[:,i] = np.transpose(a_phi)      
        
    return A_BS, phi    