#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import scipy.sparse as sp
import pybobyqa
import time
from functools import reduce


# In[2]:


#Sparse Statevector Simulator
class SparseState:
    def __init__(self, n):
        self.n = n
        self.state = sp.eye(2**self.n, 1, 0, complex, "csr")
    #This is saved as a Scipy sparse array, use print(self.state.toarray()) to see statevector explicitly,
    #Same applies to gate operations

    #Apply single qubit transformation T to qubit i
    def op(self, T, i):
    #I_{2^i}
        I_L = sp.identity(2**i, complex, "csr")

    #I_{2^{n-i-1}} 
        I_R = sp.identity(2**(self.n - i - 1), complex, "csr")

    #I_L ⊗ T ⊗ I_R
        T_full = sp.kron(I_L, sp.kron(T, I_R, "csr"), "csr")

    #Apply transformation to state (multiplication)
        self.state = T_full.dot(self.state)
               
    #RX Gate
    def rx(self, theta, i):
        rx_matrix = sp.csr_matrix(np.array([[np.cos(0.5*theta), -1j*np.sin(0.5*theta)],[-1j*np.sin(0.5*theta), np.cos(0.5*theta)]]))
        self.op(rx_matrix, i)
        
    #RY Gate
    def ry(self, theta, i):
        ry_matrix = sp.csr_matrix(np.array([[np.cos(0.5*theta), -np.sin(0.5*theta)],[np.sin(0.5*theta), np.cos(0.5*theta)]]))
        self.op(ry_matrix, i)
        
    #RZ Gate
    def rz(self, theta, i):
        rz_matrix = sp.csr_matrix(np.array([[np.exp(-0.5*1j*theta), 0],[0, np.exp(0.5*1j*theta)]]))
        self.op(rz_matrix, i)

    #Hadamard Gate
    def hadamard(self, i):
        h_matrix = sp.csr_matrix((1.0/(2.0**0.5))*np.array([[1,1],[1,-1]], complex))    
        self.op(h_matrix, i)
        
    #CNOT Gate
    def CNOT(self, i, j):
    #Projectors onto 0 and 1 states on qubit i
        P_0 = sp.csr_matrix(np.array([[1,0],[0,0]], complex))
        P_1 = sp.csr_matrix(np.array([[0,0],[0,1]], complex))
    #Identity and Pauli X on qubits i, j
        I_I = sp.identity(2, complex, "csr")
        I_J = sp.identity(2, complex, "csr")
        X = sp.csr_matrix(np.array([[0,1],[1,0]], complex))
        if i<j:
        #I_{2^i}
           I_L = sp.identity(2**i, complex, "csr")
    
        #I_{2^(j-i-1)}
           I_M = sp.identity(2**(j-i-1), complex, "csr")
     
        #I_{2^{n-j-1}}
           I_R = sp.identity(2**(self.n-j-1), complex, "csr")
        
        #I_L ⊗ P_0 ⊗ I_M ⊗ I_J ⊗ I_R + I_L ⊗ P_1 ⊗ I_M ⊗ X ⊗ I_R
           CNOT_matrix = sp.kron(I_L, sp.kron(P_0, sp.kron(I_M, sp.kron(I_J, I_R, "csr"), "csr"), "csr"), "csr") + sp.kron(I_L, sp.kron(P_1, sp.kron(I_M, sp.kron(X, I_R, "csr"), "csr"), "csr"), "csr")    
        else:
        #I_{2^j}
           I_L = sp.identity(2**j, complex, "csr")
        
        #I_{2^(i-j-1)}
           I_M = sp.identity(2**(i-j-1), complex, "csr")
        
        #I_{2^(n-i-1)}
           I_R = sp.identity(2**(self.n-i-1), complex, "csr")
        
        #I_L ⊗ I_I ⊗ I_M ⊗ P_0 ⊗ I_R + I_L ⊗ X ⊗ I_M ⊗ P_1 ⊗ I_R
           CNOT_matrix = sp.kron(I_L, sp.kron(I_I, sp.kron(I_M, sp.kron(P_0, I_R, "csr"), "csr"), "csr"), "csr") + sp.kron(I_L, sp.kron(X, sp.kron(I_M, sp.kron(P_1, I_R, "csr"), "csr"), "csr"), "csr")
    #Apply CNOT gate to state
        self.state = CNOT_matrix.dot(self.state)


# In[3]:


#Qubits on Square-Octagon Lattice w/ Lattice Parameters
qubits = [0, 1, 2, 3, 4, 5, 6, 7]
Nq = len(qubits)

xx_links = [(0, 1), (2, 3)]
yy_links = [(0, 3), (1, 2)]
yy_links_reversed = [(1, 2), (0, 3)]
zz_links = [(0, 4), (1, 5), (2, 6), (3, 7)]
zz_links_reversed = [(3, 7), (2, 6), (1, 5), (0, 4)]

Jx, Jy, Jz = -1.0/math.sqrt(2), -1.0/math.sqrt(2), -1.0
hX = hY = hZ = 0.05/math.sqrt(3)


# In[4]:


#Hamiltonian Construction 

def sparse_kron(A, B):
    return sp.kron(A, B, "csr")

#Define Sparse Pauli Operators
I = sp.identity(2, complex, "csr")
X = sp.csr_matrix(np.array([[0,1],[1,0]], complex))
Y = sp.csr_matrix(np.array([[0,-1j],[1j,0]], complex))
Z = sp.csr_matrix(np.array([[1,0],[0,-1]], complex))

#Construct Pauli Lists
XX_links_list = []
for (i,j) in xx_links:
    XX_list = []
    for q in range(Nq):
        if q != i and q != j:
            XX_list.append(I)
        else:
            XX_list.append(X)
    XX_links_list.append(XX_list)
    
YY_links_list = []
for (i,j) in yy_links:
    YY_list = []
    for q in range(Nq):
        if q != i and q != j:
            YY_list.append(I)
        else:
            YY_list.append(Y)
    YY_links_list.append(YY_list)

ZZ_links_list = []
for (i,j) in zz_links:
    ZZ_list = []
    for q in range(Nq):
        if q != i and q != j:
            ZZ_list.append(I)
        else:
            ZZ_list.append(Z)
    ZZ_links_list.append(ZZ_list)

X_qubit_list = []
Y_qubit_list = []
Z_qubit_list = []
for k in qubits:
    X_list = []
    Y_list = []
    Z_list = []
    for q in range(Nq):
        if q != k:
            X_list.append(I)
            Y_list.append(I)
            Z_list.append(I)
        else:
            X_list.append(X)
            Y_list.append(Y)
            Z_list.append(Z)
    X_qubit_list.append(X_list)
    Y_qubit_list.append(Y_list)
    Z_qubit_list.append(Z_list)
    

#Full Hamiltonian
H = 0
for nx in range(Nq):
    H += hX*reduce(sparse_kron, X_qubit_list[nx])
for ny in range(Nq):
    H += hY*reduce(sparse_kron, Y_qubit_list[ny])
for nz in range(Nq):
    H += hZ*reduce(sparse_kron, Z_qubit_list[nz])
for nx in range(len(XX_links_list)):
    H += Jx*reduce(sparse_kron, XX_links_list[nx])
for ny in range(len(YY_links_list)):
    H += Jy*reduce(sparse_kron, YY_links_list[ny])
for nz in range(len(ZZ_links_list)):
    H += Jz*reduce(sparse_kron, ZZ_links_list[nz])


# In[5]:


##Exact Diagonalization of Sparse Hamiltonian (Square-Octagon Model)
#eigvals, eigvecs = sp.linalg.eigs(H)
#print(f"Min eigvalue: {min(eigvals)}")


# In[6]:


##Hamiltonian Variational Ansatz Construction
def HVA_circuit(alpha, beta, gamma, alpha_mag, beta_mag, gamma_mag, n_layers):
    global xx_links
    global yy_links
    global yy_links_reversed
    global zz_links
    global zz_links_reversed
    global qubits
    #Initialize Statevector
    qreg_HVA = SparseState(Nq)
    for idx_layer in range(n_layers):
    #Construct X-terms portion of layer
        for q in range(Nq):
            qreg_HVA.rx(2*alpha_mag[idx_layer], q)
        for (i,j) in xx_links:
            qreg_HVA.hadamard(i)
            qreg_HVA.hadamard(j)
        for (i,j) in xx_links:
            qreg_HVA.CNOT(max(i,j),min(i,j))
        for (i,j) in xx_links:
            qreg_HVA.rz(2*alpha[idx_layer], min(i,j))
        for (i,j) in xx_links:
            qreg_HVA.CNOT(max(i,j),min(i,j))
        for (i,j) in xx_links:
            qreg_HVA.hadamard(i)
            qreg_HVA.hadamard(j)
        #Construct Y-terms portion of layer
        for q in range(Nq):
            qreg_HVA.ry(2*beta_mag[idx_layer], q)
        for (i,j) in yy_links:
            qreg_HVA.rx(np.pi/2,i)
            qreg_HVA.rx(np.pi/2,j)
        for (i,j) in yy_links:
            qreg_HVA.CNOT(max(i,j),min(i,j))
        for (i,j) in yy_links:
            qreg_HVA.rz(2*beta[idx_layer], min(i,j))
        for (i,j) in yy_links_reversed:
            qreg_HVA.CNOT(max(i,j),min(i,j))
        for (i,j) in yy_links:
            qreg_HVA.rx(-np.pi/2,i)
            qreg_HVA.rx(-np.pi/2,j)
        #Construct Z-terms portion of layer
        for q in range(Nq):
            qreg_HVA.rz(2*gamma_mag[idx_layer], q)
        for (i,j) in zz_links_reversed:
            qreg_HVA.CNOT(max(i,j),min(i,j))
        for (i,j) in zz_links:
            qreg_HVA.rz(2*gamma[idx_layer], min(i,j))
        for (i,j) in zz_links:
            qreg_HVA.CNOT(max(i,j),min(i,j))
        HVA = qreg_HVA
    return HVA


# In[7]:


##Construct Objective Function
def objective(params):
    alpha = params[:n_layers]
    beta = params[n_layers:2*n_layers]
    gamma = params[2*n_layers:3*n_layers]
    alpha_mag = params[3*n_layers:4*n_layers]
    beta_mag = params[4*n_layers:5*n_layers]
    gamma_mag = params[5*n_layers:]
    ansatz = HVA_circuit(alpha, beta, gamma, alpha_mag, beta_mag, gamma_mag, n_layers)
    expect = sp.csr_matrix.conjugate(sp.csr_matrix.transpose(ansatz.state)).dot(H.dot(ansatz.state)).toarray()
    return np.real(expect)


# In[8]:


##Perform VQE

#Specify number of layers
n_layers = 1

#Loop that executes VQE
res = pybobyqa.solve(objective, x0=np.random.uniform(-np.pi, np.pi, size=6*n_layers), maxfun = 20000)
optimal_value = res.f
cost_function_evals = res.nf
optimal_parameters = res.x


# In[ ]:


#Save the results to the corresponding text files
#with open('VQE_Statevector_Kitaev_Square-Octagon_Results/8_Qubit_Statevector_Kitaev_Square_Octagon_1_Layer_HVA_Ground_State_Energies.txt', 'a') as f:
    #f.write(str(optimal_value))
    #f.write('\n')
#with open('VQE_Statevector_Kitaev_Square-Octagon_Results/8_Qubit_Statevector_Kitaev_Square_Octagon_1_Layer_HVA_Optimal_Parameters.txt', 'a') as f:
    #np.savetxt(f, optimal_parameters, newline=" ")
    #f.write('\n')
#with open('VQE_Statevector_Kitaev_Square-Octagon_Results/8_Qubit_Statevector_Kitaev_Square_Octagon_1_Layer_HVA_Cost_Function_Evaluations.txt', 'a') as f:
    #f.write(str(cost_function_evals))
    #f.write('\n')

