#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing
import concurrent.futures as cf

import matplotlib.pyplot as plt
import numpy as np
import math
import pybobyqa

import cirq
import sympy as sp


# In[2]:


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


# In[3]:


#Hamiltonian Construction

#Qubit register for Hamiltonian
qreg = cirq.NamedQubit.range(Nq,prefix="q")

#Constructing Pauli Strings
XX_links_strings = []
for (i,j) in xx_links:
    XX_string = []
    for q in range(Nq):
        if q != i and q != j:
            XX_string.append(cirq.I(qreg[q]))
        else:
            XX_string.append(cirq.X(qreg[q]))
    XX_links_strings.append(XX_string)
    
YY_links_strings = []
for (i,j) in yy_links:
    YY_string = []
    for q in range(Nq):
        if q != i and q != j:
            YY_string.append(cirq.I(qreg[q]))
        else:
            YY_string.append(cirq.Y(qreg[q]))
    YY_links_strings.append(YY_string)

ZZ_links_strings = []
for (i,j) in zz_links:
    ZZ_string = []
    for q in range(Nq):
        if q != i and q != j:
            ZZ_string.append(cirq.I(qreg[q]))
        else:
            ZZ_string.append(cirq.Z(qreg[q]))
    ZZ_links_strings.append(ZZ_string)

X_qubit_strings = []
Y_qubit_strings = []
Z_qubit_strings = []
for k in qubits:
    X_string = []
    Y_string = []
    Z_string = []
    for q in range(Nq):
        if q != k:
            X_string.append(cirq.I(qreg[q]))
            Y_string.append(cirq.I(qreg[q]))
            Z_string.append(cirq.I(qreg[q]))
        else:
            X_string.append(cirq.X(qreg[q]))
            Y_string.append(cirq.Y(qreg[q]))
            Z_string.append(cirq.Z(qreg[q]))
    X_qubit_strings.append(X_string)
    Y_qubit_strings.append(Y_string)
    Z_qubit_strings.append(Z_string)


# In[4]:


#Hamiltonian Construction (ctd.)

#Convert Pauli Strings into Products
XX_links_terms = []
for XX_link in XX_links_strings:
    XX_links_terms.append(cirq.PauliString(XX_link))

YY_links_terms = []
for YY_link in YY_links_strings:
    YY_links_terms.append(cirq.PauliString(YY_link))
    
ZZ_links_terms = []
for ZZ_link in ZZ_links_strings:
    ZZ_links_terms.append(cirq.PauliString(ZZ_link))

X_qubit_terms = []
for qubit_string in X_qubit_strings:
    X_qubit_terms.append(cirq.PauliString(qubit_string))

Y_qubit_terms = []
for qubit_string in Y_qubit_strings:
    Y_qubit_terms.append(cirq.PauliString(qubit_string))

Z_qubit_terms = []
for qubit_string in Z_qubit_strings:
    Z_qubit_terms.append(cirq.PauliString(qubit_string))


# In[5]:


#Hamiltonian Construction (ctd.)

#Add up all the terms
Hamiltonian = 0
for XX_link in XX_links_terms:
    Hamiltonian += Jx*XX_link
for YY_link in YY_links_terms:
    Hamiltonian += Jy*YY_link
for ZZ_link in ZZ_links_terms:
    Hamiltonian += Jz*ZZ_link
for X_qubit in X_qubit_terms:
    Hamiltonian += hX*X_qubit
for Y_qubit in Y_qubit_terms:
    Hamiltonian += hY*Y_qubit
for Z_qubit in Z_qubit_terms:
    Hamiltonian += hZ*Z_qubit


# In[6]:


##Exact Diagonalization of Hamiltonian (Square-Octagon Model)
#eigvals, eigvecs = np.linalg.eig(Hamiltonian.matrix())
#print(f"Min eigvalue: {min(eigvals)}")


# In[7]:


##Hamiltonian Variational Ansatz Circuit Construction
def HVA_circuit(alpha, beta, gamma, alpha_mag, beta_mag, gamma_mag, n_layers):
    global xx_links
    global yy_links
    global yy_links_reversed
    global zz_links
    global zz_links_reversed
    global qubits
    #Initialize Circuit
    HVA = cirq.Circuit()
    qreg_HVA = cirq.NamedQubit.range(Nq, prefix="q")
    #Declare variational parameters
    #alpha = sp.symarray('alpha', n_layers)
    #beta = sp.symarray('beta', n_layers)
    #gamma = sp.symarray('gamma', n_layers)
    #alpha_mag = sp.symarray('alpha_mag', n_layers)
    #beta_mag = sp.symarray('beta_mag', n_layers)
    #gamma_mag = sp.symarray('gamma_mag', n_layers)
    #Ansatz Construction
    for idx_layer in range(n_layers):
    #Construct X-terms portion of layer
        HVA.append(cirq.rx(2*alpha_mag[idx_layer]).on(q) for q in qreg_HVA)
        for (i,j) in xx_links:
            HVA.append(cirq.H(qreg_HVA[i]))
            HVA.append(cirq.H(qreg_HVA[j]))
            HVA.append(cirq.CNOT(qreg_HVA[max(i,j)],qreg_HVA[min(i,j)]))
            HVA.append(cirq.rz(2*alpha[idx_layer]).on(qreg_HVA[min(i,j)]))
            HVA.append(cirq.CNOT(qreg_HVA[max(i,j)],qreg_HVA[min(i,j)]))
            HVA.append(cirq.H(qreg_HVA[i]))
            HVA.append(cirq.H(qreg_HVA[j]))
    #Construct Y-terms portion of layer
        HVA.append(cirq.ry(2*beta_mag[idx_layer]).on(q) for q in qreg_HVA)
        for (i,j) in yy_links:
            HVA.append(cirq.rx(math.pi/2).on(qreg_HVA[i]))
            HVA.append(cirq.rx(math.pi/2).on(qreg_HVA[j]))
            HVA.append(cirq.CNOT(qreg_HVA[max(i,j)],qreg_HVA[min(i,j)]))
        for (i,j) in yy_links_reversed:
            HVA.append(cirq.rz(2*beta[idx_layer]).on(qreg_HVA[min(i,j)]))
            HVA.append(cirq.CNOT(qreg_HVA[max(i,j)],qreg_HVA[min(i,j)]))
            HVA.append(cirq.rx(-math.pi/2).on(qreg_HVA[i]))
            HVA.append(cirq.rx(-math.pi/2).on(qreg_HVA[j]))
    #Construct Z-terms portion of layer
        HVA.append(cirq.rz(2*gamma_mag[idx_layer]).on(q) for q in qreg_HVA)
        for (i,j) in zz_links:
            HVA.append(cirq.CNOT(qreg_HVA[max(i,j)],qreg_HVA[min(i,j)]))
        for (i,j) in zz_links_reversed:
            HVA.append(cirq.rz(2*gamma[idx_layer]).on(qreg_HVA[min(i,j)]))
            HVA.append(cirq.CNOT(qreg_HVA[max(i,j)],qreg_HVA[min(i,j)]))
    return HVA


# In[8]:


##Construct Objective Function
def objective(params, hamiltonian, qreg, simulator):
    alpha = params[:n_layers]
    beta = params[n_layers:2*n_layers]
    gamma = params[2*n_layers:3*n_layers]
    alpha_mag = params[3*n_layers:4*n_layers]
    beta_mag = params[4*n_layers:5*n_layers]
    gamma_mag = params[5*n_layers:]
    ansatz = HVA_circuit(alpha, beta, gamma, alpha_mag, beta_mag, gamma_mag, n_layers)
    psi = ansatz.final_state_vector()
    expect = hamiltonian.expectation_from_state_vector(psi, qubit_map={q: i for i, q in enumerate(qreg)})
    return np.real(expect)


# In[ ]:


##Perform VQE

#Initialize the simulator
simulator = cirq.Simulator()
qreg = cirq.NamedQubit.range(Nq, prefix="q")

#Specify number of layers
n_layers = 4

#Loop that executes VQE
res = pybobyqa.solve(objective, x0=np.random.uniform(-np.pi, np.pi, size=6*n_layers), args=(Hamiltonian, qreg, simulator), maxfun = 2000)
optimal_value = res.f
cost_function_evals = res.nf
optimal_parameters = res.x


# In[ ]:


#Save the results to the corresponding text files
with open('VQE_Kitaev_Square-Octagon_Results/8_Qubit_Kitaev_Square_Octagon_4_Layer_HVA_Ground_State_Energies.txt', 'a') as f:
    f.write(str(optimal_value))
    f.write('\n')
with open('VQE_Kitaev_Square-Octagon_Results/8_Qubit_Kitaev_Square_Octagon_4_Layer_HVA_Optimal_Parameters.txt', 'a') as f:
    np.savetxt(f, optimal_parameters, newline=" ")
    f.write('\n')
with open('VQE_Kitaev_Square-Octagon_Results/8_Qubit_Kitaev_Square_Octagon_4_Layer_HVA_Cost_Function_Evaluations.txt', 'a') as f:
    f.write(str(cost_function_evals))
    f.write('\n')

