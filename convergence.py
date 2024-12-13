# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 22:46:54 2023

@author: Marvin
"""

#Required Packages
import numpy as np
#import matplotlib.pyplot as plt
import scheme
import sys

######## Setting which can be changed
mu=1
nu=1
R=2**12
sigma=1*np.sqrt(R)
T_f=1/R
reg=0  #W has H^r regularity
epsilon=10**(-3)
samples=int(sys.argv[1]) #Monte Carlo
########################

#Spatial
N_max=2**14
N_steps=6
N_start=9 # 2**N_start number of space discretizations for coarsest discretization
N_s=np.logspace(N_start,N_steps+N_start-1,num=N_steps,base=2,dtype='int')
x=np.linspace(0,1,N_max)

######################### Choice initial condition

# Initial condition is zero
u_0=np.zeros(N_max)
#u_0[0]=0.5*np.sqrt(R)

##################################

#Temporal
k=4  # Multiplication factor at every discretization step
M_exact=2**16
M_steps=6 # Number of discretizations
M_start=6 # 2**M_start time steps for the coarsest discretization
M_s=np.logspace(M_start,2*M_steps+M_start-2,num=M_steps,base=2,dtype='int')
M_Multi=M_exact//M_s
dt=T_f/M_exact
dt_s=T_f/M_s




#Generating matrices
diag=np.zeros([M_steps+1,N_max],dtype=complex)  
A_diag=np.zeros([M_steps+1,N_max],dtype=complex)  
var=np.zeros([M_steps+1,N_max])

diag[-1],A_diag[-1],var[-1]=scheme.generate_matrices(dt,nu,sigma,N_max)

for i in range(M_steps):
    diag[i],A_diag[i],var[i]=scheme.generate_matrices(dt_s[i],nu,sigma,N_max)

## Creating error arrays
S_error_exa=np.zeros([M_steps-1,samples])
N_error_exa=np.zeros([M_steps-1,samples])
P_error_exa=np.zeros([M_steps-1,samples])

S_error_exp=np.zeros([M_steps-1,samples])
N_error_exp=np.zeros([M_steps-1,samples])
P_error_exp=np.zeros([M_steps-1,samples])

S_error_tam=np.zeros([M_steps-1,samples])
N_error_tam=np.zeros([M_steps-1,samples])
P_error_tam=np.zeros([M_steps-1,samples])

## Solving problem

for i in range(samples):
    for j in range(M_steps-1):
        sqrt_q=scheme.sqrt_q_generate(reg,epsilon,N_s[j+1])
        Var=np.real(scheme.P_N(var[j+1],N_s[j+1]))
        sqrt_qvar=scheme.get_sqrt_qvar(sqrt_q,Var)
        sqrt_qdt=scheme.get_sqrt_qdt(sqrt_q,dt_s[j+1])
        diag_N = scheme.P_N(diag[j+1],N_s[j+1])
        A_diag_N=scheme.P_N(A_diag[j],N_s[j])
        A_diag_2N = scheme.P_N(A_diag[j+1],N_s[j+1])
        P_N_u_0= scheme.P_N(u_0,N_s[j+1])/N_max*N_s[j]
        P_error_exa[j,i],N_error_exa[j,i]=scheme.Exact_splitting_method(dt_s[j+1],M_s[j+1],N_s[j+1],N_s[j],R,mu,sigma,diag_N,sqrt_qvar,P_N_u_0,k)
        P_error_exp[j,i],N_error_exp[j,i]=scheme.Explicit_splitting_method(dt_s[j+1],M_s[j+1],N_s[j+1],N_s[j],R,mu,sigma,diag_N,A_diag_2N,A_diag_N,sqrt_qdt,P_N_u_0,k)
        P_error_tam[j,i],N_error_tam[j,i]=scheme.Tamed_method(dt_s[j+1],M_s[j+1],N_s[j+1],N_s[j],R,mu,sigma,diag_N,A_diag_2N,A_diag_N,sqrt_qvar,P_N_u_0,k)


