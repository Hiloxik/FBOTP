from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d # Hilbert space fermion basis
from quspin.tools.block_tools import block_diag_hamiltonian # block diagonalisation
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
try: # import python 3 zip function in python 2 and pass if already using python 3
    import itertools.izip as zip
except ImportError:
    pass 
##### define model parameters #####
L=8 # system size
t=1.0 # uniform hopping
delta=-0.2 

##### construct single-particle Hamiltonian #####
# define site-coupling lists
hop_pm=[[t+delta*(-1)**i,i,i+1] for i in range(L-1)] # OBC
hop_mp=[[-(t+delta*(-1)**i),i,i+1] for i in range(L-1)] # OBC

# define static and dynamic lists
static=[["+-",hop_pm],["-+",hop_mp]]
dynamic=[]

# define basis
basis=spinless_fermion_basis_1d(L,Nf=1)

# build real-space Hamiltonian
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)

# diagonalise real-space Hamiltonian
E,V=H.eigh()
print("\nbasis is",basis)
print("\nH=\n",H.toarray())
