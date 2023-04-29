import numpy as np
import sys

# Step 1: specifying molecular geometry, basis set and other variables
Nelec = 2
R = 1.4632 #bohr
dim = 2 #dim is the number of basis functions

# Molecular integrals are read from input files
S_int = np.genfromtxt('./s_2.dat',dtype=None) # overlap matrix 
T_int = np.genfromtxt('./t_2.dat',dtype=None) # kinetic energy matrix
V_int = np.genfromtxt('./v_2.dat',dtype=None) # potential energy matrix
te_int = np.genfromtxt('./elec_2.dat') # two electron integrals

# Step 2: Calculate all the required molecular integrals S,T,V and 2-electron
S = np.zeros((dim, dim))
T = np.zeros((dim, dim))
V = np.zeros((dim, dim))

# Put the integrals into a matrix
for i in S_int:
    S[i[0]-1, i[1]-1] = i[2] 
for i in T_int:
    T[i[0]-1, i[1]-1] = i[2]
for i in V_int:
    V[i[0]-1, i[1]-1] = i[2]
    
# Step 3: Diagonalize S and use its eigenvalues to obtain a transformation matrix X
def symmetrise(matrix):
    """
    Function to symmetrize a matrix given a triangular one
    """
    return matrix + matrix.T - np.diag(matrix.diagonal())

# Flip the triangular matrix in the diagonal 
S = symmetrise(S)
V = symmetrise(V)
T = symmetrise(T)
# Form core Hamiltonian matrix as sum of the T and V matrices
Hcore = T + V

# Diagonalize overlap matrix. S_val are the eigenvalues and S_vec the eigenvectors
S_val, S_vec = np.linalg.eigh(S)
# Find inverse square root of eigenvalues
s_half = (np.diag(S_val**(-0.5)))
# Form the transformation matrix X. The unitary matrix are the eigenvectors
X_matrix = -np.dot(S_vec, np.dot(s_half, np.transpose(S_vec)))

# Step 4: Construct a guess denisty matrix P(0) (null)
P = np.zeros((dim, dim))

# Step 5: Calculate the bielectronic term G(0) using the guess denisty matrix and the two electron integrals
def eint(a,b,c,d): 
    if a > b: 
        ab = a*(a+1)/2 + b
    else: 
        ab = b*(b+1)/2 + a
    if c > d: 
        cd = c*(c+1)/2 + d
    else: 
        cd = d*(d+1)/2 + c
    if ab > cd: 
        abcd = ab*(ab+1)/2 + cd
    else: 
        abcd = cd*(cd+1)/2 + ab
    return abcd

# two-electron integrals are stored in a dictionary
twoe = {eint(row[0], row[1], row[2], row[3]) : row[4] for row in te_int}

def two_elec_int(a, b, c, d): # Return value of two electron integral
    """
    Return value of two electron integral
    """
    return twoe.get(eint(a, b, c, d), 0)

# Step 6: Add G(0) to the one-electron term h to get a first guess of the Fock matrix F(0) = h + G(0).
def fock_matrix(Hcore, P, dim): 
    """
    Function to build the Fock Matrix
    """
    F = np.zeros((dim, dim)) # zero array
    for i in range(0, dim):
        for j in range(0, dim):
            F[i,j] = Hcore[i,j] # initial Fock matrix
            for k in range(0, dim):
                for l in range(0, dim):
                    # Form the Fock matrix using the product of the density matrix and G matrix
                    F[i,j] = F[i,j] + P[k,l]*(two_elec_int(i+1,j+1,k+1,l+1)-0.5*two_elec_int(i+1,k+1,j+1,l+1))
    return F 

# Step 7: Transform the Fock matrix
def f_transform(X, F): 
    """
    Transform Fock matrix with the transformation matrix X
    """
    return np.dot(np.transpose(X), np.dot(F, X)) 

# Step 10: Form a new density matrix P(1) using C(1)
def density_matrix(C, D, dim, Nelec): # Make density matrix and store old one to test for convergence
    """
    Make new density matrix and store old one to test for convergence
    
    Returns:
        D: new density matrix
        Dold: old denisty matrix
    """
    Dold = np.zeros((dim, dim)) # Initiate zero array
    for mu in range(0, dim):
        for nu in range(0, dim):
            Dold[mu,nu] = D[mu, nu] # Set old density matrix to the density matrix, D, input into the function 
            D[mu,nu] = 0
            for m in range(0, int(Nelec/2)):
                # Form new density matrix
                D[mu,nu] = D[mu,nu] + 2*C[mu,m]*C[nu,m]
    return D, Dold

# Step 11: Determine if the process has converged by comparing P(1) with P(0).
def threshold(D, Dold):
    """
    Calculate change in density matrix using Root Mean Square Deviation (RMSD)
    """
    DELTA = 0.0
    for i in range(0, dim):
        for j in range(0, dim):
            DELTA = DELTA + ((D[i,j] - Dold[i,j])**2)

    return (DELTA/4.0)**(0.5)

# Step 12: If the process has converged, use the resultant solution, represented by C(k), P(k), and F(k)
def energy_iteration(D, Hcore, F, dim):
    """
    Function that calculates the energy at each iteration 
    """
    EN = 0
    for mu in range(0, dim):
        for nu in range(0, dim):
            EN += 0.5*D[mu,nu]*(Hcore[mu,nu] + F[mu,nu])
    return EN

# Finally we make the iteration loop
DELTA = 1
count = 0 # cycles counter
nuclear_repulsion = 2/R
while DELTA > 1e-4:
    count += 1 
    F = fock_matrix(Hcore, P, dim) # Calculate Fock matrix (step 6)
    Fprime = f_transform(X_matrix, F) # Calculate transformed Fock matrix (step 7)
    E, Cprime = np.linalg.eigh(Fprime) # Diagonalize F' matrix (step 8)
    C = np.dot(X_matrix, Cprime) # transform the coefficients into original basis using transformation matrix (step 9)
    P, OLDP = density_matrix(C, P, dim, Nelec) # Make density matrix (step 10)
    DELTA = threshold(P, OLDP) # Test for convergence (step 11)
    print("E = {:.6f}, N(SCF) = {}".format(energy_iteration(P, Hcore, F, dim) + nuclear_repulsion, count))

print("SCF procedure complete, TOTAL E(SCF) = {} a.u.".format(energy_iteration(P, Hcore, F, dim) + nuclear_repulsion))
print("------------------------")
print("The expansion coefficients matrix is: ","\n", C)
print("------------------------")
print("The orbital energies matrix is: ","\n", np.diag(E))
print("------------------------")
print("The final delta value is: ",DELTA)