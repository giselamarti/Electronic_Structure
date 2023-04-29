import numpy as np
import matplotlib.pyplot as plt

def energies(k, R):
    Saa = Sbb = 1
    Sab = Sba = np.exp(-k*R) * (1+k*R+(1/3)*(k**2)*(R**2))
    Haa = Hbb = (1/2)*(k**2) - k - 1/R + np.exp(-2*k*R) * (k+1/R)
    Hab = Hba = -1/2 * (k**2) * Sab + k*(k-2)*(1+k*R)*np.exp(-k*R)
    w1 = (Haa + Hab)/(1 + Sab)
    w2 = (Haa - Hab)/(1- Sab)
    return w1,w2

# Aproximation for fixed values of k = 1.5 and R = 1.5
print("The ground state energy is: ", energies(1.5, 1.5)[0], "a.u.")
print("The first excited state energy is: ", energies(1.5, 1.5)[1], "a.u.")

matrix_size = 100 #matrix size = 100x100, where columns are R and rows are k
bohr = 1.88972 # change units from angstroms to bohr radius
R = np.linspace(0.8*bohr,2.0*bohr,matrix_size)
k = np.linspace(0.4,2.0,matrix_size)
E = []
for i in range(matrix_size):
    for j in range(matrix_size):
        E.append(energies(k[i], R[j])[0])
energy_values = np.array(E)
k_index = np.where(energy_values == min(energy_values))
optimal_k_index = k_index[0] / matrix_size
optimal_k = k[int(optimal_k_index)]
print("The optimal k value is: ", optimal_k)

#Ground state energy PES
PES_ground = energies(optimal_k, R)[0] + (1/R)
#First excited state PES
PES_excited = energies(optimal_k, R)[1] + (1/R)

plt.figure(figsize=(9,6))
plt.title("1D potential energy surface for the ground state and the frist excited state")
plt.xlabel("r")
plt.ylabel("E")
plt.plot(R, PES_ground, label="ground state", color="red")
plt.plot(R, PES_excited, label="first excited state", color="green")
plt.legend()
plt.grid()

R_index = np.where(PES_ground == min(PES_ground))
R_equilibrium = R[int(R_index[0])]
print("The equilibirum distance for the H2+ molecule is: ",R_equilibrium, "bohr units")