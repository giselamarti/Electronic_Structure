import numpy as np
import scipy.special as sp

N = 3 #number of Gaussian primitives
R = 1.4 #intranuclear distance between the two atoms
zeta = 1.24
Za = Zb = 1.0
H = np.zeros([2,2])
S = np.zeros([2,2])
Rab2 = R**2

def S_int(A,B,Rab2): #A and B are the exponents of the atoms
    """
    Calculates the overlap between two gaussian functions 
    """
    return (np.pi/(A+B))**1.5*np.exp(-A*B*Rab2/(A+B))

def T_int(A,B,Rab2):
    """
    Calculates the kinetic energy integrals for un-normalised primitives
    """
    return A*B/(A+B)*(3.0-2.0*A*B*Rab2/(A+B))*(np.pi/(A+B))**1.5*np.exp(-A*B*Rab2/(A+B))

def V_int(A,B,Rab2,Rcp2,Zc):
    """
    Calculates the un-normalised nuclear attraction integrals
    """
    V = 2.0*np.pi/(A+B)*F0((A+B)*Rcp2)*np.exp(-A*B*Rab2/(A+B))
    return -V*Zc

# Mathematical functions
def F0(t):
    """
    F function for 1s orbital
    """
    if (t<1e-6):
        return 1.0-t/3.0
    else:
        return 0.5*(np.pi/t)**0.5*sp.erf(t**0.5)
    
def erf(t):
    """
    Approximation for the error function
    """
    P = 0.3275911
    A = [0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429]
    T = 1.0/(1+P*t)
    Tn=T
    Poly = A[0]*Tn
    for i in range(1,5):
        Tn=Tn*T
        Poly=Poly*A[i]*Tn
    return 1.0-Poly*np.exp(-t*t)

coeff = np.array([[1.00000,0.0000000,0.000000],  #STO-1G
                  [0.678914,0.430129,0.000000],  #STO-2G
                  [0.444635,0.535328,0.154329]]) #STO-3G
    
expon = np.array([[0.270950,0.000000,0.000000],  #STO-1G
                  [0.151623,0.851819,0.000000],  #STO-2G
                  [0.109818,0.405771,2.227660]]) #STO-3G

D1 = np.zeros([3])
A1 = np.zeros([3])
D2 = np.zeros([3])
A2 = np.zeros([3])

# This loop constructs the contracted Gaussian functions
for i in range(N):
    A1[i] = expon[N-1,i]*(zeta**2)
    D1[i] = coeff[N-1,i]*((2.0*A1[i]/np.pi)**0.75)
    A2[i] = expon[N-1,i]*(zeta**2)
    D2[i] = coeff[N-1,i]*((2.0*A2[i]/np.pi)**0.75)
    
S12 = 0.0
T11 = 0.0
T12 = 0.0
T22 = 0.0
V11A = 0.0
V12A = 0.0
V22A = 0.0
V11B = 0.0
V12B = 0.0
V22B = 0.0

for i in range(N):
    for j in range(N):
        Rap = A2[j]*R/(A1[i]+A2[j])
        Rap2 = Rap**2 # Rap2 - squared distance between centre A and centre P
        Rbp2 = (R-Rap)**2
        S12 = S12 + S_int(A1[i],A2[j],Rab2)*D1[i]*D2[j]
        T11 = T11 + T_int(A1[i],A1[j],0.0)*D1[i]*D1[j]
        T12 = T12 + T_int(A1[i],A2[j],Rab2)*D1[i]*D2[j]
        T22 = T22 + T_int(A2[i],A2[j],0.0)*D2[i]*D2[j]
        V11A = V11A + V_int(A1[i],A1[j],0.0,0.0,Za)*D1[i]*D1[j]
        V12A = V12A + V_int(A1[i],A2[j],Rab2,Rap2,Za)*D1[i]*D2[j]
        V22A = V22A + V_int(A2[i],A2[j],0.0,Rab2,Za)*D2[i]*D2[j]
        V11B = V11B + V_int(A1[i],A1[j],0.0,Rab2,Zb)*D1[i]*D1[j]
        V12B = V12B + V_int(A1[i],A2[j],Rab2,Rbp2,Zb)*D1[i]*D2[j]
        V22B = V22B + V_int(A2[i],A2[j],0.0,0.0,Zb)*D2[i]*D2[j]
        
# Form core hamiltonian
H[0,0] = T11+V11A+V11B #Haa
H[0,1] = T12+V12A+V12B #Hab
H[1,0] = H[0,1]        #Hba
H[1,1] = T22+V22A+V22B #Hbb

# Form overlap matrix
S[0,0] = 1.0 #Saa
S[0,1] = S12 #Sab
S[1,0] = S12 #Sba
S[1,1] = 1.0 #Sbb

w1 = (H[0,0]+H[0,1]) / (1+S[0,1])
w2 = (H[0,0]-H[0,1]) / (1-S[0,1])
print("The ground state energy of the H2+ molecule is: ", w1, "a.u.")
print("The first excited state energy of the H2+ molecule is: ", w2, "a.u.")