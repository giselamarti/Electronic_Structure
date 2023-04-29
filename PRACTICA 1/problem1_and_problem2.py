import numpy as np
from scipy.optimize import minimize
from sympy import *

 # PROBLEM 1

def min(a):
    """
    Function to calculate the energy as a function of alpha

    """
    S = (np.pi/(2*a))**(3/2)
    T = 3*a**2*np.pi**(3/2) / (2*a)**(5/2)
    V = -2*np.pi / (2*a)
    E = (T+V)/S
    return E

# Provide an initial guess for the variational parameter a
a = 1
# Optimise the min function by varying a
ground_state = minimize(min, a)
print("Optimized alpha: ",ground_state.x,". Optimized ground state energy: ",ground_state.fun,"a.u.")

# Calculate the error
E_real = -0.5
error = abs(E_real - ground_state.fun)*100*2
print("The error is: ",error,"%")

 # PROBLEM 2

zeta, r, alpha, a1 = symbols("zeta r alpha a1", positive=True) 

sto = (zeta **3 / pi) ** (1/2) * exp(-zeta * r)  # general expression for one STO
gto = (2 * alpha /pi) ** (3/4) * exp(-alpha * r**2)  # general expression for one GF

sto_1 = sto.subs(zeta, 1.0)  # zeta = 1.0
gto_1 = gto.subs(alpha, a1)  # alpha = a1

S = integrate(sto_1 * gto_1 * r**2, (r, 0, np.inf)) * 4 * pi  # the overlap between STO(1.0, r) and GF(alpha, r)

#We maximize this integral in terms of a1. We turn the maximization problem into minimization of the negative of the overlap S
def func(a):
    res = S.subs(a1, a[0]).evalf()
    return -res

res = minimize(func, x0=[0.2])
print("The optimal alpha exponent is: ", res.x[0])
print("The ground state energy for the 1s STO-1G function is: ", min(res.x[0]),"a.u.")
error = abs(E_real - min(res.x[0]))*100*2
print("The error is: ",error,"%")