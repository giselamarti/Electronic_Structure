 # PROBLEM 3
import numpy as np
import matplotlib.pyplot as plt

# Coeff is the d parameter
coeff = np.array([[1.00000,0.0000000,0.000000],
                  [0.678914,0.430129,0.000000],
                  [0.444635,0.535328,0.154329]])

# Expon is the alpha parameter
expon = np.array([[0.270950,0.000000,0.000000],
                  [0.151623,0.851819,0.000000],
                  [0.109818,0.405771,2.227660]])

x = np.linspace(-5,5,1000)
r = abs(x)
zeta = 1.0

psi_STO = (zeta**3/np.pi)**(0.5)*np.exp(-zeta*r)
psi_STO_squared = psi_STO**2
radial_dist = 4*np.pi*r**2*psi_STO_squared

psi_CGF_STO1G = coeff[0,0]*(2*expon[0,0]/np.pi)**(0.75)*np.exp(-expon[0,0]*r**2)
psi_CGF_STO2G = coeff[1,0]*(2*expon[1,0]/np.pi)**(0.75)*np.exp(-expon[1,0]*r**2) + coeff[1,1]*(2*expon[1,1]/np.pi)**(0.75)*np.exp(-expon[1,1]*r**2) + coeff[1,2]*(2*expon[1,2]/np.pi)**(0.75)*np.exp(-expon[1,2]*r**2)
psi_CGF_STO3G = coeff[2,0]*(2*expon[2,0]/np.pi)**(0.75)*np.exp(-expon[2,0]*r**2) + coeff[2,1]*(2*expon[2,1]/np.pi)**(0.75)*np.exp(-expon[2,1]*r**2) + coeff[2,2]*(2*expon[2,2]/np.pi)**(0.75)*np.exp(-expon[2,2]*r**2)

psi_CGF_STO1G_squared = psi_CGF_STO1G**2
psi_CGF_STO2G_squared = psi_CGF_STO2G**2
psi_CGF_STO3G_squared = psi_CGF_STO3G**2

radial_dist_STO1G = 4*np.pi*r**2*psi_CGF_STO1G_squared
radial_dist_STO2G = 4*np.pi*r**2*psi_CGF_STO2G_squared
radial_dist_STO3G = 4*np.pi*r**2*psi_CGF_STO3G_squared

plt.figure(figsize=(9,6))
plt.title("Wavefunctions for 1s STO, STO-1G, STO-2G and STO-3G")
plt.xlabel("r (A)")
plt.ylabel("$\Psi$")
plt.plot(x, psi_STO, label="STO", color="red")
plt.plot(x, psi_CGF_STO1G, label="STO-1G", color="green")
plt.plot(x, psi_CGF_STO2G, label="STO-2G", color="blue")
plt.plot(x, psi_CGF_STO3G, label="STO-3G", color="orange")
plt.legend()
plt.grid()

plt.figure(figsize=(9,6))
plt.title("Squared wavefunctions for 1s STO, STO-1G, STO-2G and STO-3G")
plt.xlabel("r (A)")
plt.ylabel("$\Psi^2$")
plt.plot(x, psi_STO_squared, label="STO", color="red")
plt.plot(x, psi_CGF_STO1G_squared, label="STO-1G", color="green")
plt.plot(x, psi_CGF_STO2G_squared, label="STO-2G", color="blue")
plt.plot(x, psi_CGF_STO3G_squared, label="STO-3G", color="orange")
plt.legend()
plt.grid()

plt.figure(figsize=(9,6))
plt.title("Radial distribution functions for 1s STO, STO-1G, STO-2G and STO-3G")
plt.xlabel("r (A)")
plt.ylabel("P(r)")
plt.plot(x, radial_dist, label="STO", color="red")
plt.plot(x, radial_dist_STO1G, label="STO-1G", color="green")
plt.plot(x, radial_dist_STO2G, label="STO-2G", color="blue")
plt.plot(x, radial_dist_STO3G, label="STO-3G", color="orange")
plt.legend()
plt.grid()

# The most probable electron-nucleus distance corresponds to the peak of each radial distribution function
# a_o (angstroms)
print("Most probable electron-nucleus distance for 1s STO: ", radial_dist.max())
print("Most probable electron-nucleus distance for 1s STO-1G: ", radial_dist_STO1G.max())
print("Most probable electron-nucleus distance for 1s STO-2G: ", radial_dist_STO2G.max())
print("Most probable electron-nucleus distance for 1s STO-3G: ", radial_dist_STO3G.max())

# The average electron-nucleus distance
# r = 3*a_o/2 (angstroms)
print("Average electron-nucleus distance for 1s STO: ", 3*radial_dist.max()/2)
print("Average electron-nucleus distance for 1s STO-1G: ", 3*radial_dist_STO1G.max()/2)
print("Average electron-nucleus distance for 1s STO-2G: ", 3*radial_dist_STO2G.max()/2)
print("Average electron-nucleus distance for 1s STO-3G: ", 3*radial_dist_STO3G.max()/2)

# radius of a sphere around the nucleus containing the electron with a 99% of probability
# r = 4.2*a_o (angstroms)
print("Radius of sphere with 99% probability for 1s STO: ", 4.2*radial_dist.max())
print("Radius of sphere with 99% probability for 1s STO-1G: ", 4.2*radial_dist_STO1G.max())
print("Radius of sphere with 99% probability for 1s STO-2G: ", 4.2*radial_dist_STO2G.max())
print("Radius of sphere with 99% probability for 1s STO-3G: ", 4.2*radial_dist_STO3G.max())

 # REPEAT FOR He+
 
zeta_He = 2.0
coeff_He = coeff
expon_He = expon*zeta_He**2

psi_STO_He = (zeta_He**3/np.pi)**(0.5)*np.exp(-zeta_He*r)
psi_STO_squared_He = psi_STO_He**2
radial_dist_He = 4*np.pi*r**2*psi_STO_squared_He

psi_CGF_STO1G_He = coeff_He[0,0]*(2*expon_He[0,0]/np.pi)**(0.75)*np.exp(-expon_He[0,0]*r**2)
psi_CGF_STO2G_He = coeff_He[1,0]*(2*expon_He[1,0]/np.pi)**(0.75)*np.exp(-expon_He[1,0]*r**2) + coeff_He[1,1]*(2*expon_He[1,1]/np.pi)**(0.75)*np.exp(-expon_He[1,1]*r**2) + coeff_He[1,2]*(2*expon_He[1,2]/np.pi)**(0.75)*np.exp(-expon_He[1,2]*r**2)
psi_CGF_STO3G_He = coeff_He[2,0]*(2*expon_He[2,0]/np.pi)**(0.75)*np.exp(-expon_He[2,0]*r**2) + coeff_He[2,1]*(2*expon_He[2,1]/np.pi)**(0.75)*np.exp(-expon_He[2,1]*r**2) + coeff_He[2,2]*(2*expon_He[2,2]/np.pi)**(0.75)*np.exp(-expon_He[2,2]*r**2)

psi_CGF_STO1G_squared_He = psi_CGF_STO1G_He**2
psi_CGF_STO2G_squared_He = psi_CGF_STO2G_He**2
psi_CGF_STO3G_squared_He = psi_CGF_STO3G_He**2

radial_dist_STO1G_He = 4*np.pi*r**2*psi_CGF_STO1G_squared_He
radial_dist_STO2G_He = 4*np.pi*r**2*psi_CGF_STO2G_squared_He
radial_dist_STO3G_He = 4*np.pi*r**2*psi_CGF_STO3G_squared_He

plt.figure(figsize=(9,6))
plt.title("Wavefunctions for 1s STO, STO-1G, STO-2G and STO-3G (He+)")
plt.xlabel("r (A)")
plt.ylabel("$\Psi$")
plt.plot(x, psi_STO_He, label="STO", color="red")
plt.plot(x, psi_CGF_STO1G_He, label="STO-1G", color="green")
plt.plot(x, psi_CGF_STO2G_He, label="STO-2G", color="blue")
plt.plot(x, psi_CGF_STO3G_He, label="STO-3G", color="orange")
plt.legend()
plt.grid()

plt.figure(figsize=(9,6))
plt.title("Squared wavefunctions for 1s STO, STO-1G, STO-2G and STO-3G (He+)")
plt.xlabel("r (A)")
plt.ylabel("$\Psi^2$")
plt.plot(x, psi_STO_squared_He, label="STO", color="red")
plt.plot(x, psi_CGF_STO1G_squared_He, label="STO-1G", color="green")
plt.plot(x, psi_CGF_STO2G_squared_He, label="STO-2G", color="blue")
plt.plot(x, psi_CGF_STO3G_squared_He, label="STO-3G", color="orange")
plt.legend()
plt.grid()

plt.figure(figsize=(9,6))
plt.title("Radial distribution functions for 1s STO, STO-1G, STO-2G and STO-3G (He+)")
plt.xlabel("r (A)")
plt.ylabel("P(r)")
plt.plot(x, radial_dist_He, label="STO", color="red")
plt.plot(x, radial_dist_STO1G_He, label="STO-1G", color="green")
plt.plot(x, radial_dist_STO2G_He, label="STO-2G", color="blue")
plt.plot(x, radial_dist_STO3G_He, label="STO-3G", color="orange")
plt.legend()
plt.grid()

# The most probable electron-nucleus distance for He+
print("Most probable electron-nucleus distance for 1s STO (He+): ", radial_dist_He.max())
print("Most probable electron-nucleus distance for 1s STO-1G (He+): ", radial_dist_STO1G_He.max())
print("Most probable electron-nucleus distance for 1s STO-2G (He+): ", radial_dist_STO2G_He.max())
print("Most probable electron-nucleus distance for 1s STO-3G (He+): ", radial_dist_STO3G_He.max())

# The average electron-nucleus distance is for He+
print("Average electron-nucleus distance for 1s STO (He+): ", 3*radial_dist_He.max()/2)
print("Average electron-nucleus distance for 1s STO-1G (He+): ", 3*radial_dist_STO1G_He.max()/2)
print("Average electron-nucleus distance for 1s STO-2G (He+): ", 3*radial_dist_STO2G_He.max()/2)
print("Average electron-nucleus distance for 1s STO-3G (He+): ", 3*radial_dist_STO3G_He.max()/2)

# radius of a sphere around the nucleus containing the electron with a 99% of probability for He+
print("Radius of sphere with 99% probability for 1s STO (He+): ", 4.2*radial_dist_He.max())
print("Radius of sphere with 99% probability for 1s STO-1G (He+): ", 4.2*radial_dist_STO1G_He.max())
print("Radius of sphere with 99% probability for 1s STO-2G (He+): ", 4.2*radial_dist_STO2G_He.max())
print("Radius of sphere with 99% probability for 1s STO-3G (He+): ", 4.2*radial_dist_STO3G_He.max())
