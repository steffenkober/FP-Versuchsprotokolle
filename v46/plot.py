import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const


####################################################
#Definition der Funktionen
####################################################
def bogenmin_zu_rad(zahl):
              return (np.floor(zahl)+(zahl - np.floor(zahl))*100/60)*np.pi/180

def quadratic(x,a,b,c):
              return a*x**2+b*x+c

def linear(x,a,b):
              return a*x+b

####################################################
####################################################



####################################################
#Einlesen der Messdaten
####################################################
z, B = np.genfromtxt("content/Messdaten_B_Feld.txt", unpack=True)
lambdas, u_theta_1_rein, u_theta_2_rein = np.genfromtxt("content/Messdaten_GaAs_hochrein.txt", unpack=True)
theta_1_rein = bogenmin_zu_rad(u_theta_1_rein)
theta_2_rein = bogenmin_zu_rad(u_theta_2_rein)
d_rein = 5.11*10**-3 # meter
u_theta_1_12, u_theta_2_12 = np.genfromtxt("content/Messdaten_GaAs_n_dot_1_2.txt", unpack=True)
theta_1_12 = bogenmin_zu_rad(u_theta_1_12)
theta_2_12 = bogenmin_zu_rad(u_theta_2_12)
d_12 = 1.36*10**-3 # meter
N_12 = 1.2*10**18*10**6 #1/m^3
u_theta_1_28, u_theta_2_28 = np.genfromtxt("content/Messdaten_GaAs_n_dot_2_8.txt", unpack=True)
theta_1_28 = bogenmin_zu_rad(u_theta_1_28)
theta_2_28 = bogenmin_zu_rad(u_theta_2_28)
d_28 = 1.296*10**-3 # meter
N_28 = 2.8*10**18*10**6 #1/m^3
####################################################
####################################################


####################################################
#Magnetfeld Untersuchung
####################################################

plt.plot(z, B, marker = "x", color = "cornflowerblue", label = "Messwerte", lw = 0)
plt.plot(z, B, color = "gray",lw = 0.5)
plt.plot(98,430,marker = o, color="firebrick",label="Maximum = \qty{430}{\milli\tesla}")
plt.grid()
plt.xlabel(r"$z \mathbin{/} \unit{\milli\metre}$")
plt.ylabel(r"$B \mathbin{/} \unit{\milli\tesla}$")
plt.ylim(0, 440)
plt.xlim(70, 135)
plt.legend()
plt.tight_layout()

plt.savefig('build/magnetfeld.pdf')
plt.close()
####################################################
####################################################

