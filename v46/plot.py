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
              return (np.floor(zahl)+(zahl - np.floor(zahl))/60)*np.pi/180

def quadratic(x,a,b,c):
              return a*x**2+b*x+c

def linear(x,a,b):
              return a*x+b

def theta(t_1,t_2):
              return abs(t_1-t_2)/2


####################################################
#Einlesen der Messdaten
####################################################
z, B = np.genfromtxt("content/Messdaten_B_Feld.txt", unpack=True)
lambdas, u_theta_1_rein, u_theta_2_rein = np.genfromtxt("content/Messdaten_GaAs_hochrein.txt", unpack=True)
theta_1_rein = bogenmin_zu_rad(u_theta_1_rein)
theta_2_rein = bogenmin_zu_rad(u_theta_2_rein)
print(theta_1_rein,theta_2_rein)
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
n = 3.354
B_max = 430*10**(-3)
lambdas = lambdas*10**(-6)

####################################################
#Magnetfeld Untersuchung
####################################################

plt.plot(z, B, marker = "x", color = "cornflowerblue", label = "Messwerte", lw = 0)
#plt.plot(z, B, color = "gray",lw = 0.5)
plt.plot(98,430,marker = "o", color="firebrick",label=r"Maximum $ = \qty{430}{\milli\tesla}$")
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
#theta berechnen
####################################################

theta_rein = theta(theta_1_rein,theta_2_rein)
theta_12 = theta(theta_1_12,theta_2_12)
theta_28 = theta(theta_1_28,theta_2_28)
np.savetxt(
    "content/differenzen.txt", np.array([theta_12, theta_1_12, theta_2_12,  theta_28, theta_1_28, theta_2_28, theta_rein, theta_1_rein, theta_2_rein]).transpose(),
     header = "GaAs12 (theta, theta1, theta2),   GaAs28 (theta, theta1, theta2),    GaAs (rein) (theta, theta1, theta2)", fmt = "%.2f"
)

theta_rein = theta(theta_1_rein,theta_2_rein)/d_rein
theta_12 = theta(theta_1_12,theta_2_12)/d_12
theta_28 = theta(theta_1_28,theta_2_28)/d_28

print(theta_12)
####################################################
#effektive masse
####################################################

params1, pcov1 = op.curve_fit(linear, lambdas[0:7]**2*10**12, theta_12[0:7]-theta_rein[0:7])
err1 = np.sqrt(np.diag(pcov1))
a1 = ufloat(params1[0], err1[0])*10**(12)
b1 = ufloat(params1[1], err1[1])

params2, pcov2 = op.curve_fit(linear, lambdas[0:7]**2*10**12,theta_28[0:7]-theta_rein[0:7])
err2 = np.sqrt(np.diag(pcov2))
a2 = ufloat(params2[0], err2[0])*10**(12)
b2 = ufloat(params2[1], err2[1])

print("------------------------------------------------------")
print(f"a12 = {a1:.3e} m^-3")
print(f"b12 = {b1:.3e} m^-1", "\n")
print(f"a28 = {a2:.3e} m^-3")
print(f"b28 = {b2:.3e} m^-1")
print("------------------------------------------------------")

m1 = unp.sqrt(const.e**3*N_12*B_max/(8*const.pi**2*const.epsilon_0*const.c**3*n*a1))
m2 = unp.sqrt(const.e**3*N_28*B_max/(8*const.pi**2*const.epsilon_0*const.c**3*n*a2))
m = (m1 + m2)/2
print(m1,m2,m)
m_lit = 0.063*const.m_e

# Abweichungen
delta1 = (m1 -m_lit)/m_lit*100
delta2 = (m2 -m_lit)/m_lit*100
delta3 = (m -m_lit)/m_lit*100
print(delta1,delta2,delta3)
# Plot

xx = np.linspace(0, 8, 1000)

plt.plot(xx, linear(xx, *params1), label = "Regression", c="darkslategrey")
plt.plot(xx, linear(xx, *params2), label = "Regression",c="indigo")

plt.plot(lambdas[0:7]**2*10**12, theta_12[0:7]-theta_rein[0:7], lw = 0, c = "lightseagreen", marker = "x", label = r"GaAs n-dotiert, N = $\qty{1.2e18}{\per\cubic\centi\metre}$", ms = 6)
plt.plot(lambdas[0:7]**2*10**12, theta_28[0:7]-theta_rein[0:7], lw = 0, c = "mediumorchid", marker = "1", label = r"GaAs n-dotiert, N = $\qty{2.8e18}{\per\cubic\centi\metre}$", ms = 9)

plt.plot(lambdas[7:9]**2*10**12,theta_12[7:9]-theta_rein[7:9], lw = 0, c = "maroon", marker = "x", label = "Rausgenommen", ms = 6)
plt.plot(lambdas[7:9]**2*10**12,theta_28[7:9]-theta_rein[7:9], lw = 0, c = "maroon", marker = "1", label = "Rausgenommen", ms = 9)

plt.grid()
#plt.ylim(0, 115)
#plt.xlim(1, 7.5)
plt.ylabel(r"$\theta_\text{frei} \mathbin{/} \unit{\radian\per\metre}$")
plt.xlabel(r"$\lambda^2 \mathbin{/} \unit{\micro\metre\squared}$")
plt.legend(fontsize="8")
plt.tight_layout()

plt.savefig('build/effektive_masse_fit.pdf')
plt.close()
