import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const


#einlesen
f, I_h1, I_h2, I_s1, I_s2 = np.genfromtxt("content/Messdaten/Messdaten.txt",unpack=True)

f = f*10**3
I_h1 = I_h1*2 #A
I_h2 = I_h2*2 #A
I_s1 = I_s1*0.1 #A
I_s2 = I_s2*0.1 #A

## funktionen 
def helmholtz(I, N, R):
              return const.mu_0 * (8*I*N)/(np.sqrt(125)*R)

def quadratic(x,a,b,c):
              return a*x**2+b*x+c

def linear(x,a,b):
              return a*x+b
def lande(x):
              return const.h/(x*const.value('Bohr magneton'))

def gj(j,s,l):
              return 1+(j*(j+1)+s*(s+1)-l*(l+1))/(2*j*(j+1))

def gf(f,j,i,g_j):
              return g_j*(f*(f+1)+j*(j+1)-i*(i+1))/(2*f*(f+1))
### konstanten

#sweep
R_s = 16.39*10**(-2) #m
N_s = 11
#
#horizontal
R_h = 15.79*10**(-2)#m
N_h = 154
#vertikal
R_v = 11.735*10**(-2)#m
N_v = 20

I_v = 2.28*0.1#A

J = 0.5
S = 0.5
L = 0

###

B_1 = helmholtz(I_h1,N_h,R_h) + helmholtz(I_s1,N_s,R_s) #T
B_2 = helmholtz(I_h2,N_h,R_h) + helmholtz(I_s2,N_s,R_s) #T

B_erde = helmholtz(I_v,N_v,R_v)
print("vertikal Komponenete des Magnetfeldes der Erde: ",B_erde)

params1, pcov1 = op.curve_fit(linear, f, B_1)
err1 = np.sqrt(np.diag(pcov1))
a1 = ufloat(params1[0], err1[0])
b1 = ufloat(params1[1], err1[1])

params2, pcov2 = op.curve_fit(linear, f, B_2)
err2 = np.sqrt(np.diag(pcov2))
a2 = ufloat(params2[0], err2[0])
b2 = ufloat(params2[1], err2[1])

print("------------------------------------------------------")
print(f"a1 = {a1:.3e}")
print(f"b1 = {b1:.3e}", "\n")
print(f"a2 = {a2:.3e}")
print(f"b2 = {b2:.3e}")
print("------------------------------------------------------")

xx = np.linspace(100000, 10**6, 10000)

plt.plot(xx/10**6, linear(xx, *params1)*10**6, label = "Regression", c="darkslategrey")
plt.plot(xx/10**6, linear(xx, *params2)*10**6, label = "Regression",c="indigo")

plt.plot(f/10**6, B_1*10**6, lw = 0, c = "lightseagreen", marker = "x", label = r"Messdaten $\ce{^{85}Rb}$", ms = 6)
plt.plot(f/10**6, B_2*10**6, lw = 0, c = "mediumorchid", marker = "1", label = r"Messdaten $\ce{^{87}Rb}$", ms = 9)

plt.grid()
plt.ylabel(r"Magnetfeld$\,\mathrm{/}\,\unit{\micro\tesla}$")
plt.xlabel(r"Frequenz$\,\mathrm{/}\,\unit{\mega\hertz}$")
#plt.legend(fontsize="8")
plt.tight_layout()

plt.savefig('build/regression.pdf')
plt.close()

### lande faktoren

g_F1 = lande(a1)
g_F2 = lande(a2)
print("------------Lande-Faktoren----------------------------")
print(f"g_F1 = {g_F1}")
print(f"g_F2 = {g_F2}")
print("------------------------------------------------------")

g_j = gj(J,S,L)

I1 = g_j / (4 * g_F1) - 1 + unp.sqrt((g_j / (4 * g_F1) - 1)**2+ 3 * g_j / (4 * g_F1) - 3 / 4)
I2 = g_j / (4 * g_F2) - 1 + unp.sqrt((g_j / (4 * g_F2) - 1)**2+ 3 * g_j / (4 * g_F2) - 3 / 4)

print(f"Erster Kernspin: I1={I1}")
print(f"Zweiter Kernspin: I2={I2}")
# Abweichungen
I_delta1 = abs(I1 -1.5)/1.5*100
I_delta2 = abs(I2 -2.5)/2.5*100
print(I_delta1,I_delta2)

print("Isotopenverhältnis :",6.2/12.6)
