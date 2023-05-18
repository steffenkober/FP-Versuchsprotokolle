import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp 
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as devs
import scipy.constants as const
import scipy.optimize as op

#daten einlesen
phi, U_max, U_min = np.genfromtxt("content/messdaten_kontrast.txt", unpack=True)
Versuchsnummer, N_Glas = np.genfromtxt("content/messdaten_glas.txt", unpack=True)
p, N_luft_1, N_luft_2, N_luft_3 = np.genfromtxt("content/messdaten_luft.txt", unpack=True)
n_glas = np.ones(len(N_Glas))

#definition der nötigen funktionen
def kontrast(I_min,I_max):
              return (I_max - I_min)/(I_max+I_min)
def kontrast_theorie(Phi,a):
              return a*np.abs(np.cos(Phi*np.pi/180)*np.sin(Phi*np.pi/180))


#fit
kontrast_exp = kontrast(U_min,U_max)

params,pcov = op.curve_fit(kontrast_theorie,phi,kontrast_exp)
print("kontrast parameter a = ",params[0])
print("maximum des kontrastes:",kontrast_exp[4:6])
print("winkel des maximums:",phi[4:6])


lambda_vac = 632.990*10**(-9)
Theta0 =10*(2*np.pi/360)
Theta = 10*(2*np.pi/360)
L = 1 *10**(-3)

for i in range(len(N_Glas)):
    n_glas[i] = 1/(1-(N_Glas[i] * lambda_vac)/(2*L*Theta*Theta0))

n_glas_std = np.std(n_glas)/np.sqrt(np.size(n_glas))
print(np.mean(n_glas),n_glas_std)
print(n_glas)
print(params)


plt.figure(figsize=(6,5))

plt.plot(p, N_luft_1,"xy", label="Versuchreihe 1")
plt.plot(p, N_luft_2,"xr", label="Versuchreihe 2")
plt.plot(p, N_luft_3,"xg", label="Versuchreihe 3")


plt.xlabel(r"$p \, / \, \mathrm{mbar}$")
plt.ylabel('M')
plt.legend()
plt.grid()
#plt.show()
plt.savefig("build/MaxN.pdf")
plt.close()
L1 = 0.1

def n_gas1(M):
    return ((lambda_vac * M)/L1) + 1

n_1 = n_gas1(N_luft_1)
n_2 = n_gas1(N_luft_2)
n_3 = n_gas1(N_luft_3)

print(n_1)
print(n_2)
print(n_3)

n_std = (np.std(n_1)+np.std(n_2)+np.std(n_3))/np.sqrt(np.size(n_1))

print((np.mean(n_1)+np.mean(n_2)+np.mean(n_3))/3, n_std)

plt.figure(figsize=(6,5))

plt.plot(p, n_1,"xy", label="Versuchreihe 1")
plt.plot(p, n_2,"xr", label="Versuchreihe 2")
plt.plot(p, n_3,"xg", label="Versuchreihe 3")


plt.xlabel(r"$p \, / \, \mathrm{mbar}$")
plt.ylabel('n')
plt.legend()
plt.grid()
#plt.show()
plt.savefig("build/ngas.pdf")
plt.close()

T_1 = 22 + 273.15 #Kelvin
T_0= 15 + 273.15 #Kelvin
p_0 = 1013
R = 8.314

def n_luft(p,a,b):
    return (a * p)/(R * T_1) + b

param1, cov1 = op.curve_fit(n_luft, p, n_1)
uncertainties1 = np.sqrt(np.diag(cov1))
a1 = ufloat(param1[0], uncertainties1[0])
b1 = ufloat(param1[1], uncertainties1[1])
print(a1)
print(b1)

param2, cov2 = op.curve_fit(n_luft, p, unp.nominal_values(n_2))
uncertainties2 = np.sqrt(np.diag(cov2))
a2 = ufloat(param2[0], uncertainties2[0])
b2 = ufloat(param2[1], uncertainties2[1])
print(a2)
print(b2)

param3, cov3 = op.curve_fit(n_luft, p, unp.nominal_values(n_3))
uncertainties3 = np.sqrt(np.diag(cov3))
a3 = ufloat(param3[0], uncertainties3[0])
b3 = ufloat(param3[1], uncertainties3[1])
print(a3)
print(b3)

a_std = (uncertainties1[0]+uncertainties2[0]+uncertainties3[0])/np.sqrt(3)
b_std = (uncertainties1[1]+uncertainties2[1]+uncertainties3[1])/np.sqrt(3)

print((a1+a2+a3)/3, a_std)
print((b1+b2+b3)/3, b_std)

p_lin = np.linspace(0, p.max(),960)
plt.figure(figsize=(6,5), dpi=200)

plt.errorbar(p, unp.nominal_values(n_1), yerr=unp.std_devs(n_1),  fmt='x', label='Versuchreihe 1')
plt.plot(p_lin, n_luft(p_lin, *param1), '-', label='Fit für Versuchreihe 1')
plt.errorbar(p, unp.nominal_values(n_2), yerr=unp.std_devs(n_2),fmt='x', label='Versuchreihe 2')
plt.plot(p_lin, n_luft(p_lin, *param2),'-', label='Fit für Versuchreihe 2')
plt.errorbar(p, unp.nominal_values(n_3), yerr=unp.std_devs(n_3),fmt='x', label='Versuchreihe 3')
plt.plot(p_lin, n_luft(p_lin, *param3),'-',  label='Fit für Versuchreihe 3')

plt.xlabel(r"$p \, / \, \mathrm{mbar}$")
plt.ylabel('n')
plt.legend()
plt.savefig("build/Fits.pdf")
#plt.show()
plt.close()
def n_luft1(a,b):
    return (a * p_0)/(R * T_0) + b

print(n_luft1(a1,b1))
print(n_luft1(a2,b2))
print(n_luft1(a3,b3))

print((n_luft1(a1,b1) + n_luft1(a2,b2) + n_luft1(a3,b3))/3)

#plotting
#kontrast
x_phi = np.linspace(0,180,10000)
plt.plot(x_phi,kontrast_theorie(x_phi,params[0]), color = "cornflowerblue", label = "Theoriewerte")
plt.plot(phi,kontrast_exp, color = "firebrick",label="Messdaten",marker="x",linewidth="0")
plt.legend()
plt.grid()
plt.ylabel(r'$\mathrm{Kontrast}$')
plt.xlabel(r'$\phi\,\, \mathrm{in} \,\,^\circ$')

#plt.show()
plt.savefig("build/Kontrast.pdf")
plt.close()
