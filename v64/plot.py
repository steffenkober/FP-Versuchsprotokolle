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

#definition der nötigen funktionen
def kontrast(I_min,I_max):
              return (I_max - I_min)/(I_max+I_min)
def kontrast_theorie(Phi,a):
              return a*np.abs(np.cos(Phi*np.pi/180)*np.sin(Phi*np.pi/180))


#fit
kontrast_exp = kontrast(U_min,U_max)

params,pcov = op.curve_fit(kontrast_theorie,phi,kontrast_exp)

print(params)



#plotting
#kontrast
x_phi = np.linspace(0,180,10000)
plt.plot(x_phi,kontrast_theorie(x_phi,params[0]), color = "cornflowerblue", label = "Theoriewerte")
plt.plot(phi,kontrast_exp, color = "firebrick",label="Messdaten")
plt.show()
#plt.savefig("build/temp.pdf")

#x = np.linspace(0, 9, 1000)
#y = x ** np.sin(x)
#
#plt.subplot(1, 2, 1)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')
#
#plt.subplot(1, 2, 2)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
#plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
#plt.legend(loc='best')
#
## in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('build/plot.pdf')
