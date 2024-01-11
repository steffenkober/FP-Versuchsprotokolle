import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as devs
from uncertainties import ufloat
import scipy.optimize as op

# detector scan
theta, counts = np.genfromtxt("content/Data/detector-scan.UXD", unpack = True)

def gaussian(x,x_0,sigma,I_0,const):
    return I_0/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-x_0)**2/(2*sigma**2)) + const

bounds = [[-0.5, 0, 0, -0.001], [0.5, 0.5, 3e6, 1e4]]
params, pcov = op.curve_fit(gaussian, theta, counts, p0 = [np.mean(theta), np.std(theta), 175e3, 0], bounds=bounds)
err = np.sqrt(np.diag(pcov))
FWHM = 0.08623+0.07639
I0 = params[2]
sigma = ufloat(params[1], err[1])
print("-------------------------------------------------------")
print("Parameter der Gaußfunktion")
print(f"t0      : {params[0]:.4e} +- {err[0]:.4e}")
print(f"sigma   :  {params[1]:.4e} +- {err[1]:.4e}")
print(f"I0      :  {params[2]:.4e} +- {err[2]:.4e}")
print(f"B       :  {params[3]:.4e} +- {err[3]:.4e}")
print(f"FWHM    :  {2*unp.sqrt(2*np.log(2))*sigma}")
print("-------------------------------------------------------")

x = np.linspace(-0.5, 0.5, 1000)
plt.errorbar(theta, counts, yerr= np.sqrt(counts), marker = "x", color = "saddlebrown", lw = 0, elinewidth = 1, capsize= 1, label = "Messdaten", alpha = 1, ms = 5)
plt.plot(x, gaussian(x, *params), label = "Fit", c = "forestgreen", lw = 1)
plt.hlines(params[2]/(2*np.sqrt(2*np.pi*params[1]**2)) - params[3], xmin = -0.5, xmax = 0.5, color = "gray", ls = "dashed",lw=0.7, label = r"$\frac{1}{2} \;I_0$")
plt.vlines([params[0] - 1/2*FWHM, params[0] + 1/2*FWHM] , 0, 3e6, color = "cornflowerblue",lw=0.7, label = f"FWHM: {2*unp.sqrt(2*np.log(2))*sigma:.3f}°")
plt.legend()
plt.xlim(-0.4, 0.4)
plt.ylim(0, 3e6)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
plt.xlabel(r"$\Theta \mathbin{/} \unit{\degree}$")
plt.ylabel(r"$I \mathbin{/} \text{Hits}  \mathbin{/} \unit{\second}$")
plt.grid()
plt.tight_layout()
#plt.show()

plt.savefig("build/DScan.pdf")
plt.close()
#detector scan end

#Z-Scan
z, counts = np.genfromtxt("content/Data/Z_2b.UXD", unpack = True)

plt.errorbar(z, counts, yerr= np.sqrt(counts), marker = "x", color = "saddlebrown", lw = 0, elinewidth = 1, capsize= 1, label = "Messdaten", ms = 5)
#plt.errorbar(z2, counts2, yerr= np.sqrt(counts2), marker = "+", color = "black", lw = 0, elinewidth = 1, capsize= 1, label = "2. Scan", alpha = .7, ms = 5)
plt.vlines([z[15], z[33]], ymin=0, ymax= 3e6, ls = "dashed", color = "cornflowerblue",lw=0.7, label = f"Strahlbreite: {z[33]- z[15]} mm")
plt.xlim(-0.5, 0.5)
plt.ylim(0, 3e6)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
plt.xlabel(r"$z \mathbin{/} \unit{\milli\metre}$")
plt.ylabel(r"$I \mathbin{/} \text{Hits}  \mathbin{/} \unit{\second}$")
plt.grid()
plt.legend()
plt.tight_layout()
#plt.show()

plt.savefig("build/ZScan.pdf")
plt.close()

d0 = z[33]- z[15]
print("-------------------------------------------------------")
print("Strahlbreite")
print(f"d0 = {d0} mm")
print("-------------------------------------------------------")
#Z-Scan end
#rocking-cuve scan
theta, counts = np.genfromtxt("content/Data/rocking_1b2.UXD", unpack = True)

plt.errorbar(theta, counts, yerr= np.sqrt(counts), marker = "x", color = "saddlebrown", lw = 0, elinewidth = 1, capsize= 1, label = "Messdaten", ms = 5)
plt.vlines([theta[1], theta[25]], ymin=0, ymax= 1.4e6, ls = "dashed", color = "cornflowerblue",lw=0.7, label = f"Geometriewinkel: {theta[25]- theta[1]} °")
plt.legend()
plt.xlim(-1, 1)
plt.ylim(0, 1.4e6)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
plt.ylabel(r"$I \mathbin{/} \text{Hits}  \mathbin{/} \unit{\second}$")
plt.grid()
plt.tight_layout()
#plt.show()

plt.savefig("build/RockingScan.pdf")
plt.close()
geo_theorie = np.arcsin(d0/20) # rad
geo_exp = (theta[25]- theta[1]) # °
print("-------------------------------------------------------")
print("Geometriewinkel")
print(f"Experiment: {geo_exp} ° = {geo_exp* np.pi/180:.4f} rad")
print(f"Theorie:    {geo_theorie*180/np.pi:.4f} ° = {geo_theorie:.4f} rad")
print("-------------------------------------------------------")
#rocking end
# Reflektivitätsscan

t, c1 = np.genfromtxt("content/Data/reflectivity.UXD", unpack = True)
t2, c2 = np.genfromtxt("content/Data/diffuse.UXD", unpack = True)

plt.plot(t, c1, color = "forestgreen", label = "Reflektivität (unkorrigiert)")
plt.plot(t2, c2, color = "saddlebrown", label = "Streuintensität")
plt.plot(t, c1-c2, label = "Differenz", ls = "dashed")

plt.legend()
plt.yscale("log")
plt.xlim(0, 2.5)
plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
plt.ylabel(r"$I \mathbin{/} \text{Hits}  \mathbin{/} \unit{\second}$")
plt.tight_layout()
#plt.show()

plt.savefig("build/Reflek1.pdf")
plt.close()
#ref end
########################################################################

a_c = 0.223 # ° (kritischer Winkel von Silizium)

def r(alpha):
    return (a_c/(2*alpha))**4

# Von nun an mit Reflektivität:
R = (c1 - c2)/(5*I0)
# Korrektur mit G-Faktor 
R_c = np.array(R)
R_c[(t < geo_exp) & (t > 0)] = R[(t < geo_exp) & (t > 0)] * np.sin(np.deg2rad(geo_exp))/np.sin(np.deg2rad(t[(t < geo_exp) & (t > 0)]))

# Berechnung der Schichtdicke aus Oszillationen
idx_temp = []
for i in range(len(t)):
    if t[i] > 0.2 and t[i] < 0.9 and R_c[i] < R_c[i-1] and R_c[i] < R_c[i+1] and R_c[i] < R_c[i-2]  and R_c[i] < R_c[i+2]:
        idx_temp.append(i)
idx = [idx_temp[0],idx_temp[1],idx_temp[1]+10,idx_temp[1]+20,idx_temp[2],idx_temp[3],idx_temp[4],idx_temp[5]]
diffs = np.diff(t[idx])
lam = 1.54e-10 # Wellenlänge der Strahlung (K_alpha Linie Kupfer)

a_d = ufloat(np.mean(diffs), np.std(diffs))
d = lam/(2*a_d*np.pi/180)
print("-------------------------------------------------------")
print("Oszillationsabstand")
print(f"Delta alpha  = {a_d:.4e} °")
print(f"Schichtdicke = {d:.4e} m")
print("-------------------------------------------------------")

x = np.linspace(a_c, 2.5, 1000)
x2 = np.linspace(0, a_c, 2)
plt.plot(t, R, label = "Reflektivität (ohne G-Korrektur)",c="limegreen")
plt.plot(t, R_c, label = "Reflektivität (korrigiert)", c = "forestgreen")
plt.plot(x, r(x), label = "Fresnelreflektivität", color = "black")
plt.plot(x2, [r(a_c), r(a_c)], color = "black")
plt.plot(t[idx], R_c[idx], lw = 0, marker = "+", label = "Minima", c = "firebrick", ms = 5)
plt.vlines(a_c, 0, 10e3, label = r"$\alpha_c$ (Silizium)", color = "deeppink")
#plt.vlines(0.153, 0, 10e3, label = r"$\alpha_c$ (Polysterol)", color = "rebeccapurple")
plt.legend()
plt.yscale("log")
plt.xlim(0, 2.5)
plt.ylim(None, 10e3)
plt.xlabel(r"$\alpha \mathbin{/} \unit{\degree}$")
plt.ylabel(r"$R$")
plt.tight_layout()
#plt.show()
plt.savefig("build/Reflek2.pdf")
plt.close()
########################################################################
# Parratt-Algorithmus

lam = 1.54e-10
k = 2*np.pi/lam
n1 = 1
d1 = 0

delta_Poly = 3.5e-6 # 1. Schicht Polysterol
delta_Si = 7.6e-6 # 2. Schicht Silizium
b_Poly = delta_Poly/40
b_Si = delta_Si/40
d_ = noms(d)
sigma_Poly = 4.2e-10
sigma_Si = 1e-10

#delta_Poly = 4.2e-6 # 1. Schicht Polysterol
#delta_Si = 1.6e-5 # 2. Schicht Silizium
#b_Poly = 2.7e-8
#b_Si = 9.8e-7
#d_ = 8.2e-8
#sigma_Poly = 4e-10
#sigma_Si = 3e-10

params = [delta_Poly, delta_Si, b_Poly, b_Si, d_, sigma_Poly, sigma_Si] # Startwerte
err = np.zeros(len(params))

def parratt(a, delta2, delta3, b2, b3, d2, sigma1, sigma2):
    n2 = 1.0 - delta2 - b2*1j
    n3 = 1.0 - delta3 - b3*1j
    a = np.deg2rad(a)
    kd1 = k *  np.sqrt(n1**2 - np.cos(a)**2)
    kd2 = k * np.sqrt(n2**2 - np.cos(a)**2)
    kd3 = k * np.sqrt(n3**2 - np.cos(a)**2)

    r12 = ((kd1 - kd2)/(kd1 + kd2))*np.exp(-2*kd1*kd2*sigma1**2)
    r23 = ((kd2 - kd3)/(kd2 + kd3))*np.exp(-2*kd2*kd3*sigma2**2)

    x2 = np.exp(-2j* kd2 * d2) * r23
    x1 = (r12 + x2)/(1+ r12*x2)

    return np.abs(x1)**2


t_min = 0.35
t_max = 0.75

bounds = ([1e-6, 1e-6, 1e-9, 1e-8, 1e-9, 5e-11, 5e-11], [5e-6, 5e-5, 1e-7, 1e-6, 1e-7, 1e-9, 1e-9]) # Limits der Parameter
params, pcov = op.curve_fit(parratt, t[(t>t_min) * (t<t_max)], R_c[(t>t_min) * (t<t_max)], p0 = params, bounds = bounds,maxfev=10000)#
err = np.sqrt(np.diag(pcov))

delta_Si = ufloat(params[0], err[0])
delta_Poly = ufloat(params[1], err[1])
a_c_Poly = unp.sqrt(2*delta_Poly)*180/np.pi
a_c_Si = unp.sqrt(2*delta_Si)*180/np.pi
print("-------------------------------------------------------")
print("Parameter des Parrattalgorithmus")
print(f"delta_Poly  : {params[0]:.4e} +- {err[0]:.4e}")
print(f"delta_Si    : {params[1]:.4e} +- {err[1]:.4e}")
print(f"b_Poly      : {params[2]:.4e} +- {err[2]:.4e}")
print(f"b_Si        : {params[3]:.4e} +- {err[3]:.4e}")
print(f"d2          : {params[4]:.4e} +- {err[4]:.4e} m")
print(f"sigma_Poly  : {params[5]:.4e} +- {err[5]:.4e}")
print(f"sigma_Si    : {params[6]:.4e} +- {err[6]:.4e}")
print(f"alpha_c (Poly)  : {a_c_Poly:.4f} °")
print(f"alpha_c (Si)    : {a_c_Si:.4f} °")
print("-------------------------------------------------------")

x = np.linspace(0, 2.5, 1000)

plt.plot(t, R_c, label = "gemessene Reflektivität (korrigiert)", c = "cornflowerblue")
plt.plot(x, parratt(x, *params), color = "firebrick", alpha = .8, label = "Parrattalgorithmus")
plt.legend()
plt.yscale("log")
plt.tight_layout()
plt.savefig("build/Reflek3.pdf")
plt.close()

##Startwerte
#lam = 1.54e-10
#k=2*np.pi/lam
#n1=1
#d1=0
##Algorithmus
#def Parrattalgorithmus(alpha, delta2, delta3, kappa2, kappa3, sigma1, sigma2, d2):
#    alpha = np.deg2rad(aalpha)
#    n2 = 1.0 - delta2 - kappa2*1j
#    n3 = 1.0 - delta3 - kappa3*1j
#    
#    kd1 = k *  np.sqrt(n1**2 - np.cos(alpha)**2)
#    kd2 = k * np.sqrt(n2**2 - np.cos(alpha)**2)
#    kd3 = k * np.sqrt(n3**2 - np.cos(alpha)**2)
#
#    r12 = ((kd1 - kd2)/(kd1 + kd2))*np.exp(-2*kd1*kd2*sigma1**2)
#    r23 = ((kd2 - kd3)/(kd2 + kd3))*np.exp(-2*kd2*kd3*sigma2**2)
#
#    x2 = np.exp(-2j* kd2 * d2) * r23
#    x1 = (r12 + x2)/(1+ r12*x2)
#
#    return np.abs(x1)**2