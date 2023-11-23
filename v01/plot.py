import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as devs
import scipy.optimize as op
from uncertainties import ufloat


#Auflösungzeit
t, N, t_M = np.genfromtxt("content/data/T_VZ.txt", unpack = True)

N = unp.uarray(N, np.sqrt(N))/t_M

temp = np.array([7.65,7.95,7.3,6.55,7.4,6.8,7.4])
plateau = np.mean(temp)


plt.errorbar(t, noms(N), yerr = devs(N), lw = 0, elinewidth = 1, marker = ".", capsize = 1, label = "data", c = "firebrick")
plt.hlines(plateau, -1, 5, colors = "cornflowerblue", label = "plateau", lw = 1.5)
plt.hlines(plateau/2, -4.5, 7.5, colors = "cornflowerblue", label = "FWHM", lw = 1.5, ls = "dashed")
plt.vlines([-4.5, 7.5], 0, plateau, colors = "cornflowerblue", lw = 1, ls = "dashed")

plt.grid()
plt.legend()
plt.xlabel(r"$T_\text{VZ} \mathbin{/} \unit{\nano\second}$")
plt.ylabel(r"$\text{Countrate} \mathbin{/} \unit{\per\second}$")
plt.xlim(-15, 15)
plt.tight_layout()
plt.savefig('build/plot1.pdf')
plt.close()

#MCA

def f(x,a,b):
    return a*x+b

t_mca, c = np.genfromtxt("content/data/MCA.txt", unpack = True)

#fit
params, pcov = op.curve_fit(f, c, t_mca)
err = np.sqrt(np.diag(pcov))
print("Parameter VKA Kalibrierung: \n", params, err)

x = np.linspace(0, 450, 1000)
plt.plot(c, t_mca, marker = "x", lw = 0, c = "black", ms = 8, label = "data")
plt.plot(x, f(x, *params), c = "firebrick", lw = 0.7, label = "linear fit")
plt.plot([], [], " ", label = f"a = {params[0]:.2e} $\pm$ {err[0]:.2e}")
plt.plot([], [], " ", label = f"b = {params[1]:.2e} $\pm$ {err[1]:.2e}")

plt.xlabel("Channel Number")
plt.ylabel(r'$t \mathbin{/} \unit{\micro\second}$')
plt.legend()
plt.grid()
plt.xlim(0, 450)
plt.ylim(0, 10)
plt.tight_layout()
plt.savefig('build/plot2.pdf')
plt.close()

#lifetime of cosmic muons
counts = np.genfromtxt("content/data/lifetime.txt")#, unpack = True
channel = np.array(range(len(counts)))
time = f(channel, *unp.uarray(params, err))

def exp(t, N_0, tau, U_0):
    return N_0* np.exp(-t/tau) + U_0

mask = [(counts > 0) & (counts < 500)]
#print(mask[0])
params, pcov = op.curve_fit(exp, noms(time[mask[0]]), counts[mask[0]], maxfev=5000)#, p0 = [511, 2.3,3]
err = np.sqrt(np.diag(pcov))

print("Parameter des Fits: \n", f"N_0:  {params[0]} +- {err[0]} \n tau:  {params[1]} +- {err[1]} \n U_0:  {params[2]} +- {err[2]}")


## versuch über linearen fit
#mask1 = [(counts > 0) & (counts < 600)]
#params, pcov = op.curve_fit(f, noms(time[mask1[0]]), counts[mask1[0]])#, p0 = [500, 1*10**(-6),1]
#err = np.sqrt(np.diag(pcov))
#print(params)


# Plotting

x = np.linspace(0, 13, 1000)

plt.errorbar(noms(time), counts, xerr=devs(time), yerr=np.sqrt(counts), lw=0, elinewidth=1, capsize=2, c = "black", label = "Messdaten", alpha = .6)
plt.plot(x, exp(x, *params), label = "Fit", c = "firebrick")

plt.yscale("log")
plt.ylim(1, 1e2)
plt.xlim(0, 11)
plt.ylabel(r"$N$")
plt.xlabel(r'$t \mathbin{/} \unit{\micro\second}$')
plt.legend()
plt.grid()
#plt.show()
plt.tight_layout()
plt.savefig("build/fit_log.pdf")
plt.close()

plt.errorbar(noms(time), counts, xerr=devs(time), yerr=np.sqrt(counts), lw=0, elinewidth=1, capsize=2, c = "black", label = "Messdaten", alpha = .6)
plt.plot(x, exp(x, *params), label = "Fit", c = "firebrick")

plt.ylim(0, 50)
plt.xlim(0, 11)
plt.ylabel(r"$N$")
plt.xlabel(r'$t \mathbin{/} \unit{\micro\second}$')
plt.legend()
plt.grid()
#plt.show()
plt.tight_layout()
plt.savefig("build/fit.pdf")
plt.close()

#deviations
lifetime_lit = ufloat(2.1969811, 0.0000022)
lifetime = ufloat(params[1], err[1])
print(lifetime,lifetime_lit)
dev = abs(lifetime-lifetime_lit)/lifetime_lit
print(dev)
