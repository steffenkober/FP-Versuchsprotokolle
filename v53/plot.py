import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp 
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as devs
import scipy.constants as const
import scipy.optimize as op

#ein kästchen entspricht einer Spannung von 100 mV
kaestchen = 0.1 #volt

U_l, U_max, U_r, A_max, f_l, f_max, f_r = np.genfromtxt("messdaten_a.txt", unpack = True)

def quadratic(x,a,b,c):
              return a*x**2+b*x+c


params, pcov = op.curve_fit(quadratic, np.array([U_r[0],U_max[0],U_l[0]]), kaestchen*np.array([A_max[0]/2,A_max[0],A_max[0]/2]))
params1, pcov1 = op.curve_fit(quadratic, np.array([U_l[1],U_max[1],U_r[1]]), kaestchen*np.array([A_max[1]/2,A_max[1],A_max[1]/2]))
params2, pcov2 = op.curve_fit(quadratic, np.array([U_l[2],U_max[2],U_r[2]]), kaestchen*np.array([A_max[2]/2,A_max[2],A_max[2]/2]))

x = np.linspace(50,240,1000)
fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].plot(x,quadratic(x,params[0],params[1],params[2]),label="1. Mode Regression")
axs[0].plot(x,quadratic(x,params1[0],params1[1],params1[2]),label="2. Mode Regression")
axs[0].plot(x,quadratic(x,params2[0],params2[1],params2[2]),label="3. Mode Regression")
axs[0].plot(U_l,kaestchen*A_max/2,linewidth = 0,marker = "x",color="black",label="Messdaten")
axs[0].plot(U_max,kaestchen*A_max,linewidth = 0,marker = "x",color="black")
axs[0].plot(U_r,kaestchen*A_max/2,linewidth = 0,marker = "x",color="black")
axs[0].set_xlim(50,230)
axs[0].set_ylim(0,0.5)
axs[0].grid()
axs[0].set_xlabel(r"Reflector voltage $U/\mathrm{V}$",fontsize="6")
axs[0].set_ylabel(r"Output power in $U_{out}/\mathrm{V}$",fontsize="6")
#axs[0].tight_layout()
axs[0].legend(fontsize="4", loc ="best")
#plt.savefig("content/messung1.pdf")
#plt.close()

maxima1 = np.amax(quadratic(x,params[0],params[1],params[2]))
maxima2 = np.amax(quadratic(x,params1[0],params1[1],params1[2]))
maxima3 = np.amax(quadratic(x,params2[0],params2[1],params2[2]))


def cubic1(x,a,b,c):
              return a*(x-U_max[0])**3+b*(x-U_max[0])**2+c*(x-U_max[0])
def cubic2(x,a,b,c):
              return a*(x-U_max[1])**3+b*(x-U_max[1])**2+c*(x-U_max[1])
def cubic3(x,a,b,c):
              return a*(x-U_max[2])**3+b*(x-U_max[2])**2+c*(x-U_max[2])

params3, pcov3 = op.curve_fit(cubic1, np.array([U_r[0],U_max[0],U_l[0]]), np.array([f_r[0]-f_max[0],0,f_l[0]-f_max[0]]),p0=[0.000002,0,0.001])
params4, pcov4 = op.curve_fit(cubic2, np.array([U_l[1],U_max[1],U_r[1]]), np.array([f_l[1]-f_max[1],0,f_r[1]-f_max[1]]),p0=[0.000002,0,0.0015])
params5, pcov5 = op.curve_fit(cubic3, np.array([U_l[2],U_max[2],U_r[2]]), np.array([f_l[2]-f_max[2],0,f_r[2]-f_max[2]]),p0=[0.000002,0,0.002])


#plt.plot(x,cubic1(x,0.000002,0,0.001),label="1. Mode Regression")
x1 = np.linspace(195,225)
x2 = np.linspace(115,145)
x3 = np.linspace(61,90)
axs[1].plot(x1,cubic1(x1,params3[0],params3[1],params3[2]),label="1. Mode Regression")
axs[1].plot(x2,cubic2(x2,params4[0],params4[1],params4[2]),label="2. Mode Regression")
axs[1].plot(x3,cubic3(x3,params5[0],params5[1],params5[2]),label="3. Mode Regression")
#plt.plot(x,cubic1(x,params3[0]),label="1. Mode Regression")
#plt.plot(x,cubic2(x,params4[0]),label="2. Mode Regression")
#plt.plot(x,cubic3(x,params5[0]),label="3. Mode Regression")
axs[1].plot(U_l,f_l-f_max,linewidth = 0,marker = "x",color="black",label="Messdaten")
axs[1].plot(U_max,[0,0,0],linewidth = 0,marker = "x",color="black")
axs[1].plot(U_r,f_r-f_max,linewidth = 0,marker = "x",color="black")
axs[1].set_xlim(50,230)
axs[1].set_ylim(-0.05,0.05)
axs[1].grid()
axs[1].set_xlabel(r"Reflector voltage $U/\mathrm{V}$",fontsize="6")
axs[1].set_ylabel(r"Frequenzänderung $/\mathrm{GHz}$",fontsize="6")
#axs[1].tight_layout()
axs[1].legend(fontsize="4", loc ="best")
fig.tight_layout()
fig.savefig("content/messung1.pdf")

#differenz der frequenzen
delta_f = abs(f_r-f_l)
delta_f_mean = np.mean(delta_f)
delta_f_std = np.std(delta_f)
print("delta_f",delta_f)

F = delta_f/abs(U_r-U_l)
F_mean = np.mean(F)
F_std = np.std(F)
print("F",F)
print("Mittelwert von F",F_mean)
print("Standardabweichung von F",F_std)

def frequenz(lamda_g,a):
              return const.c*unp.sqrt((1/lamda_g)**2+(1/(2*a))**2)

a = ufloat(22.86,0.046)
a = a*10**-3
lamda_g = 50*10**-3
f1 = frequenz(lamda_g,a)
print("frequenz = ",frequenz(lamda_g,a))
f2 =ufloat(9.103*10**9,0)
v_ph1 = f1*lamda_g
print("v_ph,1 = ",v_ph1)
v_ph2 = f2*lamda_g
print("v_ph,2 = ",v_ph2)
print("mittelwert der phasengeschwindigkeiten:",np.mean([v_ph1,v_ph2]))


def linear(x,a,b):
              return a*x+b

xx = np.array([0.9,1.4,1.75,2.02,2.22])
y = np.array([0.69,1.38,1.79,2.08,2.3])
x_space = np.linspace(0.9,2.22)
paraams, pcovv = op.curve_fit(linear, xx, y)
plt.close()
plt.plot(x_space,linear(x_space,paraams[0],paraams[1]),label="Fit")
plt.plot([1,1.45,1.74,2.01,2.19],y,linewidth = 0,marker = "x",color="black",label="Messdaten")
plt.legend()
plt.grid()
plt.xlabel(r"Schraubentiefe $d/\mathrm{mm}$")
plt.ylabel(r"Dämpfung $d/\mathrm{dB}$")
plt.savefig("content/daempfung.pdf")
plt.close()

print("abweichung zu den herstellerwerten:",np.mean(abs(xx - [1,1.45,1.74,2.01,2.19])))

