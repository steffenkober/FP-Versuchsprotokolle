import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import scipy.constants as const

####################################################################################################################################################################################
#########################                       TEM-00                  ############################################################################################################
####################################################################################################################################################################################
r_00, I_00 = np.genfromtxt("content/Data/TEM_0_0.txt", unpack = True)
r_00 = r_00*10**(-3) #meter
I_00 = I_00*10**(-9) #Ampere

def TEM_00(x, I_0,x_0,w):
    return I_0*np.exp(-(2*(x-x_0)**2)/(w**2))

params_00, pcov_00 = op.curve_fit(TEM_00, r_00, I_00, p0=[9.5*10**-6,0.0038,2*10**-3])#
err_00 = np.sqrt(np.diag(pcov_00))
print("TEM_00",params_00,err_00,"TEM_00")
#plotting
r_lin = np.linspace(2*10**-3,6*10**-3,1000)
plt.plot(r_lin*10**3,TEM_00(r_lin, *params_00)*10**6,c="orangered",label="Fit",lw=0.9)
plt.plot(r_00*10**3, I_00*10**6,c="seagreen",marker="x",lw=0,label="Data TEM 00")
plt.grid()
plt.legend()
plt.ylabel(r"$I\,\mathrm{/}\,\unit{\micro\ampere}$")
plt.xlabel(r"$r \,\mathrm{/}\,\unit{\milli\meter}$")
plt.tight_layout()
#plt.show()
plt.savefig("build/TEM_00.pdf")
plt.close()

####################################################################################################################################################################################
#########################                       TEM-01                  ############################################################################################################
####################################################################################################################################################################################
r_01, I_01 = np.genfromtxt("content/Data/TEM_0_1.txt", unpack = True)
r_01 = r_01*10**(-3) #meter
I_01 = I_01*10**(-9) #Ampere

def TEM_01(x, I_0, x_0, x_1, w):
    return I_0*8*((x-x_0)**2/w**2)*np.exp(-2*(x-x_1)**2/(w**2))

params_01, pcov_01 = op.curve_fit(TEM_01, r_01, I_01,p0=[1.2*10**(-6),0.0051,0.0055,0.002],maxfev=1000)#
err_01 = np.sqrt(np.diag(pcov_01))
print("TEM_01",params_01,err_01,"TEM_01")
#plotting
r_lin_1 = np.linspace(0.004,0.007,1000)
plt.plot(r_lin_1*10**3,TEM_01(r_lin_1, *params_01)*10**6,c="orangered",label="Fit",lw=0.9)
plt.plot(r_01*10**3, I_01*10**6,c="seagreen",marker="x",lw=0,label="Data TEM 01")
plt.grid()
plt.legend()
plt.ylabel(r"$I\,\mathrm{/}\,\unit{\micro\ampere}$")
plt.xlabel(r"$r \,\mathrm{/}\,\unit{\milli\meter}$")
plt.tight_layout()
plt.savefig("build/TEM_01.pdf")
#plt.show()
plt.close()



####################################################################################################################################################################################
#########################                       Polarisation            ############################################################################################################
####################################################################################################################################################################################
Phi_pol, I_pol = np.genfromtxt("content/Data/polarisation.txt",unpack=True)
I_pol = I_pol*10**-3

def gradtorad(grad):
    return grad*np.pi/180

Phi_pol = gradtorad(Phi_pol)

def intensity_pol(phi,I_0,phi_0):
    return I_0*(np.sin(phi-phi_0))**2

params_pol, pcov_pol = op.curve_fit(intensity_pol, Phi_pol, I_pol, p0 = [0.003,0.1],maxfev=1000)#
err_pol = np.sqrt(np.diag(pcov_pol))
print("Polarisation",params_pol,err_pol,"Polarisation")

phi_lin = np.linspace(0,2*np.pi,1000)

plt.plot(phi_lin,intensity_pol(phi_lin,*params_pol)*10**3,c="orangered",label="Fit",lw=0.9)
plt.plot(Phi_pol,I_pol*10**3,c="seagreen",marker="x",lw=0,label="Data")

plt.grid()
plt.legend()
plt.ylabel(r"Leistung $\,\mathrm{/}\,\unit{\milli\watt}$")
plt.xlabel(r"$\Phi \,\mathrm{/}\,\unit{rad}$")
plt.tight_layout()
plt.savefig("build/Polarisation.pdf")
#plt.show()
plt.close()


####################################################################################################################################################################################
#########################                       Wellenlänge             ############################################################################################################
####################################################################################################################################################################################

def lamda(delta,d,g,n):
    return np.sin(np.arctan(delta/d))/(g*n)

#erstes Gitter
print("erstes Gitter")
print("g = ",1200)
print("d = ",21.3,"cm")
print("delta = ",24.4,"cm")
print("Wellenlänge = ",lamda(0.244,0.213,1200/(10**-3),1),"m")

#zweites Gitter
print("zweites Gitter")
print("g = ",600)
print("d = ",21.3,"cm")
print("delta = ",8.9,"cm")
print("Wellenlänge = ",lamda(0.089,0.213,600/(10**-3),1),"m")
array = [lamda(0.089,0.213,600/(10**-3),1),lamda(0.244,0.213,1200/(10**-3),1)]
print("Mittelwert",np.mean(array))
print("std",np.std(array))


####################################################################################################################################################################################
#########################                       multi-mode-operation             ###################################################################################################
####################################################################################################################################################################################

mm1_freq, mm1_amp = np.genfromtxt("content/Data/multimode_1.txt",unpack=True)
mm2_freq, mm2_amp = np.genfromtxt("content/Data/multimode_2.txt",unpack=True)
mm3_freq, mm3_amp = np.genfromtxt("content/Data/multimode_3.txt",unpack=True)
mm4_freq, mm4_amp = np.genfromtxt("content/Data/multimode_4.txt",unpack=True)
mm5_freq, mm5_amp = np.genfromtxt("content/Data/multimode_5.txt",unpack=True)

def mittlere_abstaende(array,returnus=[]):
    for i in range(len(array)-1):
        returnus.append(abs(array[i]-array[i+1]))
        mean = np.mean(returnus)
        std = np.std(returnus)
    return mean,std

print("delta_f : ",mittlere_abstaende(mm1_freq))
print("delta_f : ",mittlere_abstaende(mm2_freq))
print("delta_f : ",mittlere_abstaende(mm3_freq))
print("delta_f : ",mittlere_abstaende(mm4_freq))
print("delta_f : ",mittlere_abstaende(mm5_freq))

delta_f_doppler = 1500 #MHz
 
print("MM1 Anzahl: ",delta_f_doppler/mittlere_abstaende(mm1_freq)[0])
print("MM2 Anzahl: ",delta_f_doppler/mittlere_abstaende(mm2_freq)[0])
print("MM3 Anzahl: ",delta_f_doppler/mittlere_abstaende(mm3_freq)[0])
print("MM4 Anzahl: ",delta_f_doppler/mittlere_abstaende(mm4_freq)[0])
print("MM5 Anzahl: ",delta_f_doppler/mittlere_abstaende(mm5_freq)[0])

def F_theo(L):
    return const.c/(2*L)

print("Länge :",63,"f_theo",F_theo(63*10**(-2))/(10**6))
print("Länge :",77.4,"f_theo",F_theo(77.4*10**(-2))/(10**6))
print("Länge :",87,"f_theo",F_theo(87*10**(-2))/(10**6))
print("Länge :",105.8,"f_theo",F_theo(105.8*10**(-2))/(10**6))
print("Länge :",119.9,"f_theo",F_theo(119.9*10**(-2))/(10**6))

### Abw

lambda_theo = 633.8 ### nm
print(np.mean(array))

delta_lambda = abs(635.17 - lambda_theo)/lambda_theo*100

print(delta_lambda)

arrayy = [F_theo(63*10**(-2))/(10**6),F_theo(77.4*10**(-2))/(10**6),F_theo(87*10**(-2))/(10**6),F_theo(105.8*10**(-2))/(10**6),F_theo(119.9*10**(-2))/(10**6)]
f_exp = [237.25,210.8,195.0,178.04,162.83]
abw_fs = [0,0,0,0,0]
for i in range(5):
    abw_fs[i] = np.abs(f_exp[i]-arrayy[i])/arrayy[i]


abw_f = np.mean(abw_fs)
print("fs:",abw_fs)
print("Abweichung schwebung:",abw_f)