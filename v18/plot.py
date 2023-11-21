import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const
import scipy.integrate as integrate

measuretime_europium = 4648
measuretime_backgr = 71609 
measuretime_uran = 2246
measuretime_barium = 2876
measuretime_caesium = 3075

##BACKGROUND

background = np.array(np.genfromtxt("daten/Background_031302_230606.txt", unpack = True, dtype = int))
background_norm = background/measuretime_backgr

print(background_norm)

channels_backgr = np.arange(0, len(background), 1)

#############
# SAVE AS PDF:
#############
'''
plt.plot(channels_backgr, background_norm)
#plt.yscale("log")
plt.grid()
plt.xlim(0, 8000)
plt.show()
'''

##EUROPIUM 152

EU_152 = np.array(np.genfromtxt("daten/Europium_152_232002_003737.txt", unpack = True, dtype = int))

print(EU_152)

channels = np.arange(0, len(EU_152), 1)


EU_152_no_backgr = EU_152 - background_norm*measuretime_europium

peak1_eu_count = 0
peak2_eu_count = 0
peak3_eu_count = 0
peak4_eu_count = 0
peak5_eu_count = 0
peak6_eu_count = 0
peak7_eu_count = 0
peak8_eu_count = 0
peak9_eu_count = 0
peak10_eu_count = 0

#peaks = 

for i in EU_152_no_backgr[250:750]:
    peak1_eu_channel = 250 +  np.where(EU_152[250:750] == EU_152[250:750].max())[0][0]
    if i > (EU_152[250:750].max()/2):
        peak1_eu_count += i

for i in EU_152_no_backgr[1000:1500]:
    peak2_eu_channel = 1000 + np.where(EU_152[1000:1500] == EU_152[1000:1500].max())[0][0]
    if i > (EU_152[1000:1500].max()/2):
        peak2_eu_count += i

for i in EU_152_no_backgr[1500:1900]:
    peak3_eu_channel = 1500 +  np.where(EU_152[1500:1900] == EU_152[1500:1900].max())[0][0]
    if i > (EU_152[1500:1900].max()/2):
        peak3_eu_count += i

for i in EU_152_no_backgr[1900:2050]:
    peak4_eu_channel = 1900 + np.where(EU_152[1900:2050] == EU_152[1900:2050].max())[0][0] 
    if i > (EU_152[1900:2050].max()/2):
        peak4_eu_count += i

for i in EU_152_no_backgr[2050:2270]:
    peak5_eu_channel = 2050 + np.where(EU_152[2050:2270] == EU_152[2050:2270].max())[0][0] 
    if i > (EU_152[2050:2270].max()/2):
        peak5_eu_count += i

for i in EU_152_no_backgr[3400:4000]:
    peak6_eu_channel = 3400 + np.where(EU_152[3400:4000] == EU_152[3400:4000].max())[0][0] 
    if i > (EU_152[3400:4000].max()/2):
        peak6_eu_count += i

for i in EU_152_no_backgr[4500:4800]:
    peak7_eu_channel = 4500 + np.where(EU_152[4500:4800] == EU_152[4500:4800].max())[0][0] 
    if i > (EU_152[4500:4800].max()/2):
        peak7_eu_count += i

for i in EU_152_no_backgr[5100:5300]:
    peak8_eu_channel = 5100 + np.where(EU_152[5100:5300] == EU_152[5100:5300].max())[0][0] 
    if i > (EU_152[5100:5300].max()/2):
        peak8_eu_count += i

for i in EU_152_no_backgr[5300:5500]:
    peak9_eu_channel = 5300 + np.where(EU_152[5300:5500] == EU_152[5300:5500].max())[0][0] 
    if i > (EU_152[5300:5500].max()/2):
        peak9_eu_count += i
    
for i in EU_152_no_backgr[6500:7000]:
    peak10_eu_channel = 6500 + np.where(EU_152[6500:7000] == EU_152[6500:7000].max())[0][0] 
    if i > (EU_152[6500:7000].max()/2):
        peak10_eu_count += i

peaks_eu_channel = [peak1_eu_channel, peak2_eu_channel, peak3_eu_channel, peak4_eu_channel, peak5_eu_channel, peak6_eu_channel, peak7_eu_channel, peak8_eu_channel, peak9_eu_channel, peak10_eu_channel]
peaks_eu_count = [peak1_eu_count, peak2_eu_count, peak3_eu_count, peak4_eu_count, peak5_eu_count, peak6_eu_count, peak7_eu_count, peak8_eu_count, peak9_eu_count, peak10_eu_count]
stddev_peaks_eu_count = np.sqrt(peaks_eu_count)


print("------------------------------------------------------")
print("Peak-channels Europium-152: ")
print(peaks_eu_channel)
print("\nCounts in peaks Europium-152: ")
print(peaks_eu_count, stddev_peaks_eu_count)

print("------------------------------------------------------")

#############
# SAVE AS PDF:
#############

'''
plt.plot(channels, EU_152)
#plt.yscale("log")
plt.grid()
plt.xlim(0, 8000)
plt.show()
'''

##AUSGLEICHSGERADE

def energy(channel, a, b):
    return a*channel + b

energy_eu = np.array([121.78, 244.70, 344.30, 411.12, 443.96, 778.90, 964.08, 1085.90, 1112.10, 1408.00])
prob_eu_100 = np.array([28.6, 7.6, 26.5, 2.2, 3.1, 12.9, 14.6, 10.2, 13.6, 21.0])
prob_eu = prob_eu_100*0.01

params1, pcov1 = op.curve_fit(energy, peaks_eu_channel, energy_eu)
err1 = np.sqrt(np.diag(pcov1))
a1 = ufloat(params1[0], err1[0])
b1 = ufloat(params1[1], err1[1])

print("------------------------------------------------------")
print(f"a1 = {a1:.3e}")
print(f"b1 = {b1:.3e}", "\n")
print("------------------------------------------------------")

#############
# SAVE AS PDF:
#############
'''
x = np.linspace(0, 8000, 8000)
plt.plot( x , energy(x,*params1))
plt.grid()
plt.show()
plt.savefig('plots/energy_for_channel.pdf')
plt.close()
'''
##VOLLENERGIENACHWEISWAHRSCHEINLICHKEIT

def Q(Z, A, W, t, o):
    return (Z * 4 * np.pi)/(A * W * t * o)

def Akt(A_0, theta, t):
    return A_0 * np.exp(((-1)*np.log(2)*t)/theta)

theta_Eu = 426902400 #in sekunden
A_0_Eu = ufloat(4130, 60) #in Bq

Akt_Eu = Akt(A_0_Eu, theta_Eu, 728524800)

print("Activity Europium - 152: ", Akt_Eu)

def omega(a, r):
    return 2 * np.pi * (1 - (a / np.sqrt(a**2 + r**2)) )

Omega = omega(9.5 / 100, 2.25 / 100)

Q_Eu = []

for i in np.arange(1,10, 1):
    Q_Eu.append(Q(ufloat(peaks_eu_count[i], stddev_peaks_eu_count[i]) , Akt_Eu, prob_eu[i] , measuretime_europium, Omega))

print(Omega)
print(Q_Eu)
print(energy_eu[1:])

def potenz(x,a,b):
              return a * ((x)**b) 

params2, pcov2 = op.curve_fit(potenz,  energy_eu[1:], unp.nominal_values(Q_Eu))
err2 = np.sqrt(np.diag(pcov2))
a2 = ufloat(params2[0], err2[0])
b2 = ufloat(params2[1], err2[1])


print("------------------------------------------------------")
print(f"a2 = {a2:.3e}")
print(f"b2 = {b2:.3e}", "\n")

print("------------------------------------------------------")

'''
x = np.linspace(200, 1400, 1400)
plt.plot( x , potenz(x,*params2))
plt.plot( energy_eu[1:] , unp.nominal_values(Q_Eu), marker = 'x', lw = 0)
plt.grid()
plt.show()
plt.savefig('plots/quality.pdf')
plt.close()
'''
###CAESIUM 137

CS_137 = np.array(np.genfromtxt("daten/Cs_137_004008_013130.txt", unpack = True, dtype = int))

print(CS_137)

channels_Cs = np.arange(0, len(CS_137), 1)

CS_137_no_backgr = CS_137 - background_norm*measuretime_caesium

'''
plt.plot(channels_Cs, CS_137_no_backgr,marker = '.', markersize = 1.5,  lw = 0)
plt.yscale("log")
plt.grid()
plt.xlim(0, 8000)
plt.show()
plt.savefig('plots/caesium_channel.pdf')
plt.close()
'''


def gauss_function(x, a, mu, sigma):
    return (a/np.sqrt(2 * np.pi * sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))
def ugauss_function(x, a, mu, sigma):
    return (a/unp.sqrt(2 * np.pi * sigma**2))*unp.exp(-(x-mu)**2/(2*sigma**2))

mu_cs = CS_137_no_backgr[3100:3300].max()

p0 = [1.,3100 + np.where(CS_137_no_backgr[3100:3300] == CS_137_no_backgr[3100:3300].max())[0][0], 1.]
params3, pcov3 = op.curve_fit(gauss_function,  channels_Cs[3100:3300], CS_137_no_backgr[3100:3300], p0=p0)
err3 = np.sqrt(np.diag(pcov3))
a3 = ufloat(params3[0], err3[0])
mu3 = ufloat(params3[1], err3[1])
sigma3 = ufloat(params3[2], err3[2])

print("------------------------------------------------------")
print(f"a3 = {a3:.3e}")
print(f"mu3 = {mu3:.3e}")
print(f"sigma3 = {sigma3:.3e}", "\n")
print("------------------------------------------------------")




I_1_2_gauss = (ugauss_function(mu3, a3, mu3, sigma3))/2
I_1_10_gauss = (ugauss_function(mu3, a3, mu3, sigma3))/10

I_1_2_exp = mu_cs/2
I_1_10_exp = mu_cs/10



k_1_2_1 = ( - unp.sqrt(unp.log((a3/unp.sqrt(2 * np.pi * sigma3**2))/I_1_2_gauss)*2*sigma3**2) + mu3)#
k_1_2_2 = ( unp.sqrt(unp.log((a3/unp.sqrt(2 * np.pi * sigma3**2))/I_1_2_gauss)*2*sigma3**2) + mu3)#

k_1_10_1 = ( - unp.sqrt(unp.log((a3/unp.sqrt(2 * np.pi * sigma3**2))/I_1_10_gauss)*2*sigma3**2) + mu3)#
k_1_10_2  = ( unp.sqrt(unp.log((a3/unp.sqrt(2 * np.pi * sigma3**2))/I_1_10_gauss)*2*sigma3**2) + mu3)#

E_1_2_gauss = (energy(k_1_2_2 , a1, b1)-energy(k_1_2_1 , a1, b1))
E_1_10_gauss = (energy(k_1_10_2 , a1, b1)-energy(k_1_10_1 , a1, b1))

E_1_2_exp = (energy(3198 , a1, b1)-energy(3187 , a1, b1))
E_1_10_exp = (energy(3201 , a1, b1)-energy(3181 , a1, b1))

Delta_E_gauss = E_1_10_gauss/E_1_2_gauss
Delta_E_exp = E_1_10_exp/E_1_2_exp


print("------------------------------------------------------")
print("Gauss max: ", ugauss_function(mu3, a3, mu3, sigma3))
print("Impulse max: ", mu_cs)
print("Gauss Impulse Halb: ", I_1_2_gauss)
print("Gauss Impulse Zehntel: ", I_1_10_gauss)
print("Impulse Halb: ", I_1_2_exp) #3182 & 3200
print("Impulse Zehntel: ", I_1_10_exp) #3181 & 3202
print("Gauss Energie Halb: ", E_1_2_gauss)
print("Gauss Energie Zehntel: ", E_1_10_gauss)
print("Energie Halb: ", E_1_2_exp)
print("Energie Zehntel: ", E_1_10_exp)
print("Delta Energie Gauss: ", Delta_E_gauss)
print("Delta Energie exp: ", Delta_E_exp)
print("------------------------------------------------------")

#def inv_gauss_function1(x, a, mu, sigma):
#    return (unp.sqrt(unp.log((a/unp.sqrt(2 * np.pi * sigma**2))/x)*2*sigma**2) + mu)
    
#def inv_gauss_function2(x, a, mu, sigma):
#    return (- unp.sqrt(-unp.log(x/(a/unp.sqrt(2 * np.pi * sigma**2)))*2*sigma**2) + mu)

#k_1_2_1 = unp.sqrt(unp.log(2)*2*sigma3**2) + mu3


'''
x = np.linspace(3100, 3300, 1000)
k1 = np.linspace(unp.nominal_values(k_1_2_1), unp.nominal_values(k_1_2_2))
k2 = np.linspace(unp.nominal_values(k_1_10_1), unp.nominal_values(k_1_10_2))
plt.plot( x , gauss_function(x,*params3))
plt.plot( k1 , unp.nominal_values(I_1_2_gauss)*np.ones(len(k1)))
plt.plot( k2 , unp.nominal_values(I_1_10_gauss)*np.ones(len(k2)))

plt.plot(channels_Cs[3100:3300], CS_137_no_backgr[3100:3300],marker = '.', markersize = 1.5,  lw = 0)

plt.grid()
plt.show()

plt.savefig('plots/caesium_channel_gauss_fit.pdf')
plt.close()
'''

summed_imp = integrate.quad(gauss_function, 3150, 3250, args=(unp.nominal_values(a3), unp.nominal_values(mu3), unp.nominal_values(sigma3)))
print(summed_imp)

energy_cs = energy(channels_Cs, a1, b1)

print(len(energy_cs))
print(len(CS_137_no_backgr))

'''
plt.plot(channels_Cs, CS_137_no_backgr, marker = '.', markersize = 1.5,  lw = 0)
plt.grid()
plt.xlim(0, 2500)
plt.ylim(0, 60)
plt.show()  #453 #300
plt.savefig('plots/caesium_channel_compton.pdf')
plt.close()

plt.plot(unp.nominal_values(energy_cs), CS_137_no_backgr, marker = '.', markersize = 1.5,  lw = 0)
plt.grid()
plt.xlim(0, 500)
plt.ylim(0, 60)
plt.show()  #453 #300
plt.savefig('plots/caesium_energy_compton.pdf')
plt.close()
'''
#print(np.concatenate((unp.nominal_values(energy_cs[1300:2200]), unp.nominal_values(energy_cs[100:500])), axis=None))

def nichtlin(x, a, b, c, k):
    return a*(x**3) + b*(x**2)+ c*x + k

#p0 = [1.,3100 + np.where(CS_137_no_backgr[3100:3300] == CS_137_no_backgr[3100:3300].max())[0][0], 1.]
params4, pcov4 = op.curve_fit(nichtlin,  unp.nominal_values(energy_cs[1300:2200]), CS_137_no_backgr[1300:2200])
err4 = np.sqrt(np.diag(pcov4))
a4 = ufloat(params4[0], err4[0])
b4 = ufloat(params4[1], err4[1])
c4 = ufloat(params4[2], err4[2])
k4 = ufloat(params4[3], err4[3])


print("------------------------------------------------------")
#print(f"k4 = {k4:.3e}")
print(f"a4 = {a4:.3e}")
print(f"b4 = {b4:.3e}")
print(f"c4 = {c4:.3e}")
print(f"k4 = {k4:.3e}", "\n")
print("------------------------------------------------------")

'''
x = np.linspace(250, 470, 550)
plt.plot( x , nichtlin(x, unp.nominal_values(a4), unp.nominal_values(b4), unp.nominal_values(c4), unp.nominal_values(k4)))
plt.plot(unp.nominal_values(energy_cs[0:2500]), CS_137_no_backgr[0:2500],marker = '.', markersize = 1.5,  lw = 0)
plt.grid()
plt.show()
plt.savefig('plots/caesium_energy_fit_compton.pdf')
plt.close()
'''

print("Compton-Kante: ", 470)
print("Rückstreu-Kante: ", 190)
print("Gamma-Quant Energie (aus Vollenergielinie): ", 626)

'''
plt.plot(unp.nominal_values(energy_cs), CS_137_no_backgr, marker = '.', markersize = 1.5,  lw = 0)
plt.grid()
plt.xlim(500, 1000)
plt.ylim(0, 60)
plt.show()  #453 #300
'''
###########
#Barium-133
###########

BA_133 = np.array(np.genfromtxt("daten/Ba_133_014125_022923.txt", unpack = True, dtype = int))

channels_ba = np.arange(0, len(BA_133), 1)

BA_133_no_backgr = BA_133 - background_norm*measuretime_barium

energy_ba = energy(channels_ba, a1, b1)

'''
plt.plot(channels_ba, BA_133_no_backgr, marker = '.', markersize = 1.5,  lw = 1)
plt.grid()
plt.xlim(0, 2000)
#plt.ylim(0, 60)
plt.show()  #453 #300
plt.savefig('plots/barium_channel.pdf')
plt.close()

plt.plot(unp.nominal_values(energy_cs), BA_133_no_backgr, marker = '.', markersize = 1.5,  lw = 1)
plt.grid()
plt.xlim(0, 500)
#plt.ylim(0, 60)
plt.show()  #453 #300
plt.savefig('plots/barium_energy.pdf')
plt.close()
'''


p01 = [1.,1300 + np.where(BA_133_no_backgr[1300:1400] == BA_133_no_backgr[1300:1400].max())[0][0], 1.]
p02 = [1.,1400 + np.where(BA_133_no_backgr[1400:1500] == BA_133_no_backgr[1400:1500].max())[0][0], 1.]
p03 = [1.,1600 + np.where(BA_133_no_backgr[1600:1800] == BA_133_no_backgr[1600:1800].max())[0][0], 1.]
p04 = [1.,1800 + np.where(BA_133_no_backgr[1800:1900] == BA_133_no_backgr[1800:1900].max())[0][0], 1.]

params5, pcov5 = op.curve_fit(gauss_function,  channels_ba[1300:1400], BA_133_no_backgr[1300:1400], p0=p01)
params6, pcov6 = op.curve_fit(gauss_function,  channels_ba[1400:1500], BA_133_no_backgr[1400:1500], p0=p02)
params7, pcov7 = op.curve_fit(gauss_function,  channels_ba[1600:1800], BA_133_no_backgr[1600:1800], p0=p03)
params8, pcov8 = op.curve_fit(gauss_function,  channels_ba[1800:1900], BA_133_no_backgr[1800:1900], p0=p04)

err5 = np.sqrt(np.diag(pcov5))
a5 = ufloat(params5[0], err5[0])
mu5 = ufloat(params5[1], err5[1])
sigma5 = ufloat(params5[2], err5[2])

err6 = np.sqrt(np.diag(pcov6))
a6 = ufloat(params6[0], err6[0])
mu6 = ufloat(params6[1], err6[1])
sigma6 = ufloat(params6[2], err6[2])

err7 = np.sqrt(np.diag(pcov7))
a7 = ufloat(params7[0], err7[0])
mu7 = ufloat(params7[1], err7[1])
sigma7 = ufloat(params7[2], err7[2])

err8 = np.sqrt(np.diag(pcov8))
a8 = ufloat(params8[0], err8[0])
mu8 = ufloat(params8[1], err8[1])
sigma8 = ufloat(params8[2], err8[2])

Z_BA_133_1 = integrate.quad(gauss_function, 1323, 1360, args=(unp.nominal_values(a5), unp.nominal_values(mu5), unp.nominal_values(sigma5)))
Z_BA_133_2 = integrate.quad(gauss_function, 1445, 1488, args=(unp.nominal_values(a6), unp.nominal_values(mu6), unp.nominal_values(sigma6)))
Z_BA_133_3 = integrate.quad(gauss_function, 1700, 1750, args=(unp.nominal_values(a7), unp.nominal_values(mu7), unp.nominal_values(sigma7)))
Z_BA_133_4 = integrate.quad(gauss_function, 1837, 1872, args=(unp.nominal_values(a8), unp.nominal_values(mu8), unp.nominal_values(sigma8)))

def A(Z, Q, W, t, o):
    return (Z * 4 * np.pi)/(Q * W * t * o)

A_BA_133_1 = A(unp.nominal_values(Z_BA_133_1), potenz(unp.nominal_values(energy_ba[1340]) , unp.nominal_values(a2), unp.nominal_values(b2)), 0.0713, measuretime_barium, Omega)
A_BA_133_2 = A(unp.nominal_values(Z_BA_133_2), potenz(unp.nominal_values(energy_ba[1467]) , unp.nominal_values(a2), unp.nominal_values(b2)), 0.1831, measuretime_barium, Omega)
A_BA_133_3 = A(unp.nominal_values(Z_BA_133_3), potenz(unp.nominal_values(energy_ba[1723]) , unp.nominal_values(a2), unp.nominal_values(b2)), 0.6205, measuretime_barium, Omega)
A_BA_133_4 = A(unp.nominal_values(Z_BA_133_4), potenz(unp.nominal_values(energy_ba[1858]) , unp.nominal_values(a2), unp.nominal_values(b2)), 0.0894, measuretime_barium, Omega)

Akt_Ba = np.array([A_BA_133_1[0], A_BA_133_2[0], A_BA_133_3[0], A_BA_133_4[0]])

print(unp.nominal_values(Akt_Ba))

print("Aktivität Barium-133: ", unp.nominal_values(Akt_Ba).mean())
print(unp.nominal_values(Akt_Ba).std())

###########
#Uran
###########

URAN = np.array(np.genfromtxt("daten/Uran_023355_031200.txt", unpack = True, dtype = int))

channels_uran = np.arange(0, len(URAN), 1)

URAN_no_backgr = URAN - background_norm*measuretime_uran

energy_uran = energy(channels_uran, a1, b1)


plt.plot(channels_uran, URAN_no_backgr, marker = '.', markersize = 1.5,  lw = 1)
plt.grid()
#plt.xlim(0, 2000)
#plt.ylim(0, 60)
plt.tight_layout()
#plt.savefig('plots/uran_channel.pdf')
plt.show()

plt.close()

plt.plot(unp.nominal_values(energy_uran), URAN_no_backgr, marker = '.', markersize = 1.5,  lw = 1)
plt.grid()
#plt.xlim(0, 500)
#plt.ylim(0, 60)
plt.tight_layout()
plt.show()

#plt.savefig('plots/uran_energy.pdf')
#plt.close()


p05 = [1.,902, 1.]
p06 = [1.,1172, 1.]
p07 = [1.,1428, 1.]
p08 = [1.,1701, 1.]
p09 = [1.,2940, 1.]
p10 = [1.,3705, 1.]
p11 = [1.,4505, 1.]
p12 = [1.,5399, 1.]
p13 = [1.,5967, 1.]

print(np.array([p05, p06, p07, p08, p09, p10, p11, p12, p13]))

params9, pcov9   = op.curve_fit(gauss_function,  channels_uran[880:920], URAN_no_backgr[880:920], p0=p05)
params10, pcov10 = op.curve_fit(gauss_function,  channels_uran[1150:1200], URAN_no_backgr[1150:1200], p0=p06)
params11, pcov11 = op.curve_fit(gauss_function,  channels_uran[1407:1446], URAN_no_backgr[1407:1446], p0=p07)
params12, pcov12 = op.curve_fit(gauss_function,  channels_uran[1685:1720], URAN_no_backgr[1685:1720], p0=p08)
params13, pcov13 = op.curve_fit(gauss_function,  channels_uran[2920:2960], URAN_no_backgr[2920:2960], p0=p09)
params14, pcov14 = op.curve_fit(gauss_function,  channels_uran[3676:3738], URAN_no_backgr[3676:3738], p0=p10)
params15, pcov15 = op.curve_fit(gauss_function,  channels_uran[4477:4528], URAN_no_backgr[4477:4528], p0=p11)
params16, pcov16 = op.curve_fit(gauss_function,  channels_uran[5360:5440], URAN_no_backgr[5360:5440], p0=p12)
params17, pcov17 = op.curve_fit(gauss_function,  channels_uran[5922:6010], URAN_no_backgr[5922:6010], p0=p13)



'''
x = np.linspace(500, 4000, 4000)
plt.plot( x , gauss_function_(x, *params10))
plt.plot(channels_uran[500:4000], URAN_no_backgr[500:4000],marker = '.', markersize = 1.5,  lw = 0)
plt.grid()
plt.show()
#plt.savefig('plots/caesium_energy_fit_compton.pdf')
plt.close()
'''
Z_URAN_238_1 = integrate.quad(gauss_function, 893, 912, args=(params9[0], params9[1], params9[2])) 
Z_URAN_238_2 = integrate.quad(gauss_function, 1163, 1181, args=(params10[0], params10[1], params10[2])) 
Z_URAN_238_3 = integrate.quad(gauss_function, 1417, 1439, args=(params11[0], params11[1], params11[2])) 
Z_URAN_238_4 = integrate.quad(gauss_function, 1688, 1713, args=(params12[0], params12[1], params12[2])) 
Z_URAN_238_5 = integrate.quad(gauss_function, 2926, 2952, args=(params13[0], params13[1], params13[2]))
Z_URAN_238_6 = integrate.quad(gauss_function, 3689, 3717, args=(params14[0], params14[1], params14[2]))
Z_URAN_238_7 = integrate.quad(gauss_function, 4489, 4520, args=(params15[0], params15[1], params15[2]))
Z_URAN_238_8 = integrate.quad(gauss_function, 5380, 5421, args=(params16[0], params16[1], params16[2]))
Z_URAN_238_9 = integrate.quad(gauss_function, 5945, 5988, args=(params17[0], params17[1], params17[2]))

Z_URAN_238 = np.array([Z_URAN_238_1[0], Z_URAN_238_2[0], Z_URAN_238_3[0], Z_URAN_238_4[0], Z_URAN_238_5[0], Z_URAN_238_6[0], Z_URAN_238_7[0], Z_URAN_238_8[0], Z_URAN_238_9[0]])
Z_BISMUT = np.array([Z_URAN_238_5[0], Z_URAN_238_6[0], Z_URAN_238_8[0]])
Z_BLEI = np.array([Z_URAN_238_2[0], Z_URAN_238_3[0], Z_URAN_238_4[0]])
Z_RADON = np.array([Z_URAN_238_1[0]])
print(Z_URAN_238)

#def A(Z, Q, W, t, o):
#    return (Z * 4 * np.pi)/(Q * W * t * o)

A_BISMUT_214_1 = A(unp.nominal_values(Z_BISMUT[0]), potenz(unp.nominal_values(energy_uran[2940]) , unp.nominal_values(a2), unp.nominal_values(b2)), 0.4549, measuretime_uran, 0.5)
A_BISMUT_214_2 = A(unp.nominal_values(Z_BISMUT[1]), potenz(unp.nominal_values(energy_uran[3705]) , unp.nominal_values(a2), unp.nominal_values(b2)), 0.0489, measuretime_uran, 0.5)
A_BISMUT_214_3 = A(unp.nominal_values(Z_BISMUT[2]), potenz(unp.nominal_values(energy_uran[5399]) , unp.nominal_values(a2), unp.nominal_values(b2)), 0.1491, measuretime_uran, 0.5)
A_BLEI_214_1 = A(unp.nominal_values(Z_BLEI[0]), potenz(unp.nominal_values(energy_uran[1172]) , unp.nominal_values(a2), unp.nominal_values(b2)), 0.0727, measuretime_uran, 0.5)
A_BLEI_214_2 = A(unp.nominal_values(Z_BLEI[1]), potenz(unp.nominal_values(energy_uran[1428]) , unp.nominal_values(a2), unp.nominal_values(b2)), 0.1841, measuretime_uran, 0.5)
A_BLEI_214_3 = A(unp.nominal_values(Z_BLEI[2]), potenz(unp.nominal_values(energy_uran[1701]) , unp.nominal_values(a2), unp.nominal_values(b2)), 0.356, measuretime_uran, 0.5)
A_RADON_226 = A(unp.nominal_values(Z_RADON[0]), potenz(unp.nominal_values(energy_uran[902]) , unp.nominal_values(a2), unp.nominal_values(b2)), 0.0356, measuretime_uran, 0.5)

Akt_Bi = np.array([A_BISMUT_214_1, A_BISMUT_214_2, A_BISMUT_214_3])
Akt_Pb = np.array([A_BLEI_214_1, A_BLEI_214_2, A_BLEI_214_3])

print("Aktivität Bi-214: ", unp.nominal_values(Akt_Bi).mean())
print(unp.nominal_values(Akt_Bi).std())

print("Aktivität Pb-214: ", unp.nominal_values(Akt_Pb).mean())
print(unp.nominal_values(Akt_Pb).std())

print("Aktivität Ra-226: ", A_RADON_226)
