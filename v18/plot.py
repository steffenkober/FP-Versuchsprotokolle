import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import uncertainties.unumpy as unp
from uncertainties import ufloat
import scipy.constants as const

measuretime_europium = 4648
measuretime_backgr = 71609

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


EU_152_no_backgr = EU_152 - background_norm*4648

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
'''

###CAESIUM 137

CS_137 = np.array(np.genfromtxt("daten/Cs_137_004008_013130.txt", unpack = True, dtype = int))

print(CS_137)

channels_Cs = np.arange(0, len(CS_137), 1)

CS_137_no_backgr = CS_137 - background_norm*3075

'''
plt.plot(channels_Cs, CS_137_no_backgr,marker = '.', markersize = 1.5,  lw = 0)
plt.yscale("log")
plt.grid()
plt.xlim(0, 8000)
plt.show()
'''


def gauss_function(x, a, mu, sigma):
    return (a/np.sqrt(2 * np.pi * sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))
def ugauss_function(x, a, mu, sigma):
    return (a/unp.sqrt(2 * np.pi * sigma**2))*unp.exp(-(x-mu)**2/(2*sigma**2))
#mu_cs = 3100 + np.where(CS_137_no_backgr[3100:3300] == CS_137_no_backgr[3100:3300].max())[0][0] 
#print(mu_cs)
#sum_Cs_peak=0
#for i in CS_137_no_backgr[3100:3300]:
#    sum_Cs_peak += i

#sigma_cs = np.sqrt(sum_Cs_peak)
#print(sum_Cs_peak)
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



I_1_2 = (ugauss_function(mu3, a3, mu3, sigma3))/2
I_1_10 = (ugauss_function(mu3, a3, mu3, sigma3))/10

#def inv_gauss_function1(x, a, mu, sigma):
#    return (unp.sqrt(unp.log((a/unp.sqrt(2 * np.pi * sigma**2))/x)*2*sigma**2) + mu)
    
#def inv_gauss_function2(x, a, mu, sigma):
#    return (- unp.sqrt(-unp.log(x/(a/unp.sqrt(2 * np.pi * sigma**2)))*2*sigma**2) + mu)

#k_1_2_1 = unp.sqrt(unp.log(2)*2*sigma3**2) + mu3
k_1_2_1 = ( - unp.sqrt(unp.log((a3/unp.sqrt(2 * np.pi * sigma3**2))/I_1_2)*2*sigma3**2) + mu3)#
k_1_2_2 = ( unp.sqrt(unp.log((a3/unp.sqrt(2 * np.pi * sigma3**2))/I_1_2)*2*sigma3**2) + mu3)#

k_1_10_1 = ( - unp.sqrt(unp.log((a3/unp.sqrt(2 * np.pi * sigma3**2))/I_1_10)*2*sigma3**2) + mu3)#
k_1_10_2  = ( unp.sqrt(unp.log((a3/unp.sqrt(2 * np.pi * sigma3**2))/I_1_10)*2*sigma3**2) + mu3)#


x = np.linspace(3100, 3300, 1000)
k1 = np.linspace(unp.nominal_values(k_1_2_1), unp.nominal_values(k_1_2_2))
k2 = np.linspace(unp.nominal_values(k_1_10_1), unp.nominal_values(k_1_10_2))
plt.plot( x , gauss_function(x,*params3))
plt.plot( k1 , unp.nominal_values(I_1_2)*np.ones(len(k1)))
plt.plot( k2 , unp.nominal_values(I_1_10)*np.ones(len(k2)))

plt.plot(channels_Cs[3100:3300], CS_137_no_backgr[3100:3300],marker = '.', markersize = 1.5,  lw = 0)

plt.grid()
plt.show()
