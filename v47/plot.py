import matplotlib.pyplot as plt
import numpy as np



I, U, R_s, R_shield, delta_t = np.genfromtxt("content/Messdaten_v47.txt", unpack = True)

def temperature_func(R): #in kelvin
    return 0.00134*R**2 + 2.296*R - 243.02

temperature_shield = temperature_func(R_shield)
temperature_s = temperature_func(R_s)
delta_temperature_shield = np.zeros(len(temperature_shield))
delta_temperature_s = np.zeros(len(temperature_s))

for i in range(len(temperature_shield)-1):
    delta_temperature_shield[i+1] = temperature_shield[i+1]-temperature_shield[i]
print(delta_temperature_shield)



#x = np.linspace(0, 10, 1000)
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