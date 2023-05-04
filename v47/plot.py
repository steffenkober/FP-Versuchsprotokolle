import matplotlib.pyplot as plt
import numpy as np



I, U, R_s, R_shield, delta_t = np.genfromtxt("content/Messdaten_v47.txt", unpack = True)
R_s = R_s*10**3 #umrechnung von kiloohm in ohm
R_shield = R_shield*10**3
I = I*10**(-3)

#konstanten
m = 342 #in gramm
M = 63.546 #in gramm per mol
kappa = 137.8*10**9 #pascal
rho = 8960*10**3
v_0 = M/rho

T_zualpha_array = np.array([70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310])
alpha_array = np.array([7,8.5,9.75,10.70,11.5,12.1,12.65,13.15,13.6,13.9,14.25,14.5,14.75,14.95,15.2,15.4,15.6,15.75,15.9,16.1,16.25,16.35,16.5,16.65,16.8])*10**-6
alpha = np.ones(len(R_s))




#definieren der formeln
def temperature_func(R): #in celsius
    return 0.00134*R**2 + 2.296*R - 243.02
def celsius_to_kelvin(T):
    return T + 273.15
def hcap_p(M,m,delta_T,E):
    return (M*E)/(m*delta_T)
def energy(U,I,delta_t):
    return U*I*delta_t
def cp_zu_cv(cp,T,alphaa):
    return cp-9*alpha**2*kappa*v_0*T
def alpha_f(T):
    for i in range(len(T)):
        j=0
        while(T[i]>T_zualpha_array[j]):
            j = j+1
        alpha[i] = (alpha_array[j]-alpha_array[j-1])*(T[i]-T_zualpha_array[j-1])/(T_zualpha_array[j]-T_zualpha_array[j-1])+alpha_array[j-1]
    return alpha
    
    

temperature_shield_c = temperature_func(R_shield)
temperature_s_c = temperature_func(R_s)

temperature_s = celsius_to_kelvin(temperature_s_c)
temperature_shield = celsius_to_kelvin(temperature_shield_c)


delta_temperature_shield = np.zeros(len(temperature_shield))
delta_temperature_s = np.zeros(len(temperature_s))

#rint(temperature_s)
for i in range(len(temperature_shield)-1):
    delta_temperature_shield[i+1] = temperature_shield[i+1]-temperature_shield[i]
#print(delta_temperature_shield)
for i in range(len(temperature_s)-1):
    delta_temperature_s[i+1] = temperature_s[i+1]-temperature_s[i]
#print(delta_temperature_s)
#print(temperature_s)

#berechnung der energie
E = energy(U,I,delta_t)

#print(E)

#berechnung von C_p
c_p = hcap_p(M,m,delta_temperature_s,E)
#print(c_p)

# berechnung alpha
alpha_s = alpha_f(temperature_s)
#print(alpha_s)

#berechung C_v

c_v = cp_zu_cv(c_p,temperature_s,alpha_s)



#hardcodeing die theta_D werte weil die anleitung rückständig ist
theta_D_pro_T = np.array([2.8,2.6,2.8,2.7,2.6,2.3,2,1.9,1.9,1.6,1.6,1.3,1.2,1.3,1.2,1,0.1,0.5,0.8,1,1,0.8,1.4])
print(theta_D_pro_T)
theta_D = theta_D_pro_T[:9] * temperature_s[1:10]
theta_D_std = np.std(theta_D)/np.sqrt(np.size(theta_D))
print(theta_D,np.mean(theta_D),theta_D_std)

#plot von vc
plt.plot(temperature_s,c_v,label="Messdaten zu C_V",color="cornflowerblue")
plt.legend()
plt.grid()
plt.ylabel(r'$C_\mathrm{V} \,\,\text{in}\,\, \unit{\joule\per\mol\per\kelvin}$')
plt.xlabel(r'$T\,\, \text{in} \,\,\unit{\kelvin}')

plt.savefig("build/c_v_plot.pdf")