import numpy as np
import matplotlib.pyplot as plt 
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.constants import c
from scipy.optimize import curve_fit
import csv
from dataclasses import dataclass

###################################################################################################################################
# Datatypes

@dataclass
class Material_parameters:
    d: float
    n_1: float
    k_1: float 
    n_3: float 
    k_3: float

###################################################################################################################################
# Functions

def lin(A, B, x):
    return A*x+B

def H_0(data_ref, data_sam): #takes in two spectral amplitudes and dives them to return complex transfer function
    return (data_sam/data_ref)

def n(freq, d, phase): # takes in the frequency of the dataset, the thickness of the sample d and the frequency resolved phase and returns the real refractive index
    return (1 + (c* phase)/(freq* d) ) #    return (1 - c/(freq* d) *phase)


def k(freq, d, H_0, n): # takes in the frequency of the dataset, the thickness of the sample d and the frequency resolved H_0 and returns the complex refractive index
    n_real = n
    ln_a = np.log((4*n_real)/(n_real + 1)**2)
    ln_b = np.log(np.abs(H_0))
    n_im = c/(freq *d) *(ln_a - ln_b)
    return n_im

def FFT_func(I, t): # FFT
    N = len(t) #number of total data points
    timestep = np.abs(t[2]-t[3]) # the time between each data point
    FX = fft(I)[:N//2] #the fourier transform of the intensity. 
    FDelay = fftfreq(N, d=timestep)[:N//2] #FFT of the time to frequencies. 
    return [FDelay[10:], FX[10:]] # cut of the noise frequency

def unwrapping_alt(amp_ref, amp_sam, freq_ref): #unwrapping from paper Phase_correction_in_THz_TDS_JIMT_revision_clean.pdf
    angle_ref = np.angle(amp_ref)
    angle_sam = np.angle(amp_sam)
    phase_dif = (np.unwrap(np.abs(angle_ref - angle_sam)))
    params,_ = curve_fit(lin, freq_ref, phase_dif)
    phase_dif_0 = phase_dif - 2*np.pi*np.floor(params[1]/np.pi)
    phase_ref_0 = freq_ref * t_peak_ref
    phase_sam_0 = freq_sam * t_peak_sam
    phase_offset = freq_ref * (t_peak_sam - t_peak_ref)
    return phase_dif_0 - phase_ref_0 + phase_sam_0 + phase_offset

def error_function(H_0_calc, H_0_measured): # The error function needs to be minimized to find the values for n and k
    # a good explanation can be found  "A Reliable Method for Extraction of Material
    # Parameters in Terahertz Time-Domain Spectroscopy"
    delta_rho = np.zeros(H_0_calc.shape) # prepare arrays of same shape as H_0
    delta_phi = np.zeros(H_0_calc.shape)
    angle_mes = np.angle(H_0_measured)
    print(H_0_measured.size)
    if H_0_measured.size < 2:
        phase_mes = np.unwrap([angle_mes])
    else:
        phase_mes = np.unwrap(angle_mes)    
    angle_0 = np.zeros(H_0_calc.shape)
    phase_0 = np.zeros(H_0_calc.shape)
    for i in range(len(H_0_calc)):
        for j in range(len(H_0_calc)): # array should be symmetrical
            delta_rho[i,j] = np.log(np.abs(H_0_calc[i,j])) - np.log(np.abs(H_0_measured))
            angle_0[i,j] = np.angle(H_0_calc[i,j]) #angle between complex numbers
            phase_0[i,j] = np.unwrap([angle_0[i,j]])  #phase 
            delta_phi[i,j] = phase_0[i,j] - phase_mes
    print(delta_phi.shape, ' ', delta_rho.shape)
    return delta_phi**2 + delta_rho**2 # this should be minimized in the process

def delta_of_r(n, k, H_0_measured, freq, Material_parameter): #if needed n_1,n_3 and k_1, k_3 can also be added
    H_0_calc = Transfer_function_three_slabs(freq, Material_parameter.n_1 ,n, Material_parameter.n_3, Material_parameter.k_1, k, Material_parameter.k_3, Material_parameter.d, False)
    angle_mes = np.angle(H_0_measured)  
    phase_mes = np.unwrap([angle_mes])  
    #if(H_0_calc <= 0):
    #    print(H_0_calc, " n ", n, " k ", k)
    delta_rho = np.log(np.abs(H_0_calc)) - np.log(np.abs(H_0_measured))
    angle_0 = np.angle(H_0_calc) #angle between complex numbers
    phase_0 = np.unwrap([angle_0])  #phase 
    delta_phi = (phase_0 - phase_mes)
    #print("Shape delta: ", np.shape(delta_phi**2 + delta_rho**2))
    return delta_phi**2 + delta_rho**2 # this should be minimized in the process


def paraboilid(r, A, b, c): # do I need this function?
    return (1/2 * r * A * r - b * r + c)

def grad(n, k, H_0_measured, freq, Material_parameter, h = 10**(-6)):
    """ This function calculates the gradient of the delta(r_p) function.
        The result will be a vector of dim 2 [A, B] so like
        A
        B
    """
    A = (delta_of_r(n + h, k, H_0_measured, freq, Material_parameter) - delta_of_r(n - h, k, H_0_measured, freq, Material_parameter))/2*h
    # A = delta(r_p)/dn = (delta(n + h/2, k) - delta(n - h/2, k)) / h
    B = (delta_of_r(n, k + h, H_0_measured, freq, Material_parameter) - delta_of_r(n, k - h, H_0_measured, freq, Material_parameter))/2*h 
    grad_ = np.array([A[0], B[0]])
    return grad_


def hessian(n, k, H_0_measured, freq, Material_parameter, h = 10**(-6)):
    """ This function calculates the second derivitave of delta(r_p).
        The result will be a hessian matrix.
        The matrix is a 2x2 matrix [[A,B],[C,D]] so like
        A B
        C D
        """
    A = (delta_of_r(n + h, k, H_0_measured, freq, Material_parameter) - 2*delta_of_r(n, k, H_0_measured, freq, Material_parameter) - delta_of_r(n - h, k, H_0_measured, freq, Material_parameter))/h**2
    # A = d²delta(r_p)/dn² = (delta(n + h, k) - 2*delta(n,k) - delta(n-h,k))/h²
    B = (delta_of_r(n + h/2, k + h/2, H_0_measured, freq, Material_parameter) - delta_of_r(n + h/2, k - h/2, H_0_measured, freq, Material_parameter) - delta_of_r(n - h/2, k + h/2, H_0_measured, freq, Material_parameter) + delta_of_r(n - h/2, k - h/2, H_0_measured, freq, Material_parameter))/h**2
    # B = d²delta(r_p)/dkdn = (delta(n + h/2, k + h/2) - delta(n + h/2, k - h/2) - delta(n - h/2, k + h/2)  + delta(n - h/2, k - h/2))/h²
    C = B #Satz von Schwarz
    D =  (delta_of_r(n, k + h, H_0_measured, freq, Material_parameter) - 2*delta_of_r(n, k, H_0_measured, freq, Material_parameter) - delta_of_r(n, k - h, H_0_measured, freq, Material_parameter))/h**2
    # D = d²delta(r_p)/dk² = (delta(n, k + h) - 2*delta(n,k) - delta(n,k - h))/h²
    hessian_tot = np.array([[A[0],B[0]], [C[0],D[0]]])
    with open('build/testing/hessian.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(hessian_tot) 
    return hessian_tot

def newton_r_p(r_p, H_0_measured, freq, Material_parameter, h=10**(-6)): #newton iteration step to find the best value of r=(n_2,k_2)  
    A = hessian(r_p[0], r_p[1], H_0_measured, freq, Material_parameter, h) 
    grad_ = grad(r_p[0], r_p[1], H_0_measured, freq, Material_parameter, h)
    with open('build/testing/gradient.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(grad_) 
    #print("The shape of r_p ", np.shape(r_p),"The shape of the gradient ",np.shape(grad_), "The shape of the hessian ", np.shape(A))
    #print(np.linalg.inv(A).dot(grad_))
    r_p_1 = r_p - np.linalg.inv(A).dot(grad_)
    F_old = r_p - np.linalg.inv(A).dot(grad_) 
    F_new = r_p_1 - np.linalg.inv(hessian(r_p_1[0], r_p_1[1], H_0_measured, freq, Material_parameter, h)).dot(grad(r_p_1[0], r_p_1[1], H_0_measured, freq, Material_parameter, h))
    lambda_ = 0.95
    while(np.abs(F_old).dot(np.abs(F_old)) < np.abs(F_new).dot(np.abs(F_new))):
        #print("sus der Gradient ist größer geworden", end="\r")
        r_p_1 = r_p - lambda_ * (np.linalg.inv(A).dot(grad_))
        F_new = r_p_1 - np.linalg.inv(hessian(r_p_1[0], r_p_1[1], H_0_measured, freq, Material_parameter, h)).dot(grad(r_p_1[0], r_p_1[1], H_0_measured, freq, Material_parameter, h))
        lambda_ = lambda_ - 0.05
        if(lambda_ <= 0.1):
            break

    # r_p+1 = r_p - A⁽⁻¹⁾*grad(delta(r_p))
    #print(np.shape(r_p_1))   
    return r_p_1 # returns new values for [n_2,k_2] that minimize the error according to newton iteration step 

def Transfer_function(omega, n, k, l, fp):
    T = 4*n/(n + 1)**2 * np.exp(k*(omega*l/c)) * np.exp(-1j * (n - 1) *(omega*l/c))
    #if np.isinf(T) or np.isnan(T):
    #    print("Input values for T give rise to value error: n = ", n, " k = ", k, " omgea = ", omega)
    if(fp):
        FP = 1/(1 - ((1 - (n+1j*k))/(1 + (n + 1j*k)))**2 * np.exp(-2*1j * (n + 1j *k)* (omega*l/c)))
        return T*FP
    else:
        print("No FP factor")
        return T

def Transfer_function_three_slabs(omega, n_1_real, n_2_real, n_3_real, k_1, k_2, k_3, l, fp):
    n_1 = n_1_real + 1j*k_1
    n_2 = n_2_real + 1j*k_2 
    n_3 = n_3_real + 1j*k_3
    T = (2*n_2*(n_1 + n_3)/((n_2 + n_1) * (n_2 + n_3))) * np.exp(-(1j*n_2 - 1j*n_3) * omega*l/c)
    if(fp):
        FP = 1/(1 - (((n_2 - n_1)/(n_2 + n_1) * (n_2 - n_3)/(n_2 + n_3)) * np.exp(-2 * 1j*n_2 * omega*l/c)))
        return T*FP
    else:
        return T
###################################################################################################################################
#       All data is read in, in this block.
#       Necessary data are
#       - the time resolved THz measurment
#       - the thickness of the probe
###################################################################################################################################


# The thickness of the probe

d = 0.26*10**(-3) # thickness of the probe in SI
n_air = 1
n_slab = 1
k_slab = 1

Material = Material_parameters(d = d, n_1=n_air, k_1=n_air, n_3=n_slab, k_3=k_slab)

#Read the excel file
#material_properties_ref = np.genfromtxt('data/teflon_1_material_properties.txt')
data_sam = np.genfromtxt('data/ZnTe/ZnTe_purge.txt', delimiter="	", comments="#") # The time resolved dataset of the probe measurment
data_ref = np.genfromtxt('data/ZnTe/reference_purge.txt',  delimiter="	", comments="#") # the time resolved dataset of the reference measurment


###################################################################################################################################
# This block plots the raw dataset
###################################################################################################################################
ones = np.ones(10000)

data_ref[:,0] = data_ref[:,0] * 10**(-12) # ps in seconds
data_sam[:,0] = data_sam[:,0] * 10**(-12)

data_ref[:,0] = data_ref[:,0] + np.abs(np.min(data_ref[:,0])) # move everything to positiv times
data_sam[:,0] = data_sam[:,0] + np.abs(np.min(data_sam[:,0]))

#plot the intensity against time delay
plt.figure()
plt.plot(data_ref[:,0]*10**(12), data_ref[:,1], label='Reference')
#plt.vlines(data_ref[peaks_inref,0]*10**12, ymin=0, ymax=-10)
plt.plot(data_sam[:,0]*10**(12) + 100, data_sam[:,1], label='Sample moved by +100ps')
plt.xlabel(r'$ t/ps $')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.title('The reference and sample data set of silicon')
plt.savefig('build/THz_timedomain.pdf')
plt.close()


##################################################################################################################################
# Zero_padding
##################################################################################################################################
timestep = np.abs(data_ref[:,0][2]-data_ref[:,0][3]) # minimum time resolution
N = len(data_ref[:,0]) #number of total data points

num_zeros = 1500

peak_ref,prop = find_peaks(data_ref[:,1], prominence=0.3) # finds the highest peak in the dataset and returns its index
peak_ref = peak_ref[0:2]
peak_sam,prop = find_peaks(data_sam[:,1], prominence=0.3)
peak_sam = peak_sam[0:2]
peak_sam[1] = peak_sam[1] - 100 # we assume that we cut off the array 50 steps before we hit the post pulse

data_ref_zero = [np.append(data_ref[:peak_sam[1], 0], np.linspace(data_ref[peak_sam[1], 0], data_ref[peak_sam[1], 0]+num_zeros*timestep, num_zeros)),
                 np.append(data_ref[:peak_sam[1], 1], (np.zeros(num_zeros)))]
data_sam_zero = [np.append(data_sam[:peak_sam[1], 0], np.linspace(data_sam[peak_sam[1], 0], data_sam[peak_sam[1], 0]+num_zeros*timestep, num_zeros)),
                 np.append(data_sam[:peak_sam[1], 1], (np.zeros(num_zeros)))]

print(len(data_ref_zero[0]))

##################################################################################################################################
# Plot zero padding in time domain
##################################################################################################################################

plt.figure()
plt.plot(data_ref_zero[0]*10**(12), data_ref_zero[1], label='Reference')
#plt.vlines(data_ref[peaks_inref,0]*10**12, ymin=0, ymax=-10)
plt.plot(data_sam_zero[0]*10**(12) + 100, data_sam_zero[1], label='Sample moved by +100ps')
plt.xlabel(r'$ t/ps $')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.title('The reference and sample data set of silicon with zero padding')
plt.savefig('build/THz_timedomain_zero.pdf')
plt.close()


##################################################################################################################################
# Some necessary calculations on frequency and time resolution
##################################################################################################################################

Delta_f = 1/(N*timestep) #frequency resolution
print('Delta t = ', " ",timestep/10**(-12), "ps")
print("T ", N*timestep/10**(-12), "ps")
print("Delta f = ", " ", Delta_f*10**(-12), "THz")

peak_ref,prop = find_peaks(data_ref[:,1], prominence=1) # finds the highest peak in the dataset and returns its index
peak_ref = peak_ref[0]
peak_sam,prop = find_peaks(data_sam[:,1], prominence=0.8) ### hier hat steven gefuscht !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
peak_sam = peak_sam[0]

t_peak_ref = data_ref[peak_ref,0] # the time value of the highest peak in the dataset
t_peak_sam = data_sam[peak_sam,0]
###################################################################################################################################
# This block applies the FFT to the data, aswell as masking frequencies that we dont need for the analization
###################################################################################################################################

freq_ref, amp_ref = FFT_func(data_ref[:,1], data_ref[:,0])  #in Hz
freq_sam, amp_sam = FFT_func(data_sam[:,1], data_sam[:,0])

mask1 = freq_ref < 4.5*10**12 # mask1ed for THz frequency below 4.5 THz
amp_ref = amp_ref[mask1]
amp_sam = amp_sam[mask1]
freq_ref = freq_ref[mask1]
freq_sam = freq_sam[mask1]

###################################################################################################################################
# This block applies the FFT to the zero padded data, aswell as masking frequencies that we dont need for the analization
###################################################################################################################################
freq_ref_zero, amp_ref_zero = FFT_func(data_ref_zero[1], data_ref_zero[0])  #in Hz
freq_sam_zero, amp_sam_zero = FFT_func(data_sam_zero[1], data_sam_zero[0])

mask1_zero = freq_ref_zero < 4.5*10**12 # mask1_zero masks for THz frequency below 4.5 THz
amp_ref_zero = amp_ref_zero[mask1_zero]
amp_sam_zero = amp_sam_zero[mask1_zero]
freq_ref_zero = freq_ref_zero[mask1_zero]
freq_sam_zero = freq_sam_zero[mask1_zero]

##########################################################################################
# This block plots the FFT
##########################################################################################

plt.figure()
plt.plot(freq_ref* 10**(-12), np.abs(amp_ref), label='Reference FFT') # plot in Thz
plt.plot(freq_sam* 10**(-12), np.abs(amp_sam), label='Sample FFT')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('Spectral Amplitude')
plt.legend()
plt.grid()
plt.title('The FFT of the data sets of silicon')
plt.savefig('build/THz_FFT.pdf')
plt.close()

##########################################################################################
# This block plots the FFT of the zerp padded data
##########################################################################################

plt.figure()
plt.plot(freq_ref_zero* 10**(-12), np.abs(amp_ref_zero), label='Reference FFT') # plot in Thz
plt.plot(freq_sam_zero* 10**(-12), np.abs(amp_sam_zero), label='Sample FFT')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('Spectral Amplitude')
plt.legend()
plt.grid()
plt.title('The FFT of the zero padded data sets of silicon')
plt.savefig('build/THz_FFT_zero.pdf')
plt.close()


###################################################################################################################################
# This block calculates the complex transfer function and does the unwrapping porcess
# It also plots the angle and phase of the transferfunction
###################################################################################################################################

H_0_value = H_0(amp_ref, amp_sam) # complex transfer function

angle = np.angle(H_0_value) #angle between complex numbers
phase = np.unwrap(angle)  #phase 

#phase_dif = unwrapping_alt(amp_ref, amp_sam, freq_ref)

# Note that phase and phase_dif should be equal. However, there are not. So either the paper is wrong. Or I am doing something wrong

plt.figure()
#plt.plot(freq_ref*10**(-12),phase_dif, label='phase differenz')
plt.plot(freq_ref*10**(-12),angle, label='angle directly from H_0')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\Phi$')
plt.legend()
plt.grid()
plt.title('The phase unwarpped')
plt.savefig('build/THzPhase.pdf')
plt.close()

###################################################################################################################################
# This block calculates the complex transfer function and does the unwrapping porcess
# It also plots the angle and phase of the transferfunction
###################################################################################################################################

H_0_value_zero = H_0(amp_ref_zero, amp_sam_zero) # complex transfer function

angle_zero = np.angle(H_0_value_zero) #angle between complex numbers
phase_zero = np.unwrap(angle_zero)  #phase 

#phase_dif = unwrapping_alt(amp_ref, amp_sam, freq_ref)

# Note that phase and phase_dif should be equal. However, there are not. So either the paper is wrong. Or I am doing something wrong

plt.figure()
#plt.plot(freq_ref_zero*10**(-12),angle_zero, label='angle')
plt.plot(freq_ref_zero*10**(-12),phase_zero, label='phase directly from H_0')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\Phi$')
plt.legend()
plt.grid()
plt.title('The phase unwarpped of the zero padded data of silicon')
plt.savefig('build/THzPhase_zero.pdf')
plt.close()


###################################################################################################################################
#   This block calculates the real and complex part of the refractive index
###################################################################################################################################

n_real = n(freq_ref, d, phase)
#n_real_alt = n(freq_ref, d, phase_dif)
n_im = k(freq_ref, d, H_0_value, n_real)

###################################################################################################################################
#   This block calculates the real and complex part of the refractive index
###################################################################################################################################

n_real_zero = n(freq_ref_zero, d, phase_zero)
#n_real_alt = n(freq_ref, d, phase_dif)
n_im_zero = k(freq_ref_zero, d, H_0_value_zero, n_real_zero)


###################################################################################################################################
#   This block plots the real part of the refractive index
###################################################################################################################################

plt.figure()
#plt.plot(freq_ref*10**(-12), n(freq_ref, d, phase_dif), label='real part of refractive index calculated by the alternative unwrap')
plt.plot(freq_ref*10**(-12), n_real, label='real part of refractive index')
#plt.plot(material_properties_ref[10:,0], material_properties_ref[10:,1], c='k', label='refference n from PCA program')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('n (arb.)')
plt.xlim(0.18, 4.5)
#plt.ylim(0,2)
plt.legend()
plt.grid()
plt.title('The real part of the refractive index of silicon')
plt.savefig('build/THz_real_index.pdf')
plt.close()

###################################################################################################################################
#   This block plots the real part of the refractive index
###################################################################################################################################

plt.figure()
#plt.plot(freq_ref*10**(-12), n(freq_ref, d, phase_dif), label='real part of refractive index calculated by the alternative unwrap')
plt.plot(freq_ref_zero*10**(-12), n_real_zero, label='real part of refractive index')
#plt.plot(material_properties_ref[10:,0], material_properties_ref[10:,1], c='k', label='refference n from PCA program')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('n (arb.)')
plt.xlim(0.18, 4.5)
#plt.ylim(0,4)
plt.legend()
plt.grid()
plt.title('The real part of the refractive index from zero padded data of silicon')
plt.savefig('build/THz_real_index_zero.pdf')
plt.close()


##################################################################################################
#   This block plots the complex part of the refractive index
###################################################################################################################################

plt.figure()
#plt.plot(freq_ref*10**(-12), k(freq_ref, d, H_0_value, n_real_alt), label='complex part of refractive index by alt')
plt.plot(freq_ref*10**(-12), n_im, label='complex part of refractive index')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('k (arb.)')
plt.xlim(0.18, 4.5)
#plt.ylim(0,2000)
plt.legend()
plt.grid()
plt.title('The complex part of the refractive index of silicon')
plt.savefig('build/THz_complex_index.pdf')
plt.close()

##################################################################################################
#   This block plots the complex part of the refractive index
###################################################################################################################################

plt.figure()
#plt.plot(freq_ref*10**(-12), k(freq_ref, d, H_0_value, n_real_alt), label='complex part of refractive index by alt')
plt.plot(freq_ref_zero*10**(-12), n_im_zero, label='complex part of refractive index')
plt.xlabel(r'$ \omega/THz $')
plt.ylabel('k (arb.)')
plt.xlim(0.18, 4.5)
#plt.ylim(0,2000)
plt.legend()
plt.grid()
plt.title('The complex part of the refractive index from zero padded data of silicon')
plt.savefig('build/THz_complex_index_zero.pdf')
plt.close()

###################################################################################################################################
# This block calculates the absorption coefficient and plots it
###################################################################################################################################

alpha = 2*freq_ref *n_im/c 

plt.figure()
plt.plot(freq_ref*10**(-12), alpha/100, label='Absorption coefficient')
#plt.plot(material_properties_ref[10:,0], material_properties_ref[10:,2], c='k', label='refference k from PCA program')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\alpha$')
plt.xlim(0.18, 4.5)
#plt.ylim(0, 500)
plt.legend()
plt.grid()
plt.title('The absorption coeffiecient of silicon')
plt.savefig('build/THz_absorption.pdf')
plt.close()

###################################################################################################################################
# This block calculates the absorption coefficient for zero padding and plots it
###################################################################################################################################

alpha_zero = 2*freq_ref_zero*n_im_zero/c

plt.figure()
plt.plot(freq_ref_zero*10**(-12), alpha_zero/100, label='Absorption coefficient')
#plt.plot(material_properties_ref[10:,0], material_properties_ref[10:,2], c="k",label='refference k from PCA program')
#plt.plot(data_ref[:,0], filter_ref)
plt.xlabel(r'$ \omega/THz $')
plt.ylabel(r'$\alpha$')
plt.xlim(0.18, 4.5)
#plt.ylim(0, 500)
plt.legend()
plt.grid()
plt.title('The absorption coeffiecient of silicon for zero padded data')
plt.savefig('build/THz_absorption_zero.pdf')
plt.close()

###################################################################################################################################

###################################################################################################################################
# Here Starts the numerical process of finding the refractive index
###################################################################################################################################

###################################################################################################################################
""" For an actual calculation of n and k we would want to do it over the whole measured frequency range.
    However, this will used up way too much calculation power for my litle laptop so I start with just one frequency.
    If I get that stuff right I move on to the whole frequency range."""

print("-----------------------------------------------------------")

""""Okay ich glaub was ich eigentlich machen muss ist ein startwert zu wählen.
    Den nutze ich um eine Abweichung also delta zu berechnen.
    Mit dem Wert mache ich zudem dann den Newton schritt.
    Die Hesse Matrix berechne ich dabei an dem entsprechenden Punkt.
    Jetzt ist die Frage nur noch wie ich die Hesse Matrix berechne """


""" Die Transferfunktion gibt mir irgendwie keine sinnvollen werte zurück also plotte ich die erstmal für ein paar testwerte
    komischer weise scheint sie gegen hohe frequezen zu divergieren.
    
    Dieses Problem scheint zu verschwinden wenn ich die Fabry perot faktoren beachte.
    Allerdings stoße ich weiterhin auch divergenzen für bestimmte kombinationen aus n,k und omega."""

plt.figure()
plt.plot(freq_ref, Transfer_function_three_slabs(freq_ref, 1 , 2, 1, 1, 2, 1, d, True).real, label='Transferfunction real part')
plt.plot(freq_ref, Transfer_function_three_slabs(freq_ref, 1 , 2, 1, 1, 2, 1, d, True).imag, label='Transferfunction imag part')
plt.legend()
plt.savefig('build/testing/Transferfunction_n_1_5_k_1_5.pdf')
plt.close()

n_0 = 1
k_0 = 1

test_freq_index = 20

###################################################################################################################################
    # Minimizing with newton
###################################################################################################################################
steps = 50000

r_0 = np.array([n_0,k_0]) # r_p[0] = n, r_p[1] = k
r_p = r_0 # set the start value
r_p_1 = np.zeros((steps,2))
i = 1
h = 0.1
Transfer_function_values = np.zeros(steps)
for l in range(steps - 1):
    r_p_1[i] = newton_r_p(r_p, H_0_value[test_freq_index], freq_ref[test_freq_index], Material, h = h) # r_p[0] = n, r_p[1] = k
    Transfer_function_values[i] = np.abs(Transfer_function_three_slabs(freq_ref[test_freq_index], Material.n_1, r_p[0], Material.n_3, Material.k_1, r_p[1], Material.k_3, Material.d, fp=False))**2
    if(r_p_1[i][0] == r_p_1[i - 1][0] or r_p_1[i][1] == r_p_1[i -1][1]): # if the algorythm brakes we go back two steps and try again with smaller h
        h = h/10
        i = i - 2
    r_p = r_p_1[i]
    print(f"{l/steps * 100:.2f} % {r_p}", end="\r")
    i = i + 1 

plt.figure
plt.plot(np.linspace(1,steps - 2, steps - 3), r_p_1[1:steps-2,0], label='n')
plt.plot(np.linspace(1,steps - 2, steps - 3), r_p_1[1:steps-2,1], label='k')
plt.title(label="Convergence of the parameters over " + str(steps) + " steps")
plt.legend()
plt.savefig('build/testing/convergence_tes.pdf')
plt.close()

plt.figure
plt.plot(np.linspace(1,steps, steps), Transfer_function_values, label='T')
plt.title(label="Convergence of the parameters over " + str(steps) + " steps")
plt.legend()
plt.savefig('build/testing/Transferfunction_test.pdf')
plt.close()


"""Things that dont work:
    - Transferfunction gives weird values and is divergent for some inputs
    - Delta plot should look like a paraboloid, however it is just a parabel. So one dimension is missing
    - I have no idea how to calculate the matrix A that is necessary for the newton raphson methode
        it should be the hessian matrix of the taylor expansion of delta.
        However, I dont know how to calculate neither the taylor expansion of delta nor the hessian matrix as my function is basically just a matrix.
        I could try to take the numerical gradient of the matrox two times, but that does not feel right, as I also could just search the array for the lowest number or not?
    - If I would know how to calculate A the progamm is basically done"""