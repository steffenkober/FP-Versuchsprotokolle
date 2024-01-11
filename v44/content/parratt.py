#Startwerte
lam = 1.54e-10
k=2*np.pi/lam
n1=1
d1=0
#Algorithmus
def Parrattalgorithmus(alpha, delta2, delta3, kappa2, kappa3, sigma1, sigma2, d2):
    alpha = np.deg2rad(aalpha)
    n2 = 1.0 - delta2 - kappa2*1j
    n3 = 1.0 - delta3 - kappa3*1j
    
    kd1 = k *  np.sqrt(n1**2 - np.cos(alpha)**2)
    kd2 = k * np.sqrt(n2**2 - np.cos(alpha)**2)
    kd3 = k * np.sqrt(n3**2 - np.cos(alpha)**2)

    r12 = ((kd1 - kd2)/(kd1 + kd2))*np.exp(-2*kd1*kd2*sigma1**2)
    r23 = ((kd2 - kd3)/(kd2 + kd3))*np.exp(-2*kd2*kd3*sigma2**2)

    x2 = np.exp(-2j* kd2 * d2) * r23
    x1 = (r12 + x2)/(1+ r12*x2)

    return np.abs(x1)**2