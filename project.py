# %% COPYRIGHT
"""
Spyder Editor
This is a temporary script file.
author: KAGAN 

"""
# %% LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d

# %%MAIN DIMENSIONS
L_r=232.5
B_r=32.2
T=10.8
disp=53910.55
Cb = disp / (L_r*B_r*T)
KG=7.28
GM=0.6
beta = np.pi/6
scale=16
g=9.81
Hs=2.3
Tz=8.5
V_m = 3 #m/s
D = 1.5 * T
# %% PARAMETERS
x_1 = np.array([0.25,0.4,0.5,0.75,0.9,1,1.25,1.5,1.75,2])
x_2 = np.array([0.08,0.24,0.56,0.95,1.22,1.47,1.33,1.15,1.09,1.02])
x_3 = np.array([0.06,0.37,0.56,0.74,1.19,1.36,1.18,1.1,1.05,1.03])

# %% INTERPOLATION AND SPECTRUM CALCULATIONS
f_1 = interp1d(x_1,x_2)
f_2 = interp1d(x_1,x_3)
x_1_new = np.linspace(0.25,2,100)
x_2_new = f_1(x_1_new)
x_3_new = f_2(x_1_new)
# Lambda Calculation
lamda = L_r * x_1_new
k = 2*np.pi / lamda
#ITTC Spectrum Calculation
A= (123.8*Hs**2)/(Tz**4)
B= 495/(Tz**4)
w= (g*k)**(0.5)
we = w - ( (w**2/g) * V_m*np.cos(beta))
S_ittc = (A / w**5 ) * np.exp(-B/w**4)
eta_3 = x_2_new*A
eta_5 = x_3_new*k*A
# Checkpoint-1
if True:
    m = []
    for i in range(6):
        m.append(np.abs(integrate.simpson(S_ittc*w**i,w)))
    c_1 = 4*(np.abs(m[0])**(0.5))
d_w=(1-(2*w*V_m*np.cos(beta)/g))
S_we = S_ittc / d_w
# Checkpoint-2
if True:
    m_we = []
    for i in range(6):
        m_we.append(np.abs(integrate.simpson(S_we*we**i,we)))
    c_2 = 4*(np.abs(m_we[0])**(0.5))
# %% RMS CALCULATIONS
delta_we=[]
for i in range(len(we)-1):
    delta_we.append(we[i]-we[i+1])
delta_we = np.array(delta_we)
h_3 = S_we*x_2_new**2
h_5 = S_we*x_3_new**2*k**2

RMS_3=np.zeros(100)
RMS_3[0]=h_3[0]
RMS_3[1:100]=list(h_3[1:100]*delta_we)
RMS_3_val=np.sqrt(np.sum(RMS_3))

RMS_5=np.zeros(100)
RMS_5[0]=h_5[0]
RMS_5[1:100]=list(h_5[1:100]*delta_we)
RMS_5_val=np.sqrt(np.sum(RMS_5))

h_3_var = S_ittc*x_2_new**2
h_5_var = S_ittc*x_3_new**2*k**2

delta_w=[]
for i in range(len(w)-1):
    delta_w.append(w[i]-w[i+1])

RMS_3_var=np.zeros(100)
RMS_3_var[0]=h_3_var[0]
RMS_3_var[1:100]=list(h_3_var[1:100]*delta_w)
RMS_3_val_var=np.sqrt(np.sum(RMS_3_var))

RMS_5_var=np.zeros(100)
RMS_5_var[0]=h_5_var[0]
RMS_5_var[1:100]=list(h_5_var[1:100]*delta_w)
RMS_5_val_var=np.sqrt(np.sum(RMS_5_var))

# %% ACCELERATION AT GG
k_r = 0.39*B_r
T_roll = 2.3*np.pi*k_r / np.sqrt(g*GM)
f_bk = 1 # Ship with bilge keel
f_ps = 0.8
f_r = 1
f_t = 0.5
f_p_roll = f_r * (0.23-(4*f_t*B_r*10**-4))
teta_roll = 9000* (1.4-0.035*T_roll)*f_p_roll*f_bk / ((1.15*B_r+55)*np.pi)
a_0 = (1.58 - 0.47*Cb)*((2.4/np.sqrt(L_r))+(34/L_r)-(600/L_r**2))
f_p_heave = f_r * ((0.27+ 0.02*f_t)-17*L_r*10**-5)
a_heave=(1.15-(6.5/np.sqrt(g*L_r)))*f_p_heave*a_0*g
a_roll = (f_p_roll *teta_roll*np.pi/180)*(2*np.pi/T_roll)**2
f_p_pitch = f_r*((0.28-(5+6*f_t)*L_r*10**-5))
lam_pitch = 0.6*(1+f_t)*L_r
T_pitch = np.sqrt(2*np.pi*lam_pitch/g)
a_pitch = f_p_pitch*(1.75-(22/np.sqrt(g*L_r)))*RMS_5_val*np.pi/180*(2*np.pi/T_pitch)**2

a_gg = m_we[4]*we*delta_we


# %% ACCELERATION AT BOW
# Vertical
x = L_r / 2
y = B_r/2
a_pitch_z = a_pitch*(1.08*x-0.45*L_r)
a_roll_z = a_roll * y
a_vertical = (a_heave**2+((0.95+np.exp(-L_r/15)*a_pitch_z))**2+(1.2*a_roll_z)**2)**0.5
# Lateral
R = min(((D/4)+(T/2)),D/2)
z = D / 2
a_roll_y = a_roll * (z-R)
a_lateral = (1-np.exp(-(B_r*L_r/213*GM)))*(((g*np.sin(np.pi/30))+a_roll_y)**2)**0.5


# %%PLOTS
plt.plot(w,we)
plt.xlabel('W(rad/s)')
plt.ylabel('We(rad/s)')
plt.grid()
#plt.savefig('W-We.jpg',dpi=1500)
plt.show()

plt.plot(we,h_3,label='Heave')
plt.plot(we,h_5*1000,label='Pitch(scale=1000)')
plt.xlabel('We')
plt.grid()
plt.legend()
#plt.savefig('Heave',dpi=1500)
plt.show()

plt.plot(w,S_ittc,label='S(W)')
plt.plot(w,x_2_new,label='RAO Heave')
plt.plot(w,x_3_new,label='RAO Pitch')
plt.title('W-S(W) Graph')
plt.xlabel('W')
plt.legend()
plt.grid()
#plt.savefig('normalfrekans.jpg',dpi=1500)
plt.show()

plt.plot(we,S_we,label='S(We)')
plt.plot(we,x_2_new,label='RAO Heave')
plt.title('We-S(We) Graph')
plt.plot(we,x_3_new,label='RAO Pitch')
plt.xlabel('We')
plt.legend()
plt.grid()
#plt.savefig('karşılaşma.jpg',dpi=1500)
plt.show()


