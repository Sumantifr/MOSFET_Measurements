# Extraction of the doping concentration using MOSFETs
# Author Ioannis
# Date 02/03/17
# This simple programme calculates the Drain-Source current vs Gate voltage for a constant Drain-Source voltage.
# Description of the mobility behavior on the nMOSFET
# Polunomial model Ghibaudo  
# Output fitted parameters important for Martin's BSc thesis
# MOSFET device produced from Hamamatsu, MCz material, orientation <100>,
# doping Nd=4.4x10**(12) cm**(-3), insulator 700 nm SiO2
# The measurements performed on the probe station, without humidity control and temperature ~ 20.0 degrees Celsius
# Ghibaudo mobility model and fit

from numpy import *
from matplotlib.pyplot import *
from matplotlib import *
from matplotlib.font_manager import *
from matplotlib.pyplot import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import OldScalarFormatter
import numpy as np
import matplotlib.pyplot as plt
import pylab
from math import exp
import scipy as sp
from scipy.optimize import curve_fit
import scipy.optimize as optimization
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#MARKERS = ["-", "o-.", "s--", "^:", "v--", "s--", "p--", "*--", "o--", ">--"]
DOTS = ["o", "s", "^","p", "*", ">", "<"]
LINES = ["-", "-.", "--", ":"]
COLORS = ["k", "r", "b", "g","y", "c", "m", "k", "r"]

# Fonts setup
#For UBUNTU-32 bits
FONT = "Arial"
# For UBUNTU-64 bits
fp = FontProperties(family="Arial", fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")

FILE1 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_11.iv"
FILE2 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_12.iv"
FILE3 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_13.iv"
FILE4 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_14.iv"
FILE5 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_15.iv"
FILE6 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_16.iv"
FILE7 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_17.iv"
FILE8 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_18.iv"
FILE9 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_19.iv"
FILE10 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_20.iv"
FILE11 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_21.iv"
FILE12 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_22.iv"
FILE13 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_23.iv"
FILE14 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_24.iv"
FILE15 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_25.iv"
FILE16 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_26.iv"
FILE17 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_27.iv"
FILE18 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_28.iv"
FILE19 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_29.iv"
FILE20 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_30.iv"
FILE21 = "M200P_04_HPKTS_1_MOSFET_2017-01-16_31.iv"
FILE22 = "M200P_04_HPKTS_1_MOSFET_2017-01-16_32.iv"
FILE23 = "M200P_04_HPKTS_1_MOSFET_2017-01-16_33.iv"
FILE24 = "M200P_04_HPKTS_1_MOSFET_2017-01-16_34.iv"
FILE25 = "M200P_04_HPKTS_1_MOSFET_2017-01-16_35.iv"


##############    Constants used for calculations     ###################
epsilon0 = 8.85418781762e-14                   # Vaccuum permitivity, unit:F/cm
epsilonSi02 = 3.9                              # Dielectric permittivity of the SiO2 
epsilonSi = 11.9                               # Dielectric permittivity of the Silicon  
d = 700e-7                                     # Oxide thickness, unit:cm

Radius1 = (99.0)*1e-6                          # Small gate radius, unit:m
Radius2 = (351.0)*1e-6                         # Large gate radius, unit:m

T = 293.0                                      # Temperature during the measurements, unit:K
q0 = 1.60217657e-19                            # Elementary charge, unit:Cb
Nd = 4.39770462776e12                                    # Bulk doping concentration, unit:cm^(-3)
k0 = 1.3806488e-23                             # Boltzmann constant, unit:J/K 
k10 = 8.6173324e-5                             # Boltzmann constant, unit:eV/K    
eV = 1.60217657e-19                            # 1 electron volt, unit J
fm = 4.1                                       # Metal work function, unit:V, from Sze
###########################################################################

Vds = 0.05                                     # V Drain-Source Voltage

###########################################################################

# Intrinsic carrier density Schroeder
Ni = 5.29*10**19*(T/300.0)**2.54*np.exp(-6726.0/T)     
print "The intrinsic carrier density is:", Ni*10**(-9), "e9 1/cm^3."

# Bandgap temperature dependence from TCAD manual
Eg = 1.1696 - ((4.73e-4)*T**2/(T+636.0))
print "The bandgap is:", Eg , "eV."

# Electron affinity temperature dependence implemented from TCAD manual (BennettWilson model)
x = 4.05 + (((4.73e-4)*T**2)/(2*(T+636.0))) + 0.5*6.84e-3
print "The electron affinity is:", x, "V."

# Oxide capacitance per area
Cox=(epsilon0*epsilonSi02)/d
print "The oxide capacitance per area is:", Cox, "F/cm^2."

# Debye length
L_b = sqrt((epsilon0*epsilonSi*k10*T)/(q0*Nd))   
print "The debye length is:", L_b, "cm."

# Geometrical coefficient
Coeff_circ = 2.0*pi*(1.0/log(Radius2/Radius1))
print "The W/L ratio circ of the channel is:", Coeff_circ,"."

# Const factor is
fac_circ = Cox*Coeff_circ*Vds
print "The factor circ is:", fac_circ, "."

# 1_Const factor is
Rev_fac_circ = 1.0/(Cox*Coeff_circ*Vds)
print "The reverse factor circ is:", Rev_fac_circ, "."

#TITLE = ""
#LABEL = ""

MAT_DATA1 = loadtxt(FILE1, skiprows = 37, comments = "END")
MAT_DATA2 = loadtxt(FILE2, skiprows = 37, comments = "END")
MAT_DATA3 = loadtxt(FILE3, skiprows = 37, comments = "END")
MAT_DATA4 = loadtxt(FILE4, skiprows = 37, comments = "END")
MAT_DATA5 = loadtxt(FILE5, skiprows = 37, comments = "END")
MAT_DATA6 = loadtxt(FILE6, skiprows = 37, comments = "END")
MAT_DATA7 = loadtxt(FILE7, skiprows = 37, comments = "END")
MAT_DATA8 = loadtxt(FILE8, skiprows = 37, comments = "END")
MAT_DATA9 = loadtxt(FILE9, skiprows = 37, comments = "END")
MAT_DATA10 = loadtxt(FILE10, skiprows = 37, comments = "END")
MAT_DATA11 = loadtxt(FILE11, skiprows = 37, comments = "END")
MAT_DATA12 = loadtxt(FILE12, skiprows = 37, comments = "END")
MAT_DATA13 = loadtxt(FILE13, skiprows = 37, comments = "END")
MAT_DATA14 = loadtxt(FILE14, skiprows = 37, comments = "END")
MAT_DATA15 = loadtxt(FILE15, skiprows = 37, comments = "END")
MAT_DATA16 = loadtxt(FILE16, skiprows = 37, comments = "END")
MAT_DATA17 = loadtxt(FILE17, skiprows = 37, comments = "END")
MAT_DATA18 = loadtxt(FILE18, skiprows = 37, comments = "END")
MAT_DATA19 = loadtxt(FILE19, skiprows = 37, comments = "END")
MAT_DATA20 = loadtxt(FILE20, skiprows = 37, comments = "END")
MAT_DATA21 = loadtxt(FILE21, skiprows = 37, comments = "END")
MAT_DATA22 = loadtxt(FILE22, skiprows = 37, comments = "END")
MAT_DATA23 = loadtxt(FILE23, skiprows = 37, comments = "END")
MAT_DATA24 = loadtxt(FILE24, skiprows = 37, comments = "END")
MAT_DATA25 = loadtxt(FILE25, skiprows = 37, comments = "END")

# Plot for voltage
Vol1 = MAT_DATA1[:,0]
Vol2 = MAT_DATA2[:,0]
Vol3 = MAT_DATA3[:,0]
Vol4 = MAT_DATA4[:,0]
Vol5 = MAT_DATA5[:,0]
Vol6 = MAT_DATA6[:,0]
Vol7 = MAT_DATA7[:,0]
Vol8 = MAT_DATA8[:,0]
Vol9 = MAT_DATA9[:,0]
Vol10 = MAT_DATA10[:,0]
Vol11 = MAT_DATA11[:,0]
Vol12 = MAT_DATA12[:,0]
Vol13 = MAT_DATA13[:,0]
Vol14 = MAT_DATA14[:,0]
Vol15 = MAT_DATA15[:,0]
Vol16 = MAT_DATA16[:,0]
Vol17 = MAT_DATA17[:,0]
Vol18 = MAT_DATA18[:,0]
Vol19 = MAT_DATA19[:,0]
Vol20 = MAT_DATA20[:,0]
Vol21 = MAT_DATA21[:,0]
Vol22 = MAT_DATA22[:,0]
Vol23 = MAT_DATA23[:,0]
Vol24 = MAT_DATA24[:,0]
Vol25 = MAT_DATA25[:,0]


Temp1 = mean(MAT_DATA1[:,1])
Temp2 = mean(MAT_DATA2[:,1])
Temp3 = mean(MAT_DATA3[:,1])
Temp4 = mean(MAT_DATA4[:,1])
Temp5 = mean(MAT_DATA5[:,1])
Temp6 = mean(MAT_DATA6[:,1])
Temp7 = mean(MAT_DATA7[:,1])
Temp8 = mean(MAT_DATA8[:,1])
Temp9 = mean(MAT_DATA9[:,1])
Temp10 = mean(MAT_DATA10[:,1])
Temp11 = mean(MAT_DATA11[:,1])
Temp12 = mean(MAT_DATA12[:,1])
Temp13 = mean(MAT_DATA13[:,1])
Temp14 = mean(MAT_DATA14[:,1])
Temp15 = mean(MAT_DATA15[:,1])
Temp16 = mean(MAT_DATA16[:,1])
Temp17 = mean(MAT_DATA17[:,1])
Temp18 = mean(MAT_DATA18[:,1])
Temp19 = mean(MAT_DATA19[:,1])
Temp20 = mean(MAT_DATA20[:,1])
Temp21 = mean(MAT_DATA21[:,1])
Temp22 = mean(MAT_DATA22[:,1])
Temp23 = mean(MAT_DATA23[:,1])
Temp24 = mean(MAT_DATA24[:,1])
Temp25 = mean(MAT_DATA25[:,1])


print "The measurement was performed at", Temp1, "degree C"
print "The measurement was performed at", Temp2, "degree C"
print "The measurement was performed at", Temp3, "degree C"
print "The measurement was performed at", Temp4, "degree C"
print "The measurement was performed at", Temp5, "degree C"
print "The measurement was performed at", Temp6, "degree C"
print "The measurement was performed at", Temp7, "degree C"
print "The measurement was performed at", Temp8, "degree C"
print "The measurement was performed at", Temp9, "degree C"
print "The measurement was performed at", Temp10, "degree C"
print "The measurement was performed at", Temp11, "degree C"
print "The measurement was performed at", Temp12, "degree C"
print "The measurement was performed at", Temp13, "degree C"
print "The measurement was performed at", Temp14, "degree C"
print "The measurement was performed at", Temp15, "degree C"
print "The measurement was performed at", Temp16, "degree C"
print "The measurement was performed at", Temp17, "degree C"
print "The measurement was performed at", Temp18, "degree C"
print "The measurement was performed at", Temp19, "degree C"
print "The measurement was performed at", Temp20, "degree C"
print "The measurement was performed at", Temp21, "degree C"
print "The measurement was performed at", Temp22, "degree C"
print "The measurement was performed at", Temp23, "degree C"
print "The measurement was performed at", Temp24, "degree C"
print "The measurement was performed at", Temp25, "degree C"

# Plot for current
Curr1 = MAT_DATA1[:,2]
Curr2 = MAT_DATA2[:,2]
Curr3 = MAT_DATA3[:,2]
Curr4 = MAT_DATA4[:,2]
Curr5 = MAT_DATA5[:,2]
Curr6 = MAT_DATA6[:,2]
Curr7 = MAT_DATA7[:,2]
Curr8 = MAT_DATA8[:,2]
Curr9 = MAT_DATA9[:,2]
Curr10 = MAT_DATA10[:,2]
Curr11 = MAT_DATA11[:,2]
Curr12 = MAT_DATA12[:,2]
Curr13 = MAT_DATA13[:,2]
Curr14 = MAT_DATA14[:,2]
Curr15 = MAT_DATA15[:,2]
Curr16 = MAT_DATA16[:,2]
Curr17 = MAT_DATA17[:,2]
Curr18 = MAT_DATA18[:,2]
Curr19 = MAT_DATA19[:,2]
Curr20 = MAT_DATA20[:,2]
Curr21 = MAT_DATA21[:,2]
Curr22 = MAT_DATA22[:,2]
Curr23 = MAT_DATA23[:,2]
Curr24 = MAT_DATA24[:,2]
Curr25 = MAT_DATA25[:,2]

################# Definitions of functions ###############################


###################### Ghibaudo fit function #############################
def Id_fit(Vg,a,b,Vt):
	Id_fit = a*(Vg-Vt)*(1.0/(1.0 + b*(Vg-Vt)))
	return Id_fit	
###########################################################################	


###################### Ghibaudo plot function #############################
def Id_plot(Vgate,Alpha,beta,V_th):
	Id_plot = Alpha*(Vgate-V_th)*(1.0/(1.0 + beta*(Vgate-V_th)))
	return Id_plot	
###########################################################################	


###################### Chi square function ################################
def Chi_sq(Cur_data,model,w):
	Chi_sq = sum(((Cur_data-model)/(Cur_data*w))**2)
	return Chi_sq
###########################################################################

######################  Relative error % ##################################
def Rel_er(Cur_data,model,w):
	Rel_er = (((Cur_data-model)/(Cur_data))*100.0)
	return Rel_er
###########################################################################

##################### Mobility function ###################################
def mu_circ(Vgate,Alpha,beta,V_th):
	mu_circ = (Alpha/(Coeff_circ*Cox*Vds))*(1.0/(1.0 + beta*(Vgate-V_th)))
	return mu_circ
###########################################################################

fb=((k0*T)/q0)*np.log(Nd/Ni)
print "The fb is:" ,fb, "V."

################### f_f function ##########################################

def f_f(N):
	f_f = ((k0*T)/q0)*log(Nd/Ni)
	return f_f
###########################################################################


######################## Vt function ######################################

def Vt(Vf, N, Vb):
	Vt = Vf + 2.0*f_f(N) + np.sqrt(N)*(np.sqrt(2.0*q0*epsilon0*epsilonSi*(2.0*f_f(N) + Vb))/Cox)
	return Vt

#print Vt(-1.8, Nd, 0.5)
###########################################################################

########### Plot for V_back = 0 V #############################

Min_plot_1 = 0                                      # Region of plot, lower limit
Max_plot_1 = 241                                    # Region of plot, upper limit
Vr_plot_1 = np.array(Vol1[Min_plot_1:Max_plot_1])   # Voltage region
Cur_plot_1 = np.array(Curr1[Min_plot_1:Max_plot_1]) # Current region

###############################################################

###### Optimization for V_back = 0 V ###########################

Min_1 = 38                                          # Region of fit, lower limit
Max_1 = 111                                         # Region of fit, upper limit
w_1 = 0.002                                         # Weight for the data fit
Vr_1 = np.array(Vol1[Min_1:Max_1])                  # Voltage region
Cur_1 = np.array(Curr1[Min_1:Max_1])                # Current region             
                                         
#################################################################


########### Plot for V_back = -5 V #############################

Min_plot_2 = 0                                      # Region of plot, lower limit
Max_plot_2 = 251                                    # Region of plot, upper limit
Vr_plot_2 = np.array(Vol2[Min_plot_2:Max_plot_2])   # Voltage region
Cur_plot_2 = np.array(Curr2[Min_plot_2:Max_plot_2]) # Current region

###############################################################

###### Optimization for V_back = -5 V ###########################

Min_2 = 48                                          # Region of fit, lower limit
Max_2 = 121                                         # Region of fit, upper limit
w_2 = 0.002                                         # Weight for the data fit
Vr_2 = np.array(Vol2[Min_2:Max_2])                  # Voltage region
Cur_2 = np.array(Curr2[Min_2:Max_2])                # Current region             
                                         
#################################################################


########### Plot for V_back = -10 V #############################

Min_plot_3 = 0                                      # Region of plot, lower limit
Max_plot_3 = 276                                    # Region of plot, upper limit
Vr_plot_3 = np.array(Vol3[Min_plot_3:Max_plot_3])   # Voltage region
Cur_plot_3 = np.array(Curr3[Min_plot_3:Max_plot_3]) # Current region

###############################################################

###### Optimization for V_back = -10 V ###########################

Min_3 = 73                                          # Region of fit, lower limit
Max_3 = 146                                         # Region of fit, upper limit
w_3 = 0.002                                         # Weight for the data fit
Vr_3 = np.array(Vol3[Min_3:Max_3])                  # Voltage region
Cur_3 = np.array(Curr3[Min_3:Max_3])                # Current region             
                                         
#################################################################


########### Plot for V_back = -15 V #############################

Min_plot_4 = 0                                      # Region of plot, lower limit
Max_plot_4 = 301                                    # Region of plot, upper limit
Vr_plot_4 = np.array(Vol4[Min_plot_4:Max_plot_4])   # Voltage region
Cur_plot_4 = np.array(Curr4[Min_plot_4:Max_plot_4]) # Current region

###############################################################

###### Optimization for V_back = -15 V ###########################

Min_4 = 98                                          # Region of fit, lower limit
Max_4 = 171                                         # Region of fit, upper limit
w_4 = 0.002                                         # Weight for the data fit
Vr_4 = np.array(Vol4[Min_4:Max_4])                  # Voltage region
Cur_4 = np.array(Curr4[Min_4:Max_4])                # Current region             
                                         
#################################################################

########### Plot for V_back = -20 V #############################

Min_plot_5 = 0                                      # Region of plot, lower limit
Max_plot_5 = 326                                    # Region of plot, upper limit
Vr_plot_5 = np.array(Vol5[Min_plot_5:Max_plot_5])   # Voltage region
Cur_plot_5 = np.array(Curr5[Min_plot_5:Max_plot_5]) # Current region

###############################################################

###### Optimization for V_back = -20 V ###########################

Min_5 = 123                                          # Region of fit, lower limit
Max_5 = 196                                         # Region of fit, upper limit
w_5 = 0.002                                         # Weight for the data fit
Vr_5 = np.array(Vol5[Min_5:Max_5])                  # Voltage region
Cur_5 = np.array(Curr5[Min_5:Max_5])                # Current region             
                                         
#################################################################

########### Plot for V_back = -25 V #############################

Min_plot_6 = 0                                      # Region of plot, lower limit
Max_plot_6 = 351                                    # Region of plot, upper limit
Vr_plot_6 = np.array(Vol6[Min_plot_6:Max_plot_6])   # Voltage region
Cur_plot_6 = np.array(Curr6[Min_plot_6:Max_plot_6]) # Current region

###############################################################

###### Optimization for V_back = -25 V ###########################

Min_6 = 148                                          # Region of fit, lower limit
Max_6 = 221                                         # Region of fit, upper limit
w_6 = 0.002                                         # Weight for the data fit
Vr_6 = np.array(Vol6[Min_6:Max_6])                  # Voltage region
Cur_6 = np.array(Curr6[Min_6:Max_6])                # Current region             
                                         
#################################################################

########### Plot for V_back = -30 V #############################

Min_plot_7 = 0                                      # Region of plot, lower limit
Max_plot_7 = 376                                    # Region of plot, upper limit
Vr_plot_7 = np.array(Vol7[Min_plot_7:Max_plot_7])   # Voltage region
Cur_plot_7 = np.array(Curr7[Min_plot_7:Max_plot_7]) # Current region

###############################################################

###### Optimization for V_back = -30 V ###########################

Min_7 = 173                                         # Region of fit, lower limit
Max_7 = 246                                         # Region of fit, upper limit
w_7 = 0.002                                         # Weight for the data fit
Vr_7 = np.array(Vol7[Min_7:Max_7])                  # Voltage region
Cur_7 = np.array(Curr7[Min_7:Max_7])                # Current region             
                                         
#################################################################


########### Plot for V_back = -35 V #############################

Min_plot_8 = 0                                      # Region of plot, lower limit
Max_plot_8 = 401                                    # Region of plot, upper limit
Vr_plot_8 = np.array(Vol8[Min_plot_8:Max_plot_8])   # Voltage region
Cur_plot_8 = np.array(Curr8[Min_plot_8:Max_plot_8]) # Current region

###############################################################

###### Optimization for V_back = -35 V ###########################

Min_8 = 198                                         # Region of fit, lower limit
Max_8 = 271                                         # Region of fit, upper limit
w_8 = 0.002                                         # Weight for the data fit
Vr_8 = np.array(Vol8[Min_8:Max_8])                  # Voltage region
Cur_8 = np.array(Curr8[Min_8:Max_8])                # Current region             
                                         
#################################################################


########### Plot for V_back = -40 V #############################

Min_plot_9 = 0                                      # Region of plot, lower limit
Max_plot_9 = 426                                    # Region of plot, upper limit
Vr_plot_9 = np.array(Vol9[Min_plot_9:Max_plot_9])   # Voltage region
Cur_plot_9 = np.array(Curr9[Min_plot_9:Max_plot_9]) # Current region

###############################################################

###### Optimization for V_back = -40 V ###########################

Min_9 = 223                                         # Region of fit, lower limit
Max_9 = 296                                         # Region of fit, upper limit
w_9 = 0.002                                         # Weight for the data fit
Vr_9 = np.array(Vol9[Min_9:Max_9])                  # Voltage region
Cur_9 = np.array(Curr9[Min_9:Max_9])                # Current region             
                                         
#################################################################

########### Plot for V_back = -45 V #############################

Min_plot_10 = 0                                      # Region of plot, lower limit
Max_plot_10 = 426                                    # Region of plot, upper limit
Vr_plot_10 = np.array(Vol10[Min_plot_10:Max_plot_10])   # Voltage region
Cur_plot_10 = np.array(Curr10[Min_plot_10:Max_plot_10]) # Current region

###############################################################

###### Optimization for V_back = -45 V ###########################

Min_10 = 223                                         # Region of fit, lower limit
Max_10 = 296                                         # Region of fit, upper limit
w_10 = 0.002                                         # Weight for the data fit
Vr_10 = np.array(Vol10[Min_10:Max_10])                  # Voltage region
Cur_10 = np.array(Curr10[Min_10:Max_10])                # Current region             
                                         
#################################################################


########### Plot for V_back = -50 V #############################

Min_plot_11 = 0                                      # Region of plot, lower limit
Max_plot_11 = 426                                    # Region of plot, upper limit
Vr_plot_11 = np.array(Vol11[Min_plot_11:Max_plot_11])   # Voltage region
Cur_plot_11 = np.array(Curr11[Min_plot_11:Max_plot_11]) # Current region

###############################################################

###### Optimization for V_back = -50 V ###########################

Min_11 = 223                                         # Region of fit, lower limit
Max_11 = 296                                         # Region of fit, upper limit
w_11 = 0.002                                         # Weight for the data fit
Vr_11 = np.array(Vol11[Min_11:Max_11])                  # Voltage region
Cur_11 = np.array(Curr11[Min_11:Max_11])                # Current region             
                                         
#################################################################


########### Plot for V_back = -55 V #############################

Min_plot_12 = 0                                      # Region of plot, lower limit
Max_plot_12 = 426                                    # Region of plot, upper limit
Vr_plot_12 = np.array(Vol12[Min_plot_12:Max_plot_12])   # Voltage region
Cur_plot_12 = np.array(Curr12[Min_plot_12:Max_plot_12]) # Current region

###############################################################

###### Optimization for V_back = -55 V ###########################

Min_12 = 223                                         # Region of fit, lower limit
Max_12 = 296                                         # Region of fit, upper limit
w_12 = 0.002                                         # Weight for the data fit
Vr_12 = np.array(Vol12[Min_12:Max_12])                  # Voltage region
Cur_12 = np.array(Curr12[Min_12:Max_12])                # Current region             
                                         
#################################################################

########### Plot for V_back = -60 V #############################

Min_plot_13 = 0                                      # Region of plot, lower limit
Max_plot_13 = 426                                    # Region of plot, upper limit
Vr_plot_13 = np.array(Vol13[Min_plot_13:Max_plot_13])   # Voltage region
Cur_plot_13 = np.array(Curr13[Min_plot_13:Max_plot_13]) # Current region

###############################################################

###### Optimization for V_back = -60 V ###########################

Min_13 = 223                                         # Region of fit, lower limit
Max_13 = 296                                         # Region of fit, upper limit
w_13 = 0.002                                         # Weight for the data fit
Vr_13 = np.array(Vol13[Min_13:Max_13])                  # Voltage region
Cur_13 = np.array(Curr13[Min_13:Max_13])                # Current region             
                                         
#################################################################

########### Plot for V_back = -65 V #############################

Min_plot_14 = 0                                      # Region of plot, lower limit
Max_plot_14 = 426                                    # Region of plot, upper limit
Vr_plot_14 = np.array(Vol14[Min_plot_14:Max_plot_14])   # Voltage region
Cur_plot_14 = np.array(Curr14[Min_plot_14:Max_plot_14]) # Current region

###############################################################

###### Optimization for V_back = -65 V ###########################

Min_14 = 223                                         # Region of fit, lower limit
Max_14 = 296                                         # Region of fit, upper limit
w_14 = 0.002                                         # Weight for the data fit
Vr_14 = np.array(Vol14[Min_14:Max_14])                  # Voltage region
Cur_14 = np.array(Curr14[Min_14:Max_14])                # Current region             
                                         
#################################################################

########### Plot for V_back = -70 V #############################

Min_plot_15 = 0                                      # Region of plot, lower limit
Max_plot_15 = 426                                    # Region of plot, upper limit
Vr_plot_15 = np.array(Vol15[Min_plot_15:Max_plot_15])   # Voltage region
Cur_plot_15 = np.array(Curr15[Min_plot_15:Max_plot_15]) # Current region

###############################################################

###### Optimization for V_back = -70 V ###########################

Min_15 = 223                                         # Region of fit, lower limit
Max_15 = 296                                         # Region of fit, upper limit
w_15 = 0.002                                         # Weight for the data fit
Vr_15 = np.array(Vol15[Min_15:Max_15])                  # Voltage region
Cur_15 = np.array(Curr15[Min_15:Max_15])                # Current region             
                                         
#################################################################


########### Plot for V_back = -75 V #############################

Min_plot_16 = 0                                      # Region of plot, lower limit
Max_plot_16 = 426                                    # Region of plot, upper limit
Vr_plot_16 = np.array(Vol16[Min_plot_16:Max_plot_16])   # Voltage region
Cur_plot_16 = np.array(Curr16[Min_plot_16:Max_plot_16]) # Current region

###############################################################

###### Optimization for V_back = -75 V ###########################

Min_16 = 223                                         # Region of fit, lower limit
Max_16 = 296                                         # Region of fit, upper limit
w_16 = 0.002                                         # Weight for the data fit
Vr_16 = np.array(Vol16[Min_16:Max_16])                  # Voltage region
Cur_16 = np.array(Curr16[Min_16:Max_16])                # Current region             
                                         
#################################################################

########### Plot for V_back = -80 V #############################

Min_plot_17 = 0                                      # Region of plot, lower limit
Max_plot_17 = 426                                    # Region of plot, upper limit
Vr_plot_17 = np.array(Vol17[Min_plot_17:Max_plot_17])   # Voltage region
Cur_plot_17 = np.array(Curr17[Min_plot_17:Max_plot_17]) # Current region

###############################################################

###### Optimization for V_back = -80 V ###########################

Min_17 = 223                                         # Region of fit, lower limit
Max_17 = 296                                         # Region of fit, upper limit
w_17 = 0.002                                         # Weight for the data fit
Vr_17 = np.array(Vol17[Min_17:Max_17])                  # Voltage region
Cur_17 = np.array(Curr17[Min_17:Max_17])                # Current region             
                                         
#################################################################

########### Plot for V_back = -85 V #############################

Min_plot_18 = 0                                      # Region of plot, lower limit
Max_plot_18 = 426                                    # Region of plot, upper limit
Vr_plot_18 = np.array(Vol18[Min_plot_18:Max_plot_18])   # Voltage region
Cur_plot_18 = np.array(Curr18[Min_plot_18:Max_plot_18]) # Current region

###############################################################

###### Optimization for V_back = -85 V ###########################

Min_18 = 223                                         # Region of fit, lower limit
Max_18 = 296                                         # Region of fit, upper limit
w_18 = 0.002                                         # Weight for the data fit
Vr_18 = np.array(Vol18[Min_18:Max_18])                  # Voltage region
Cur_18 = np.array(Curr18[Min_18:Max_18])                # Current region             
                                         
#################################################################

########### Plot for V_back = -90 V #############################

Min_plot_19 = 0                                      # Region of plot, lower limit
Max_plot_19 = 426                                    # Region of plot, upper limit
Vr_plot_19 = np.array(Vol19[Min_plot_19:Max_plot_19])   # Voltage region
Cur_plot_19 = np.array(Curr19[Min_plot_19:Max_plot_19]) # Current region

###############################################################

###### Optimization for V_back = -90 V ###########################

Min_19 = 223                                         # Region of fit, lower limit
Max_19 = 296                                         # Region of fit, upper limit
w_19 = 0.002                                         # Weight for the data fit
Vr_19 = np.array(Vol19[Min_19:Max_19])                  # Voltage region
Cur_19 = np.array(Curr19[Min_19:Max_19])                # Current region             
                                         
#################################################################

########### Plot for V_back = -95 V #############################

Min_plot_20 = 0                                      # Region of plot, lower limit
Max_plot_20 = 426                                    # Region of plot, upper limit
Vr_plot_20 = np.array(Vol20[Min_plot_20:Max_plot_20])   # Voltage region
Cur_plot_20 = np.array(Curr20[Min_plot_20:Max_plot_20]) # Current region

###############################################################

###### Optimization for V_back = -95 V ###########################

Min_20 = 223                                         # Region of fit, lower limit
Max_20 = 296                                         # Region of fit, upper limit
w_20 = 0.002                                         # Weight for the data fit
Vr_20 = np.array(Vol20[Min_20:Max_20])                  # Voltage region
Cur_20 = np.array(Curr20[Min_20:Max_20])                # Current region             
                                         
#################################################################

########### Plot for V_back = -100 V #############################

Min_plot_21 = 0                                      # Region of plot, lower limit
Max_plot_21 = 426                                    # Region of plot, upper limit
Vr_plot_21 = np.array(Vol21[Min_plot_21:Max_plot_21])   # Voltage region
Cur_plot_21 = np.array(Curr21[Min_plot_21:Max_plot_21]) # Current region

###############################################################

###### Optimization for V_back = -100 V ###########################

Min_21 = 223                                         # Region of fit, lower limit
Max_21 = 296                                         # Region of fit, upper limit
w_21 = 0.002                                         # Weight for the data fit
Vr_21 = np.array(Vol21[Min_21:Max_21])                  # Voltage region
Cur_21 = np.array(Curr21[Min_21:Max_21])                # Current region             
                                         
#################################################################

########### Plot for V_back = -105 V #############################

Min_plot_22 = 0                                      # Region of plot, lower limit
Max_plot_22 = 426                                    # Region of plot, upper limit
Vr_plot_22 = np.array(Vol22[Min_plot_22:Max_plot_22])   # Voltage region
Cur_plot_22 = np.array(Curr22[Min_plot_22:Max_plot_22]) # Current region

###############################################################

###### Optimization for V_back = -105 V ###########################

Min_22 = 223                                         # Region of fit, lower limit
Max_22 = 296                                         # Region of fit, upper limit
w_22 = 0.002                                         # Weight for the data fit
Vr_22 = np.array(Vol22[Min_22:Max_22])               # Voltage region
Cur_22 = np.array(Curr22[Min_22:Max_22])                # Current region             
                                         
#################################################################


########### Plot for V_back = -110 V #############################

Min_plot_23 = 0                                      # Region of plot, lower limit
Max_plot_23 = 426                                    # Region of plot, upper limit
Vr_plot_23 = np.array(Vol23[Min_plot_23:Max_plot_23])   # Voltage region
Cur_plot_23 = np.array(Curr23[Min_plot_23:Max_plot_23]) # Current region

###############################################################

###### Optimization for V_back = -110 V ###########################

Min_23 = 223                                         # Region of fit, lower limit
Max_23 = 296                                         # Region of fit, upper limit
w_23 = 0.002                                         # Weight for the data fit
Vr_23 = np.array(Vol23[Min_23:Max_23])                  # Voltage region
Cur_23 = np.array(Curr23[Min_23:Max_23])                # Current region             
                                         
#################################################################

########### Plot for V_back = -115 V #############################

Min_plot_24 = 0                                      # Region of plot, lower limit
Max_plot_24 = 426                                    # Region of plot, upper limit
Vr_plot_24 = np.array(Vol24[Min_plot_24:Max_plot_24])   # Voltage region
Cur_plot_24 = np.array(Curr24[Min_plot_24:Max_plot_24]) # Current region

###############################################################

###### Optimization for V_back = -115 V ###########################

Min_24 = 223                                         # Region of fit, lower limit
Max_24 = 296                                         # Region of fit, upper limit
w_24 = 0.002                                         # Weight for the data fit
Vr_24 = np.array(Vol24[Min_24:Max_24])                  # Voltage region
Cur_24 = np.array(Curr24[Min_24:Max_24])                # Current region             
                                         
#################################################################


########### Plot for V_back = -120 V #############################

Min_plot_25 = 0                                      # Region of plot, lower limit
Max_plot_25 = 426                                    # Region of plot, upper limit
Vr_plot_25 = np.array(Vol25[Min_plot_25:Max_plot_25])   # Voltage region
Cur_plot_25 = np.array(Curr25[Min_plot_25:Max_plot_25]) # Current region

###############################################################

###### Optimization for V_back = -120 V ###########################

Min_25 = 223                                         # Region of fit, lower limit
Max_25 = 296                                         # Region of fit, upper limit
w_25 = 0.002                                         # Weight for the data fit
Vr_25 = np.array(Vol25[Min_25:Max_25])                  # Voltage region
Cur_25 = np.array(Curr25[Min_25:Max_25])                # Current region             
                                         
#################################################################


############ Curve fit function V_back = 0 V ###########################

sigma_1 = np.array(Curr1[Min_1:Max_1]*w_1)
p0_1 = np.array([1.52e-06, 0.0146, 2.341])
print "Parameters Vback = 0 V", optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)

Alpha_1 = optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[0][0]
beta_1 = optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[0][1]
V_th_1 =   optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -5 V ###########################

sigma_2 = np.array(Curr2[Min_2:Max_2]*w_2)
p0_2 = np.array([1.52e-06, 0.0146, 2.341])
print "Parameters Vback = -5 V", optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)

Alpha_2 = optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[0][0]
beta_2 = optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[0][1]
V_th_2 =   optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -10 V ###########################

sigma_3 = np.array(Curr3[Min_3:Max_3]*w_3)
p0_3 = np.array([1.52e-06, 0.0146, 2.341])
print "Parameters Vback = -10 V", optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)

Alpha_3 = optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[0][0]
beta_3 = optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[0][1]
V_th_3 =   optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -15 V ###########################

sigma_4 = np.array(Curr4[Min_4:Max_4]*w_4)
p0_4 = np.array([1.52e-06, 0.0146, 2.341])
print "Parameters Vback = -15 V", optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)

Alpha_4 = optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[0][0]
beta_4 = optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[0][1]
V_th_4 =   optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -20 V ###########################

sigma_5 = np.array(Curr5[Min_5:Max_5]*w_5)
p0_5 = np.array([1.52e-06, 0.0146, 2.341])
print "Parameters Vback = -20 V", optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)

Alpha_5 = optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[0][0]
beta_5 = optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[0][1]
V_th_5 =   optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -25 V ###########################

sigma_6 = np.array(Curr6[Min_6:Max_6]*w_6)
p0_6 = np.array([ 1.52930864e-06,   2.42926553e-02,  -1.41510893e+00])
print "Parameters Vback = -25 V", optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)

Alpha_6 = optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[0][0]
beta_6 = optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[0][1]
V_th_6 =   optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -30 V ###########################

sigma_7 = np.array(Curr7[Min_7:Max_7]*w_7)
p0_7 = np.array([ 1.52930864e-06,   2.42926553e-02,  -1.41510893e+00])
print "Parameters Vback = -30 V", optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7, sigma_7, absolute_sigma=True)

Alpha_7 = optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[0][0]
beta_7 = optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[0][1]
V_th_7 =   optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -35 V ###########################

sigma_8 = np.array(Curr8[Min_8:Max_8]*w_8)
p0_8 = np.array([1.35470088e-06,   1.59309044e-02,  -8.26539997e-01])
print "Parameters Vback = -35 V", optimization.curve_fit(Id_fit, Vr_8, Cur_8, p0_8, sigma_8, absolute_sigma=True)

Alpha_8 = optimization.curve_fit(Id_fit, Vr_8, Cur_8, p0_8 ,sigma_8, absolute_sigma=True)[0][0]
beta_8 = optimization.curve_fit(Id_fit, Vr_8, Cur_8, p0_8 ,sigma_8, absolute_sigma=True)[0][1]
V_th_8 =   optimization.curve_fit(Id_fit, Vr_8, Cur_8, p0_8 ,sigma_8, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -40 V ###########################

sigma_9 = np.array(Curr9[Min_9:Max_9]*w_9)
p0_9 = np.array([ 1.52930864e-06,   2.42926553e-02,  -1.41510893e+00])
print "Parameters Vback = -40 V", optimization.curve_fit(Id_fit, Vr_9, Cur_9, p0_9, sigma_9, absolute_sigma=True)

Alpha_9 = optimization.curve_fit(Id_fit, Vr_9, Cur_9, p0_9 ,sigma_9, absolute_sigma=True)[0][0]
beta_9 = optimization.curve_fit(Id_fit, Vr_9, Cur_9, p0_9 ,sigma_9, absolute_sigma=True)[0][1]
V_th_9 =   optimization.curve_fit(Id_fit, Vr_9, Cur_9, p0_9 ,sigma_9, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -45 V ###########################

sigma_10 = np.array(Curr10[Min_10:Max_10]*w_10)
p0_10 = np.array([ 1.52930864e-06,   2.42926553e-02,  -1.41510893e+00])
print "Parameters Vback = -45 V", optimization.curve_fit(Id_fit, Vr_10, Cur_10, p0_10, sigma_10, absolute_sigma=True)

Alpha_10 = optimization.curve_fit(Id_fit, Vr_10, Cur_10, p0_10 ,sigma_10, absolute_sigma=True)[0][0]
beta_10 = optimization.curve_fit(Id_fit, Vr_10, Cur_10, p0_10 ,sigma_10, absolute_sigma=True)[0][1]
V_th_10 =   optimization.curve_fit(Id_fit, Vr_10, Cur_10, p0_10 ,sigma_10, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -50 V ###########################

sigma_11 = np.array(Curr11[Min_11:Max_11]*w_11)
p0_11 = np.array([ 1.52930864e-06,   2.42926553e-02,  -1.41510893e+00])
print "Parameters Vback = -50 V", optimization.curve_fit(Id_fit, Vr_11, Cur_11, p0_11, sigma_11, absolute_sigma=True)

Alpha_11 = optimization.curve_fit(Id_fit, Vr_11, Cur_11, p0_11 ,sigma_11, absolute_sigma=True)[0][0]
beta_11 = optimization.curve_fit(Id_fit, Vr_11, Cur_11, p0_11 ,sigma_11, absolute_sigma=True)[0][1]
V_th_11 =   optimization.curve_fit(Id_fit, Vr_11, Cur_11, p0_11 ,sigma_11, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -55 V ###########################

sigma_12 = np.array(Curr12[Min_12:Max_12]*w_12)
p0_12 = np.array([ 1.52930864e-06,   2.42926553e-02,  -1.41510893e+00])
print "Parameters Vback = -55 V", optimization.curve_fit(Id_fit, Vr_12, Cur_12, p0_12, sigma_12, absolute_sigma=True)

Alpha_12 = optimization.curve_fit(Id_fit, Vr_12, Cur_12, p0_12 ,sigma_12, absolute_sigma=True)[0][0]
beta_12 = optimization.curve_fit(Id_fit, Vr_12, Cur_12, p0_12 ,sigma_12, absolute_sigma=True)[0][1]
V_th_12 =   optimization.curve_fit(Id_fit, Vr_12, Cur_12, p0_12 ,sigma_12, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -60 V ###########################

sigma_13 = np.array(Curr13[Min_13:Max_13]*w_13)
p0_13 = np.array([ 1.35417002e-06,   1.62060627e-02,  -4.01458248e-01])
print "Parameters Vback = -60 V", optimization.curve_fit(Id_fit, Vr_13, Cur_13, p0_13, sigma_13, absolute_sigma=True)

Alpha_13 = optimization.curve_fit(Id_fit, Vr_13, Cur_13, p0_13 ,sigma_13, absolute_sigma=True)[0][0]
beta_13 = optimization.curve_fit(Id_fit, Vr_13, Cur_13, p0_13 ,sigma_13, absolute_sigma=True)[0][1]
V_th_13 =   optimization.curve_fit(Id_fit, Vr_13, Cur_13, p0_13 ,sigma_13, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -65 V ###########################

sigma_14 = np.array(Curr14[Min_14:Max_14]*w_14)
p0_14 = np.array([ 7.26440485e-07,  -2.69059902e-02,  -4.00516943e-01])
print "Parameters Vback = -65 V", optimization.curve_fit(Id_fit, Vr_14, Cur_14, p0_14, sigma_14, absolute_sigma=True)

Alpha_14 = optimization.curve_fit(Id_fit, Vr_14, Cur_14, p0_14 ,sigma_14, absolute_sigma=True)[0][0]
beta_14 = optimization.curve_fit(Id_fit, Vr_14, Cur_14, p0_14 ,sigma_14, absolute_sigma=True)[0][1]
V_th_14 =   optimization.curve_fit(Id_fit, Vr_14, Cur_14, p0_14 ,sigma_14, absolute_sigma=True)[0][2]


###########################################################################


############ Curve fit function V_back = -70 V ###########################

sigma_15 = np.array(Curr15[Min_15:Max_15]*w_15)
p0_15 = np.array([ 7.26440485e-07,  -2.69059902e-02,  -4.00516943e-01])
print "Parameters Vback = -70 V", optimization.curve_fit(Id_fit, Vr_15, Cur_15, p0_15, sigma_15, absolute_sigma=True)

Alpha_15 = optimization.curve_fit(Id_fit, Vr_15, Cur_15, p0_15 ,sigma_15, absolute_sigma=True)[0][0]
beta_15 = optimization.curve_fit(Id_fit, Vr_15, Cur_15, p0_15 ,sigma_15, absolute_sigma=True)[0][1]
V_th_15 =   optimization.curve_fit(Id_fit, Vr_15, Cur_15, p0_15 ,sigma_15, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -75 V ###########################

sigma_16 = np.array(Curr16[Min_16:Max_16]*w_16)
p0_16 = np.array([ 7.26440485e-07,  -2.69059902e-02,  -4.00516943e-01])
print "Parameters Vback = -75 V", optimization.curve_fit(Id_fit, Vr_16, Cur_16, p0_16, sigma_16, absolute_sigma=True)

Alpha_16 = optimization.curve_fit(Id_fit, Vr_16, Cur_16, p0_16 ,sigma_16, absolute_sigma=True)[0][0]
beta_16 = optimization.curve_fit(Id_fit, Vr_16, Cur_16, p0_16 ,sigma_16, absolute_sigma=True)[0][1]
V_th_16 =   optimization.curve_fit(Id_fit, Vr_16, Cur_16, p0_16 ,sigma_16, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -80 V ###########################

sigma_17 = np.array(Curr17[Min_17:Max_17]*w_17)
p0_17 = np.array([ 7.26440485e-07,  -2.69059902e-02,  -4.00516943e-01])
print "Parameters Vback = -80 V", optimization.curve_fit(Id_fit, Vr_17, Cur_17, p0_17, sigma_17, absolute_sigma=True)

Alpha_17 = optimization.curve_fit(Id_fit, Vr_17, Cur_17, p0_17 ,sigma_17, absolute_sigma=True)[0][0]
beta_17 = optimization.curve_fit(Id_fit, Vr_17, Cur_17, p0_17 ,sigma_17, absolute_sigma=True)[0][1]
V_th_17 =   optimization.curve_fit(Id_fit, Vr_17, Cur_17, p0_17 ,sigma_17, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -85 V ###########################

sigma_18 = np.array(Curr18[Min_18:Max_18]*w_18)
p0_18 = np.array([ 7.26440485e-07,  -2.69059902e-02,  -4.00516943e-01])
print "Parameters Vback = -85 V", optimization.curve_fit(Id_fit, Vr_18, Cur_18, p0_18, sigma_18, absolute_sigma=True)

Alpha_18 = optimization.curve_fit(Id_fit, Vr_18, Cur_18, p0_18 ,sigma_18, absolute_sigma=True)[0][0]
beta_18 = optimization.curve_fit(Id_fit, Vr_18, Cur_18, p0_18 ,sigma_18, absolute_sigma=True)[0][1]
V_th_18 =   optimization.curve_fit(Id_fit, Vr_18, Cur_18, p0_18 ,sigma_18, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -90 V ###########################

sigma_19 = np.array(Curr19[Min_19:Max_19]*w_19)
p0_19 = np.array([ 7.26440485e-07,  -2.69059902e-02,  -4.00516943e-01])
print "Parameters Vback = -90 V", optimization.curve_fit(Id_fit, Vr_19, Cur_19, p0_19, sigma_19, absolute_sigma=True)

Alpha_19 = optimization.curve_fit(Id_fit, Vr_19, Cur_19, p0_19 ,sigma_19, absolute_sigma=True)[0][0]
beta_19 = optimization.curve_fit(Id_fit, Vr_19, Cur_19, p0_19 ,sigma_19, absolute_sigma=True)[0][1]
V_th_19 =   optimization.curve_fit(Id_fit, Vr_19, Cur_19, p0_19 ,sigma_19, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -95 V ###########################

sigma_20 = np.array(Curr20[Min_20:Max_20]*w_20)
p0_20 = np.array([ 7.26440485e-07,  -2.69059902e-02,  -4.00516943e-01])
print "Parameters Vback = -95 V", optimization.curve_fit(Id_fit, Vr_20, Cur_20, p0_20, sigma_20, absolute_sigma=True)

Alpha_20 = optimization.curve_fit(Id_fit, Vr_20, Cur_20, p0_20 ,sigma_20, absolute_sigma=True)[0][0]
beta_20 = optimization.curve_fit(Id_fit, Vr_20, Cur_20, p0_20 ,sigma_20, absolute_sigma=True)[0][1]
V_th_20 =   optimization.curve_fit(Id_fit, Vr_20, Cur_20, p0_20 ,sigma_20, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -100 V ###########################

sigma_21 = np.array(Curr21[Min_21:Max_21]*w_21)
p0_21 = np.array([ 7.26440485e-07,  -2.69059902e-02,  -4.00516943e-01])
print "Parameters Vback = -100 V", optimization.curve_fit(Id_fit, Vr_21, Cur_21, p0_21, sigma_21, absolute_sigma=True)

Alpha_21 = optimization.curve_fit(Id_fit, Vr_21, Cur_21, p0_21 ,sigma_21, absolute_sigma=True)[0][0]
beta_21 = optimization.curve_fit(Id_fit, Vr_21, Cur_21, p0_21 ,sigma_21, absolute_sigma=True)[0][1]
V_th_21 =   optimization.curve_fit(Id_fit, Vr_21, Cur_21, p0_21 ,sigma_21, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -105 V ###########################

sigma_22 = np.array(Curr22[Min_22:Max_22]*w_22)
p0_22 = np.array([ 7.26440485e-07,  -2.69059902e-02,  -4.00516943e-01])
print "Parameters Vback = -105 V", optimization.curve_fit(Id_fit, Vr_22, Cur_22, p0_22, sigma_22, absolute_sigma=True)

Alpha_22 = optimization.curve_fit(Id_fit, Vr_22, Cur_22, p0_22 ,sigma_22, absolute_sigma=True)[0][0]
beta_22 = optimization.curve_fit(Id_fit, Vr_22, Cur_22, p0_22 ,sigma_22, absolute_sigma=True)[0][1]
V_th_22 =   optimization.curve_fit(Id_fit, Vr_22, Cur_22, p0_22 ,sigma_22, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -110 V ###########################

sigma_23 = np.array(Curr23[Min_23:Max_23]*w_23)
p0_23 = np.array([  7.26440485e-07,  -2.69059902e-02,  -4.00516943e-01])
print "Parameters Vback = -110 V", optimization.curve_fit(Id_fit, Vr_23, Cur_23, p0_23, sigma_23, absolute_sigma=True)

Alpha_23 = optimization.curve_fit(Id_fit, Vr_23, Cur_23, p0_23 ,sigma_23, absolute_sigma=True)[0][0]
beta_23 = optimization.curve_fit(Id_fit, Vr_23, Cur_23, p0_23 ,sigma_23, absolute_sigma=True)[0][1]
V_th_23 =   optimization.curve_fit(Id_fit, Vr_23, Cur_23, p0_23 ,sigma_23, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -115 V ###########################

sigma_24 = np.array(Curr24[Min_24:Max_24]*w_24)
p0_24 = np.array([ 9.88423937e-07,   1.62054930e-02,   2.29223599e-01])
print "Parameters Vback = -115 V", optimization.curve_fit(Id_fit, Vr_24, Cur_24, p0_24, sigma_24, absolute_sigma=True)

Alpha_24 = optimization.curve_fit(Id_fit, Vr_24, Cur_24, p0_24 ,sigma_24, absolute_sigma=True)[0][0]
beta_24 = optimization.curve_fit(Id_fit, Vr_24, Cur_24, p0_24 ,sigma_24, absolute_sigma=True)[0][1]
V_th_24 =   optimization.curve_fit(Id_fit, Vr_24, Cur_24, p0_24 ,sigma_24, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -120 V ###########################

sigma_25 = np.array(Curr25[Min_25:Max_25]*w_25)
p0_25 = np.array([   7.61471285e-07,   1.58691770e-02,   4.00377396e-01])
print "Parameters Vback = -120 V", optimization.curve_fit(Id_fit, Vr_25, Cur_25, p0_25, sigma_25, absolute_sigma=True)

Alpha_25 = optimization.curve_fit(Id_fit, Vr_25, Cur_25, p0_25 ,sigma_25, absolute_sigma=True)[0][0]
beta_25 = optimization.curve_fit(Id_fit, Vr_25, Cur_25, p0_25 ,sigma_25, absolute_sigma=True)[0][1]
V_th_25 =   optimization.curve_fit(Id_fit, Vr_25, Cur_25, p0_25 ,sigma_25, absolute_sigma=True)[0][2]

###########################################################################



######### Mobility for dif. backside voltage ############################## 

print "The m_0 - Vback=0 V is:", Alpha_1*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-5 V is:", Alpha_2*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-10 V is:", Alpha_3*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-15 V is:", Alpha_4*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-20 V is:", Alpha_5*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-25 V is:", Alpha_6*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-30 V is:", Alpha_7*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-35 V is:", Alpha_8*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_8, Cur_8, p0_8 ,sigma_8, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-40 V is:", Alpha_9*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_9, Cur_9, p0_9 ,sigma_9, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-45 V is:", Alpha_10*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_10, Cur_10, p0_10 ,sigma_10, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-50 V is:", Alpha_11*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_11, Cur_11, p0_11 ,sigma_11, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-55 V is:", Alpha_12*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_12, Cur_12, p0_12 ,sigma_12, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-60 V is:", Alpha_13*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_13, Cur_13, p0_13 ,sigma_13, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-65 V is:", Alpha_14*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_14, Cur_14, p0_14 ,sigma_14, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-70 V is:", Alpha_15*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_15, Cur_15, p0_15 ,sigma_15, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-75 V is:", Alpha_16*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_16, Cur_16, p0_16 ,sigma_16, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-80 V is:", Alpha_17*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_17, Cur_17, p0_17 ,sigma_17, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-85 V is:", Alpha_18*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_18, Cur_18, p0_18 ,sigma_18, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-90 V is:", Alpha_19*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_19, Cur_19, p0_19 ,sigma_19, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-95 V is:", Alpha_20*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_20, Cur_20, p0_20 ,sigma_20, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-100 V is:", Alpha_21*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_21, Cur_21, p0_21 ,sigma_21, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-105 V is:", Alpha_22*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_22, Cur_22, p0_22 ,sigma_22, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-110 V is:", Alpha_23*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_23, Cur_23, p0_23 ,sigma_23, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-115 V is:", Alpha_24*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_24, Cur_24, p0_24 ,sigma_24, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
print "The m_0 - Vback=-120 V is:", Alpha_25*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_25, Cur_25, p0_25 ,sigma_25, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."


###########################################################################

########## V1/2 for dif. backside voltage #################################

print "The V_1/2 - Vback=0 V is:", 1.0/beta_1,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[1][1][1])*(1.0/beta_1)**2,"."
print "The V_1/2 - Vback=-5 V is:", 1.0/beta_2,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[1][1][1])*(1.0/beta_2)**2,"."
print "The V_1/2 - Vback=-10 V is:", 1.0/beta_3,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[1][1][1])*(1.0/beta_3)**2,"."
print "The V_1/2 - Vback=-15 V is:", 1.0/beta_4,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[1][1][1])*(1.0/beta_4)**2,"."
print "The V_1/2 - Vback=-20 V is:", 1.0/beta_5,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[1][1][1])*(1.0/beta_5)**2,"."
print "The V_1/2 - Vback=-25 V is:", 1.0/beta_6,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[1][1][1])*(1.0/beta_6)**2,"."
print "The V_1/2 - Vback=-30 V is:", 1.0/beta_7,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[1][1][1])*(1.0/beta_7)**2,"."
print "The V_1/2 - Vback=-35 V is:", 1.0/beta_8,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_8, Cur_8, p0_8 ,sigma_8, absolute_sigma=True)[1][1][1])*(1.0/beta_8)**2,"."
print "The V_1/2 - Vback=-40 V is:", 1.0/beta_9,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_9, Cur_9, p0_9 ,sigma_9, absolute_sigma=True)[1][1][1])*(1.0/beta_9)**2,"."
print "The V_1/2 - Vback=-45 V is:", 1.0/beta_10,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_10, Cur_10, p0_10 ,sigma_10, absolute_sigma=True)[1][1][1])*(1.0/beta_10)**2,"."
print "The V_1/2 - Vback=-50 V is:", 1.0/beta_11,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_11, Cur_11, p0_11 ,sigma_11, absolute_sigma=True)[1][1][1])*(1.0/beta_11)**2,"."
print "The V_1/2 - Vback=-55 V is:", 1.0/beta_12,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_12, Cur_12, p0_12 ,sigma_12, absolute_sigma=True)[1][1][1])*(1.0/beta_12)**2,"."
print "The V_1/2 - Vback=-60 V is:", 1.0/beta_13,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_13, Cur_13, p0_13 ,sigma_13, absolute_sigma=True)[1][1][1])*(1.0/beta_13)**2,"."
print "The V_1/2 - Vback=-65 V is:", 1.0/beta_14,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_14, Cur_14, p0_14 ,sigma_14, absolute_sigma=True)[1][1][1])*(1.0/beta_14)**2,"."
print "The V_1/2 - Vback=-70 V is:", 1.0/beta_15,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_15, Cur_15, p0_15 ,sigma_15, absolute_sigma=True)[1][1][1])*(1.0/beta_15)**2,"."
print "The V_1/2 - Vback=-75 V is:", 1.0/beta_16,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_16, Cur_16, p0_16 ,sigma_16, absolute_sigma=True)[1][1][1])*(1.0/beta_16)**2,"."
print "The V_1/2 - Vback=-80 V is:", 1.0/beta_17,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_17, Cur_17, p0_17 ,sigma_17, absolute_sigma=True)[1][1][1])*(1.0/beta_17)**2,"."
print "The V_1/2 - Vback=-85 V is:", 1.0/beta_18,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_18, Cur_18, p0_18 ,sigma_18, absolute_sigma=True)[1][1][1])*(1.0/beta_18)**2,"."
print "The V_1/2 - Vback=-90 V is:", 1.0/beta_19,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_19, Cur_19, p0_19 ,sigma_19, absolute_sigma=True)[1][1][1])*(1.0/beta_19)**2,"."
print "The V_1/2 - Vback=-95 V is:", 1.0/beta_20,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_20, Cur_20, p0_20 ,sigma_20, absolute_sigma=True)[1][1][1])*(1.0/beta_20)**2,"."
print "The V_1/2 - Vback=-100 V is:", 1.0/beta_21,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_21, Cur_21, p0_21 ,sigma_21, absolute_sigma=True)[1][1][1])*(1.0/beta_21)**2,"."
print "The V_1/2 - Vback=-105 V is:", 1.0/beta_22,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_22, Cur_22, p0_22 ,sigma_22, absolute_sigma=True)[1][1][1])*(1.0/beta_22)**2,"."
print "The V_1/2 - Vback=-110 V is:", 1.0/beta_23,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_23, Cur_23, p0_23 ,sigma_23, absolute_sigma=True)[1][1][1])*(1.0/beta_23)**2,"."
print "The V_1/2 - Vback=-115 V is:", 1.0/beta_24,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_24, Cur_24, p0_24 ,sigma_24, absolute_sigma=True)[1][1][1])*(1.0/beta_24)**2,"."
print "The V_1/2 - Vback=-120 V is:", 1.0/beta_25,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_25, Cur_25, p0_25 ,sigma_25, absolute_sigma=True)[1][1][1])*(1.0/beta_25)**2,"."

###########################################################################

########## Vth for dif. backside voltage #################################

print "The Vth - Vback=0 V is:", V_th_1 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-5 V is:", V_th_2 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-10 V is:", V_th_3 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-15 V is:", V_th_4 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-20 V is:", V_th_5 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-25 V is:", V_th_6 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-30 V is:", V_th_7 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-35 V is:", V_th_8 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_8, Cur_8, p0_8 ,sigma_8, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-40 V is:", V_th_9 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_9, Cur_9, p0_9 ,sigma_9, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-45 V is:", V_th_10 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_10, Cur_10, p0_10 ,sigma_10, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-50 V is:", V_th_11 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_11, Cur_11, p0_11 ,sigma_11, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-55 V is:", V_th_12 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_12, Cur_12, p0_12 ,sigma_12, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-60 V is:", V_th_13 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_13, Cur_13, p0_13 ,sigma_13, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-65 V is:", V_th_14 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_14, Cur_14, p0_14 ,sigma_14, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-70 V is:", V_th_15 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_15, Cur_15, p0_15 ,sigma_15, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-75 V is:", V_th_16 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_16, Cur_16, p0_16 ,sigma_16, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-80 V is:", V_th_17 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_17, Cur_17, p0_17 ,sigma_17, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-85 V is:", V_th_18 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_18, Cur_18, p0_18 ,sigma_18, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-90 V is:", V_th_19 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_19, Cur_19, p0_19 ,sigma_19, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-95 V is:", V_th_20 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_20, Cur_20, p0_20 ,sigma_20, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-100 V is:", V_th_21 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_21, Cur_21, p0_21 ,sigma_21, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-105 V is:", V_th_22 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_22, Cur_22, p0_22 ,sigma_22, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-110 V is:", V_th_23 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_23, Cur_23, p0_23 ,sigma_23, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-115 V is:", V_th_24 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_24, Cur_24, p0_24 ,sigma_24, absolute_sigma=True)[1][2][2]),"."
print "The Vth - Vback=-120 V is:", V_th_25 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_25, Cur_25, p0_25 ,sigma_25, absolute_sigma=True)[1][2][2]),"."


#V_th_1_err = sqrt(optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[1][2][2])
#V_th_2_err = sqrt(optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[1][2][2])
#V_th_3_err = sqrt(optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[1][2][2])
#V_th_4_err = sqrt(optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[1][2][2])
#V_th_5_err = sqrt(optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[1][2][2])
#V_th_6_err = sqrt(optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[1][2][2])
#V_th_7_err = sqrt(optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[1][2][2])
###########################################################################

################ Chi square ###############################################

print "The chi_square - Vback= 0 V is:", Chi_sq(Cur_1,Id_plot(Vr_1,Alpha_1,beta_1,V_th_1),w_1), "."     
print "The chi_square - Vback= -5 V is:", Chi_sq(Cur_2,Id_plot(Vr_2,Alpha_2,beta_2,V_th_2),w_2), "." 
print "The chi_square - Vback= -10 V is:", Chi_sq(Cur_3,Id_plot(Vr_3,Alpha_3,beta_3,V_th_1),w_3), "."         
print "The chi_square - Vback= -15 V is:", Chi_sq(Cur_4,Id_plot(Vr_4,Alpha_4,beta_4,V_th_4),w_4), "."   
print "The chi_square - Vback= -20 V is:", Chi_sq(Cur_5,Id_plot(Vr_5,Alpha_5,beta_5,V_th_5),w_5), "."   
print "The chi_square - Vback= -25 V is:", Chi_sq(Cur_6,Id_plot(Vr_6,Alpha_6,beta_6,V_th_6),w_6), "."   
print "The chi_square - Vback= -30 V is:", Chi_sq(Cur_7,Id_plot(Vr_7,Alpha_7,beta_7,V_th_7),w_7), "."   
print "The chi_square - Vback= -35 V is:", Chi_sq(Cur_8,Id_plot(Vr_8,Alpha_8,beta_8,V_th_8),w_8), "." 
print "The chi_square - Vback= -40 V is:", Chi_sq(Cur_9,Id_plot(Vr_9,Alpha_9,beta_9,V_th_9),w_9), "."         
print "The chi_square - Vback= -45 V is:", Chi_sq(Cur_10,Id_plot(Vr_10,Alpha_10,beta_10,V_th_10),w_10), "."   
print "The chi_square - Vback= -50 V is:", Chi_sq(Cur_11,Id_plot(Vr_11,Alpha_11,beta_11,V_th_11),w_11), "."   
print "The chi_square - Vback= -55 V is:", Chi_sq(Cur_12,Id_plot(Vr_12,Alpha_12,beta_12,V_th_12),w_12), "."   
print "The chi_square - Vback= -60 V is:", Chi_sq(Cur_13,Id_plot(Vr_13,Alpha_13,beta_13,V_th_13),w_13), "."   
print "The chi_square - Vback= -65 V is:", Chi_sq(Cur_14,Id_plot(Vr_14,Alpha_14,beta_14,V_th_14),w_14), "." 
print "The chi_square - Vback= -70 V is:", Chi_sq(Cur_15,Id_plot(Vr_15,Alpha_15,beta_15,V_th_15),w_15), "."         
print "The chi_square - Vback= -75 V is:", Chi_sq(Cur_16,Id_plot(Vr_16,Alpha_16,beta_16,V_th_16),w_16), "."   
print "The chi_square - Vback= -80 V is:", Chi_sq(Cur_17,Id_plot(Vr_17,Alpha_17,beta_17,V_th_17),w_17), "."   
print "The chi_square - Vback= -85 V is:", Chi_sq(Cur_18,Id_plot(Vr_18,Alpha_18,beta_18,V_th_18),w_18), "."   
print "The chi_square - Vback= -90 V is:", Chi_sq(Cur_19,Id_plot(Vr_19,Alpha_19,beta_19,V_th_19),w_19), "."   
print "The chi_square - Vback= -95 V is:", Chi_sq(Cur_20,Id_plot(Vr_20,Alpha_20,beta_20,V_th_20),w_20), "." 
print "The chi_square - Vback= -100 V is:", Chi_sq(Cur_21,Id_plot(Vr_21,Alpha_21,beta_21,V_th_21),w_21), "."         
print "The chi_square - Vback= -105 V is:", Chi_sq(Cur_22,Id_plot(Vr_22,Alpha_22,beta_22,V_th_22),w_22), "."   
print "The chi_square - Vback= -110 V is:", Chi_sq(Cur_23,Id_plot(Vr_23,Alpha_23,beta_23,V_th_23),w_23), "."   
print "The chi_square - Vback= -115 V is:", Chi_sq(Cur_24,Id_plot(Vr_24,Alpha_24,beta_24,V_th_24),w_24), "."   
print "The chi_square - Vback= -120 V is:", Chi_sq(Cur_25,Id_plot(Vr_25,Alpha_25,beta_25,V_th_25),w_25), "."   

###########################################################################



######## Fit dunction #######################################################


V_back = np.array([0.0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120])
m0 = np.array([Alpha_1, Alpha_2, Alpha_3, Alpha_4, Alpha_5, Alpha_6, Alpha_7, Alpha_8, Alpha_9, Alpha_10, Alpha_11, Alpha_12, Alpha_13, Alpha_14, Alpha_15, Alpha_16, Alpha_17, Alpha_18, Alpha_19, Alpha_20, Alpha_21, Alpha_22, Alpha_23, Alpha_24, Alpha_25]*Rev_fac_circ)
V_half = np.array([1.0/beta_1, 1.0/beta_2, 1.0/beta_3, 1.0/beta_4, 1.0/beta_5, 1.0/beta_6, 1.0/beta_7, 1.0/beta_8, 1.0/beta_9, 1.0/beta_10, 1.0/beta_11, 1.0/beta_12, 1.0/beta_13, 1.0/beta_14, 1.0/beta_15, 1.0/beta_16, 1.0/beta_17, 1.0/beta_18, 1.0/beta_19, 1.0/beta_20, 1.0/beta_21, 1.0/beta_22, 1.0/beta_23, 1.0/beta_24, 1.0/beta_25])
V_th = np.array([V_th_1, V_th_2, V_th_3, V_th_4, V_th_5, V_th_6, V_th_7, V_th_8, V_th_9, V_th_10, V_th_11, V_th_12, V_th_13, V_th_14, V_th_15, V_th_16, V_th_17, V_th_18, V_th_19, V_th_20, V_th_21, V_th_22, V_th_23, V_th_24, V_th_25])
#V_th_error = np.array([V_th_1_err, V_th_2_err, V_th_3_err, V_th_4_err, V_th_5_err, V_th_6_err, V_th_7_err])


#Min_f = 0
#Max_f = 7

#def Vt_func(vol, Vt_func_slope, Vt_func_inter):
#	Vt_func = Vt_func_slope*vol + Vt_func_inter
#	return Vt_func
	
#p0_Vt_func = np.array([0.25212634, -1.878022])	
	
#print "Vt func", optimization.curve_fit(Vt_func, sqrt(2.0*f_f(Nd) + V_back[Min_f:Max_f]), V_th[Min_f:Max_f], p0_Vt_func, V_th_error[Min_f:Max_f], absolute_sigma=True) 	
	 
#Vt_func_s = optimization.curve_fit(Vt_func, sqrt(2.0*f_f(Nd) + V_back[Min_f:Max_f]), V_th[Min_f:Max_f], p0_Vt_func, V_th_error[Min_f:Max_f], absolute_sigma=True)[0][0]
#Vt_func_s_error = sqrt(optimization.curve_fit(Vt_func, sqrt(2.0*f_f(Nd) + V_back[Min_f:Max_f]), V_th[Min_f:Max_f], p0_Vt_func, V_th_error[Min_f:Max_f], absolute_sigma=True)[1][0][0])

#Vt_func_i = optimization.curve_fit(Vt_func, sqrt(2.0*f_f(Nd) + V_back[Min_f:Max_f]), V_th[Min_f:Max_f], p0_Vt_func, V_th_error[Min_f:Max_f], absolute_sigma=True)[0][1] 	 	 
#Vt_func_i_error = sqrt(optimization.curve_fit(Vt_func, sqrt(2.0*f_f(Nd) + V_back[Min_f:Max_f]), V_th[Min_f:Max_f], p0_Vt_func, V_th_error[Min_f:Max_f], absolute_sigma=True)[1][1][1])

#N_ext = ((Vt_func_s**2.0)*Cox**2.0)/(2.0*q0*epsilonSi*epsilon0)*1e-12

#print "The extracted doping is:",  N_ext, "."




############################################################################

figure(1)
PLOT_IV = plot(Vol1, Curr1*1e6, "--", label = "$V_{back}$ = 0 V",  markersize = 5, color='blue', linewidth = 2.2)
PLOT_IV = plot(Vol2, Curr2*1e6, "--", label = "$V_{back}$ = -5 V",  markersize = 5, color='blueviolet', linewidth = 2.2)
PLOT_IV = plot(Vol3, Curr3*1e6, "--", label = "$V_{back}$ = -10 V",  markersize = 5, color='brown', linewidth = 2.2)
PLOT_IV = plot(Vol4, Curr4*1e6, "--", label = "$V_{back}$ = -15 V",  markersize = 5, color='burlywood', linewidth = 2.2)
PLOT_IV = plot(Vol5, Curr5*1e6, "--", label = "$V_{back}$ = -20 V",  markersize = 5, color='cadetblue', linewidth = 2.2)
PLOT_IV = plot(Vol6, Curr6*1e6, "--", label = "$V_{back}$ = -25 V",  markersize = 5, color='chocolate', linewidth = 2.2)
PLOT_IV = plot(Vol7, Curr7*1e6, "--", label = "$V_{back}$ = -30 V",  markersize = 5, color='coral', linewidth = 2.2)
PLOT_IV = plot(Vol8, Curr8*1e6, "--", label = "$V_{back}$ = -35 V",  markersize = 5, color='cornflowerblue', linewidth = 2.2)
PLOT_IV = plot(Vol9, Curr9*1e6, "--", label = "$V_{back}$ = -40 V",  markersize = 5, color='red', linewidth = 2.2)
PLOT_IV = plot(Vol10, Curr10*1e6, "--", label = "$V_{back}$ = -45 V",  markersize = 5, color='crimson', linewidth = 2.2)
PLOT_IV = plot(Vol11, Curr11*1e6, "--", label = "$V_{back}$ = -50 V",  markersize = 5, color='cyan', linewidth = 2.2)
PLOT_IV = plot(Vol12, Curr12*1e6, "--", label = "$V_{back}$ = -55 V",  markersize = 5, color='darkblue', linewidth = 2.2)
PLOT_IV = plot(Vol13, Curr13*1e6, "--", label = "$V_{back}$ = -60 V",  markersize = 5, color='darkcyan', linewidth = 2.2)
PLOT_IV = plot(Vol14, Curr14*1e6, "--", label = "$V_{back}$ = -65 V",  markersize = 5, color='darkgoldenrod', linewidth = 2.2)
PLOT_IV = plot(Vol15, Curr15*1e6, "--", label = "$V_{back}$ = -70 V",  markersize = 5, color='darkgreen', linewidth = 2.2)
PLOT_IV = plot(Vol16, Curr16*1e6, "--", label = "$V_{back}$ = -75 V",  markersize = 5, color='darkkhaki', linewidth = 2.2)
PLOT_IV = plot(Vol17, Curr17*1e6, "--", label = "$V_{back}$ = -80 V",  markersize = 5, color='gold', linewidth = 2.2)
PLOT_IV = plot(Vol18, Curr18*1e6, "--", label = "$V_{back}$ = -85 V",  markersize = 5, color='hotpink', linewidth = 2.2)
PLOT_IV = plot(Vol19, Curr19*1e6, "--", label = "$V_{back}$ = -90 V",  markersize = 5, color='lightpink', linewidth = 2.2)
PLOT_IV = plot(Vol20, Curr20*1e6, "--", label = "$V_{back}$ = -95 V",  markersize = 5, color='orange', linewidth = 2.2)
PLOT_IV = plot(Vol21, Curr21*1e6, "--", label = "$V_{back}$ = -100 V",  markersize = 5, color='rosybrown', linewidth = 2.2)
PLOT_IV = plot(Vol22, Curr22*1e6, "--", label = "$V_{back}$ = -105 V",  markersize = 5, color='salmon', linewidth = 2.2)
PLOT_IV = plot(Vol23, Curr23*1e6, "--", label = "$V_{back}$ = -110 V",  markersize = 5, color='steelblue', linewidth = 2.2)
PLOT_IV = plot(Vol24, Curr24*1e6, "--", label = "$V_{back}$ = -115 V",  markersize = 5, color='thistle', linewidth = 2.2)
PLOT_IV = plot(Vol25, Curr25*1e6, "--", label = "$V_{back}$ = -120 V",  markersize = 5, color='violet', linewidth = 2.2)

#title(TITLE, fontsize = 18, fontproperties = fp)
legend(loc=2, frameon=True, prop = FontProperties(size=14, family = FONT, fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"), numpoints=1)
#legend(bbox_to_anchor=(0.75,1.1), loc=2, borderaxespad=0.,prop = FontProperties(size=10, family = FONT, fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"), numpoints=1)
xlabel("V$_{gate}$ [V]", fontsize = 18, fontproperties = fp, fontweight='bold')
ylabel("I$_{ds}$ [$\mu$A]", fontsize = 18, fontproperties = fp, fontweight='bold')
grid(True)
#yscale("log")
# Setup the range of two axis
xticks(fontsize=16)
yticks(fontsize=16)
#yscale("log")
# Setup the range of two axis
#xlim(0,1.60)
#ylim(5*10**(-12),1*10**(-5))

fig = plt.figure(2)
ax = fig.add_subplot(111)
gs = gridspec.GridSpec(4,1)
PLOT_CAP = plot(Vr_plot_1, Cur_plot_1*1e6, "--", label = "data $V_{back}$ = 0 V", markersize = 4, color='blue', linewidth = 1.8)
PLOT_CAP = plot(Vr_1, Id_plot(Vr_1,Alpha_1,beta_1,V_th_1)*1e6, "-", label = "fit $V_{back}$ = 0 V", markersize = 4, color='blue', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_2, Cur_plot_2*1e6, "--", label = "data $V_{back}$ = -0.5 V", markersize = 4, color='blueviolet', linewidth = 1.8)
PLOT_CAP = plot(Vr_2, Id_plot(Vr_2,Alpha_2,beta_2,V_th_2)*1e6, "-", label = "fit $V_{back}$ = -0.5 V", markersize = 4, color='blueviolet', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_3, Cur_plot_3*1e6, "--", label = "data $V_{back}$ = -1.0 V", markersize = 4, color='brown', linewidth = 1.8)
PLOT_CAP = plot(Vr_3, Id_plot(Vr_3,Alpha_3,beta_3,V_th_3)*1e6, "-", label = "fit $V_{back}$ = -1.0 V", markersize = 4, color='brown', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_4, Cur_plot_4*1e6, "--", label = "data $V_{back}$ = -2.0 V", markersize = 4, color='burlywood', linewidth = 1.8)
PLOT_CAP = plot(Vr_4, Id_plot(Vr_4,Alpha_4,beta_4,V_th_4)*1e6, "-", label = "fit $V_{back}$ = -2.0 V", markersize = 4, color='burlywood', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_5, Cur_plot_5*1e6, "--", label = "data $V_{back}$ = -3.0 V", markersize = 4, color='cadetblue', linewidth = 1.8)
PLOT_CAP = plot(Vr_5, Id_plot(Vr_5,Alpha_5,beta_5,V_th_5)*1e6, "-", label = "fit $V_{back}$ = -3.0 V", markersize = 4, color='cadetblue', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_6, Cur_plot_6*1e6, "--", label = "data $V_{back}$ = -5.0 V", markersize = 4, color='chocolate', linewidth = 1.8)
PLOT_CAP = plot(Vr_6, Id_plot(Vr_6,Alpha_6,beta_6,V_th_6)*1e6, "-", label = "fit $V_{back}$ = -5.0 V", markersize = 4, color='chocolate', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_7, Cur_plot_7*1e6, "--", label = "data $V_{back}$ = -10.0 V", markersize = 4, color='coral', linewidth = 1.8)
PLOT_CAP = plot(Vr_7, Id_plot(Vr_7,Alpha_7,beta_7,V_th_7)*1e6, "-", label = "fit $V_{back}$ = -10.0 V", markersize = 4, color='coral', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_8, Cur_plot_8*1e6, "--", label = "data $V_{back}$ = -2.0 V", markersize = 4, color='cornflowerblue', linewidth = 1.8)
PLOT_CAP = plot(Vr_8, Id_plot(Vr_8,Alpha_8,beta_8,V_th_8)*1e6, "-", label = "fit $V_{back}$ = -2.0 V", markersize = 4, color='cornflowerblue', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_9, Cur_plot_9*1e6, "--", label = "data $V_{back}$ = -3.0 V", markersize = 4, color='red', linewidth = 1.8)
PLOT_CAP = plot(Vr_9, Id_plot(Vr_9,Alpha_9,beta_9,V_th_9)*1e6, "-", label = "fit $V_{back}$ = -3.0 V", markersize = 4, color='red', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_10, Cur_plot_10*1e6, "--", label = "data $V_{back}$ = -5.0 V", markersize = 4, color='crimson', linewidth = 1.8)
PLOT_CAP = plot(Vr_10, Id_plot(Vr_10,Alpha_10,beta_10,V_th_10)*1e6, "-", label = "fit $V_{back}$ = -5.0 V", markersize = 4, color='crimson', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_11, Cur_plot_11*1e6, "--", label = "data $V_{back}$ = -10.0 V", markersize = 4, color='cyan', linewidth = 1.8)
PLOT_CAP = plot(Vr_11, Id_plot(Vr_11,Alpha_11,beta_11,V_th_11)*1e6, "-", label = "fit $V_{back}$ = -10.0 V", markersize = 4, color='cyan', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_12, Cur_plot_12*1e6, "--", label = "data $V_{back}$ = -0.5 V", markersize = 4, color='darkblue', linewidth = 1.8)
PLOT_CAP = plot(Vr_12, Id_plot(Vr_12,Alpha_12,beta_12,V_th_12)*1e6, "-", label = "fit $V_{back}$ = -0.5 V", markersize = 4, color='darkblue', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_13, Cur_plot_13*1e6, "--", label = "data $V_{back}$ = -1.0 V", markersize = 4, color='darkcyan', linewidth = 1.8)
PLOT_CAP = plot(Vr_13, Id_plot(Vr_13,Alpha_13,beta_13,V_th_13)*1e6, "-", label = "fit $V_{back}$ = -1.0 V", markersize = 4, color='darkcyan', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_14, Cur_plot_14*1e6, "--", label = "data $V_{back}$ = -2.0 V", markersize = 4, color='darkgoldenrod', linewidth = 1.8)
PLOT_CAP = plot(Vr_14, Id_plot(Vr_14,Alpha_14,beta_14,V_th_14)*1e6, "-", label = "fit $V_{back}$ = -2.0 V", markersize = 4, color='darkgoldenrod', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_15, Cur_plot_15*1e6, "--", label = "data $V_{back}$ = -3.0 V", markersize = 4, color='darkgreen', linewidth = 1.8)
PLOT_CAP = plot(Vr_15, Id_plot(Vr_15,Alpha_15,beta_15,V_th_15)*1e6, "-", label = "fit $V_{back}$ = -3.0 V", markersize = 4, color='darkgreen', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_16, Cur_plot_16*1e6, "--", label = "data $V_{back}$ = -5.0 V", markersize = 4, color='darkkhaki', linewidth = 1.8)
PLOT_CAP = plot(Vr_16, Id_plot(Vr_16,Alpha_16,beta_16,V_th_16)*1e6, "-", label = "fit $V_{back}$ = -5.0 V", markersize = 4, color='darkkhaki', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_17, Cur_plot_17*1e6, "--", label = "data $V_{back}$ = -10.0 V", markersize = 4, color='gold', linewidth = 1.8)
PLOT_CAP = plot(Vr_17, Id_plot(Vr_17,Alpha_17,beta_17,V_th_17)*1e6, "-", label = "fit $V_{back}$ = -10.0 V", markersize = 4, color='gold', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_18, Cur_plot_18*1e6, "--", label = "data $V_{back}$ = -2.0 V", markersize = 4, color='hotpink', linewidth = 1.8)
PLOT_CAP = plot(Vr_18, Id_plot(Vr_18,Alpha_18,beta_18,V_th_18)*1e6, "-", label = "fit $V_{back}$ = -2.0 V", markersize = 4, color='hotpink', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_19, Cur_plot_19*1e6, "--", label = "data $V_{back}$ = -3.0 V", markersize = 4, color='lightpink', linewidth = 1.8)
PLOT_CAP = plot(Vr_19, Id_plot(Vr_19,Alpha_19,beta_19,V_th_19)*1e6, "-", label = "fit $V_{back}$ = -3.0 V", markersize = 4, color='lightpink', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_20, Cur_plot_20*1e6, "--", label = "data $V_{back}$ = -5.0 V", markersize = 4, color='orange', linewidth = 1.8)
PLOT_CAP = plot(Vr_20, Id_plot(Vr_20,Alpha_20,beta_20,V_th_20)*1e6, "-", label = "fit $V_{back}$ = -5.0 V", markersize = 4, color='orange', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_21, Cur_plot_21*1e6, "--", label = "data $V_{back}$ = -10.0 V", markersize = 4, color='rosybrown', linewidth = 1.8)
PLOT_CAP = plot(Vr_21, Id_plot(Vr_21,Alpha_21,beta_21,V_th_21)*1e6, "-", label = "fit $V_{back}$ = -10.0 V", markersize = 4, color='rosybrown', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_22, Cur_plot_22*1e6, "--", label = "data $V_{back}$ = -2.0 V", markersize = 4, color='salmon', linewidth = 1.8)
PLOT_CAP = plot(Vr_22, Id_plot(Vr_22,Alpha_22,beta_22,V_th_22)*1e6, "-", label = "fit $V_{back}$ = -2.0 V", markersize = 4, color='salmon', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_23, Cur_plot_23*1e6, "--", label = "data $V_{back}$ = -3.0 V", markersize = 4, color='steelblue', linewidth = 1.8)
PLOT_CAP = plot(Vr_23, Id_plot(Vr_23,Alpha_23,beta_23,V_th_23)*1e6, "-", label = "fit $V_{back}$ = -3.0 V", markersize = 4, color='steelblue', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_24, Cur_plot_24*1e6, "--", label = "data $V_{back}$ = -5.0 V", markersize = 4, color='thistle', linewidth = 1.8)
PLOT_CAP = plot(Vr_24, Id_plot(Vr_24,Alpha_24,beta_24,V_th_24)*1e6, "-", label = "fit $V_{back}$ = -5.0 V", markersize = 4, color='thistle', linewidth = 1.8)

PLOT_CAP = plot(Vr_plot_25, Cur_plot_25*1e6, "--", label = "data $V_{back}$ = -10.0 V", markersize = 4, color='violet', linewidth = 1.8)
PLOT_CAP = plot(Vr_25, Id_plot(Vr_25,Alpha_25,beta_25,V_th_25)*1e6, "-", label = "fit $V_{back}$ = -10.0 V", markersize = 4, color='violet', linewidth = 1.8)

#title(TITLE, fontsize = 18, fontproperties = fp)
#xlabel("$-V_{gate}$ [$V$]", fontsize = 12, fontproperties = fp, fontweight='bold')
ylabel("$I_{ds}$ [$\mu A$]", fontsize = 12, fontproperties = fp, fontweight='bold')
#legend(loc=2, frameon=True, prop = FontProperties(size=7.5, family = FONT, fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"), numpoints=1)
grid(True)
#yscale("log")
# Setup the range of two axis
#xlim(0,-4.0)
ax.set_position(gs[0:3].get_position(fig))
ax.set_subplotspec(gs[0:3]) 

xticks(fontsize=1)
yticks(fontsize=16)
xlim(-7,20.0)
ylim(-2.0, 25.0)

fig.add_subplot(gs[3:4])
PLOT_CAP = plot(Vr_1, np.array(Id_plot(Vr_1,Alpha_1,beta_1,V_th_1)/Cur_1), "-", label = "$V_{back}$ = 0 V", markersize = 4, color='blue', linewidth = 1.8)
PLOT_CAP = plot(Vr_2, np.array(Id_plot(Vr_2,Alpha_2,beta_2,V_th_2)/Cur_2), "-", label = "$V_{back}$ = -0.5 V", markersize = 4, color='blueviolet', linewidth = 1.8)
PLOT_CAP = plot(Vr_3, np.array(Id_plot(Vr_3,Alpha_3,beta_3,V_th_3)/Cur_3), "-", label = "$V_{back}$ = -1.0 V", markersize = 4, color='brown', linewidth = 1.8)
PLOT_CAP = plot(Vr_4, np.array(Id_plot(Vr_4,Alpha_4,beta_4,V_th_4)/Cur_4), "-", label = "$V_{back}$ = -2.0 V", markersize = 4, color='burlywood', linewidth = 1.8)
PLOT_CAP = plot(Vr_5, np.array(Id_plot(Vr_5,Alpha_5,beta_5,V_th_5)/Cur_5), "-", label = "$V_{back}$ = -3.0 V", markersize = 4, color='cadetblue', linewidth = 1.8)
PLOT_CAP = plot(Vr_6, np.array(Id_plot(Vr_6,Alpha_6,beta_6,V_th_6)/Cur_6), "-", label = "$V_{back}$ = -5.0 V", markersize = 4, color='chocolate', linewidth = 1.8)
PLOT_CAP = plot(Vr_7, np.array(Id_plot(Vr_7,Alpha_7,beta_7,V_th_7)/Cur_7), "-", label = "$V_{back}$ = -10.0 V", markersize = 4, color='coral', linewidth = 1.8)
PLOT_CAP = plot(Vr_8, np.array(Id_plot(Vr_8,Alpha_8,beta_8,V_th_8)/Cur_8), "-", label = "$V_{back}$ = -3.0 V", markersize = 4, color='cornflowerblue', linewidth = 1.8)
PLOT_CAP = plot(Vr_9, np.array(Id_plot(Vr_9,Alpha_9,beta_9,V_th_9)/Cur_9), "-", label = "$V_{back}$ = -5.0 V", markersize = 4, color='red', linewidth = 1.8)
PLOT_CAP = plot(Vr_10, np.array(Id_plot(Vr_10,Alpha_10,beta_10,V_th_10)/Cur_10), "-", label = "$V_{back}$ = -10.0 V", markersize = 4, color='crimson', linewidth = 1.8)
PLOT_CAP = plot(Vr_11, np.array(Id_plot(Vr_11,Alpha_11,beta_11,V_th_11)/Cur_11), "-", label = "$V_{back}$ = -3.0 V", markersize = 4, color='cyan', linewidth = 1.8)
PLOT_CAP = plot(Vr_12, np.array(Id_plot(Vr_12,Alpha_12,beta_12,V_th_12)/Cur_12), "-", label = "$V_{back}$ = -5.0 V", markersize = 4, color='darkblue', linewidth = 1.8)
PLOT_CAP = plot(Vr_13, np.array(Id_plot(Vr_13,Alpha_13,beta_13,V_th_13)/Cur_13), "-", label = "$V_{back}$ = -10.0 V", markersize = 4, color='darkcyan', linewidth = 1.8)
PLOT_CAP = plot(Vr_14, np.array(Id_plot(Vr_14,Alpha_14,beta_14,V_th_14)/Cur_14), "-", label = "$V_{back}$ = -3.0 V", markersize = 4, color='darkgoldenrod', linewidth = 1.8)
PLOT_CAP = plot(Vr_15, np.array(Id_plot(Vr_15,Alpha_15,beta_15,V_th_15)/Cur_15), "-", label = "$V_{back}$ = -5.0 V", markersize = 4, color='darkgreen', linewidth = 1.8)
PLOT_CAP = plot(Vr_16, np.array(Id_plot(Vr_16,Alpha_16,beta_16,V_th_16)/Cur_16), "-", label = "$V_{back}$ = -10.0 V", markersize = 4, color='darkkhaki', linewidth = 1.8)
PLOT_CAP = plot(Vr_17, np.array(Id_plot(Vr_17,Alpha_17,beta_17,V_th_17)/Cur_17), "-", label = "$V_{back}$ = -3.0 V", markersize = 4, color='gold', linewidth = 1.8)
PLOT_CAP = plot(Vr_18, np.array(Id_plot(Vr_18,Alpha_18,beta_18,V_th_18)/Cur_18), "-", label = "$V_{back}$ = -5.0 V", markersize = 4, color='hotpink', linewidth = 1.8)
PLOT_CAP = plot(Vr_19, np.array(Id_plot(Vr_19,Alpha_19,beta_19,V_th_19)/Cur_19), "-", label = "$V_{back}$ = -10.0 V", markersize = 4, color='lightpink', linewidth = 1.8)
PLOT_CAP = plot(Vr_20, np.array(Id_plot(Vr_20,Alpha_20,beta_20,V_th_20)/Cur_20), "-", label = "$V_{back}$ = -3.0 V", markersize = 4, color='orange', linewidth = 1.8)
PLOT_CAP = plot(Vr_21, np.array(Id_plot(Vr_21,Alpha_21,beta_21,V_th_21)/Cur_21), "-", label = "$V_{back}$ = -5.0 V", markersize = 4, color='rosybrown', linewidth = 1.8)
PLOT_CAP = plot(Vr_22, np.array(Id_plot(Vr_22,Alpha_22,beta_22,V_th_22)/Cur_22), "-", label = "$V_{back}$ = -10.0 V", markersize = 4, color='salmon', linewidth = 1.8)
PLOT_CAP = plot(Vr_23, np.array(Id_plot(Vr_23,Alpha_23,beta_23,V_th_23)/Cur_23), "-", label = "$V_{back}$ = -3.0 V", markersize = 4, color='steelblue', linewidth = 1.8)
PLOT_CAP = plot(Vr_24, np.array(Id_plot(Vr_24,Alpha_24,beta_24,V_th_24)/Cur_24), "-", label = "$V_{back}$ = -5.0 V", markersize = 4, color='thistle', linewidth = 1.8)
PLOT_CAP = plot(Vr_25, np.array(Id_plot(Vr_25,Alpha_25,beta_25,V_th_25)/Cur_25), "-", label = "$V_{back}$ = -10.0 V", markersize = 4, color='violet', linewidth = 1.8)

#title(TITLE, fontsize = 18, fontproperties = fp)
xlabel("$V_{gate}$ [$V$]", fontsize = 12, fontproperties = fp, fontweight='bold')
ylabel('fit/data', fontsize = 12, fontproperties = fp, fontweight='bold')
#legend(loc=2, frameon=True, prop = FontProperties(size=7, family = FONT, fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"), numpoints=1)
grid(True)
#yscale("log")
# Setup the range of two axis
xticks(fontsize=16)
yticks(fontsize=16)
plt.subplots_adjust(hspace=0.2, bottom=0.125)
pyplot.locator_params(axis = 'y', nbins = 5)
xlim(-7,20.0)
ylim(0.99, 1.01)


figure(3)
PLOT_IV = plot(sqrt(2.0*f_f(Nd) + V_back), V_th, "o", label = "",  markersize = 5, color='b', linewidth = 2.0)
#PLOT_IV = plot(sqrt(2.0*f_f(Nd) + V_back), Vt_func(sqrt(2.0*f_f(Nd) + V_back), Vt_func_s, Vt_func_i), "-", label = "fit",  markersize = 5, color='r', linewidth = 2.0)

#title(TITLE, fontsize = 18, fontproperties = fp)
#legend(loc=2, frameon=True, prop = FontProperties(size=14, family = FONT, fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"), numpoints=1)
xlabel("$\sqrt{(2 \phi_{f} +V_{back}}$ [V]", fontsize = 18, fontproperties = fp, fontweight='bold')
ylabel("V$_{th}$ [V]", fontsize = 18, fontproperties = fp, fontweight='bold')
grid(True)
#ylim(-3.0, 0)
#yscale("log")
# Setup the range of two axis
xticks(fontsize=16)
yticks(fontsize=16)

figure(4)
PLOT_IV = plot(V_back, m0, "o-", label = "",  markersize = 5, color='b', linewidth = 2.0)

#title(TITLE, fontsize = 18, fontproperties = fp)
#legend(loc=3, frameon=True, prop = FontProperties(size=14, family = FONT, fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"), numpoints=1)
xlabel("-V$_{back}$ [V]", fontsize = 18, fontproperties = fp, fontweight='bold')
ylabel("$\mu_{0}$ [$cm^2$/V*sec]", fontsize = 18, fontproperties = fp, fontweight='bold')
grid(True)
ylim(0, 1450)
#yscale("log")
# Setup the range of two axis
xticks(fontsize=16)
yticks(fontsize=16)

figure(5)
PLOT_IV = plot(V_back, V_half, "o-", label = "",  markersize = 5, color='b', linewidth = 2.0)
#title(TITLE, fontsize = 18, fontproperties = fp)
#legend(loc=2, frameon=True, prop = FontProperties(size=14, family = FONT, fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"), numpoints=1)
xlabel("-V$_{back}$ [V]", fontsize = 18, fontproperties = fp, fontweight='bold')
ylabel("V$_{1/2}$ [V]", fontsize = 18, fontproperties = fp, fontweight='bold')
grid(True)
ylim(0, 100)
#yscale("log")
# Setup the range of two axis
xticks(fontsize=16)
yticks(fontsize=16)


show()





