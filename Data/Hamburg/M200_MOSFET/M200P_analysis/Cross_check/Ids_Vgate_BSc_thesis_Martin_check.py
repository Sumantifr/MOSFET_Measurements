# Extraction of the doping concentration using MOSFETs
# Author Ioannis
# Date 01/11/16
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

FILE1 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_41.iv"
FILE2 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_42.iv"
FILE3 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_43.iv"
FILE4 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_44.iv"
FILE5 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_45.iv"
FILE6 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_46.iv"
FILE7 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_47.iv"
FILE8 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_48.iv"
FILE9 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_49.iv"
FILE10 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_50.iv"
FILE11 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_51.iv"
FILE12 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_52.iv"
FILE13 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_53.iv"
FILE14 = "M200P_04_HPKTS_1_MOSFET_2017-02-24_54.iv"
#FILE15 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_25.iv"
#FILE16 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_26.iv"
#FILE17 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_27.iv"
#FILE18 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_28.iv"
#FILE19 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_29.iv"
#FILE20 = "M200P_04_HPKTS_1_MOSFET_2017-01-15_30.iv"
#FILE21 = "M200P_04_HPKTS_1_MOSFET_2017-01-16_31.iv"
#FILE22 = "M200P_04_HPKTS_1_MOSFET_2017-01-16_32.iv"
#FILE23 = "M200P_04_HPKTS_1_MOSFET_2017-01-16_33.iv"
#FILE24 = "M200P_04_HPKTS_1_MOSFET_2017-01-16_34.iv"
#FILE25 = "M200P_04_HPKTS_1_MOSFET_2017-01-16_35.iv"


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
#MAT_DATA15 = loadtxt(FILE15, skiprows = 37, comments = "END")
#MAT_DATA16 = loadtxt(FILE16, skiprows = 37, comments = "END")
#MAT_DATA17 = loadtxt(FILE17, skiprows = 37, comments = "END")
#MAT_DATA18 = loadtxt(FILE18, skiprows = 37, comments = "END")
#MAT_DATA19 = loadtxt(FILE19, skiprows = 37, comments = "END")
#MAT_DATA20 = loadtxt(FILE20, skiprows = 37, comments = "END")
#MAT_DATA21 = loadtxt(FILE21, skiprows = 37, comments = "END")
#MAT_DATA22 = loadtxt(FILE22, skiprows = 37, comments = "END")
#MAT_DATA23 = loadtxt(FILE23, skiprows = 37, comments = "END")
#MAT_DATA24 = loadtxt(FILE24, skiprows = 37, comments = "END")
#MAT_DATA25 = loadtxt(FILE25, skiprows = 37, comments = "END")

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
#Vol15 = MAT_DATA15[:,0]
#Vol16 = MAT_DATA16[:,0]
#Vol17 = MAT_DATA17[:,0]
#Vol18 = MAT_DATA18[:,0]
#Vol19 = MAT_DATA19[:,0]
#Vol20 = MAT_DATA20[:,0]
#Vol21 = MAT_DATA21[:,0]
#Vol22 = MAT_DATA22[:,0]
#Vol23 = MAT_DATA23[:,0]
#Vol24 = MAT_DATA24[:,0]
#Vol25 = MAT_DATA25[:,0]


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
#Temp15 = mean(MAT_DATA15[:,1])
#Temp16 = mean(MAT_DATA16[:,1])
#Temp17 = mean(MAT_DATA17[:,1])
#Temp18 = mean(MAT_DATA18[:,1])
#Temp19 = mean(MAT_DATA19[:,1])
#Temp20 = mean(MAT_DATA20[:,1])
#Temp21 = mean(MAT_DATA21[:,1])
#Temp22 = mean(MAT_DATA22[:,1])
#Temp23 = mean(MAT_DATA23[:,1])
#Temp24 = mean(MAT_DATA24[:,1])
#Temp25 = mean(MAT_DATA25[:,1])


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
#print "The measurement was performed at", Temp15, "degree C"
#print "The measurement was performed at", Temp16, "degree C"
#print "The measurement was performed at", Temp17, "degree C"
#print "The measurement was performed at", Temp18, "degree C"
#print "The measurement was performed at", Temp19, "degree C"
#print "The measurement was performed at", Temp20, "degree C"
#print "The measurement was performed at", Temp21, "degree C"
#print "The measurement was performed at", Temp22, "degree C"
#print "The measurement was performed at", Temp23, "degree C"
#print "The measurement was performed at", Temp24, "degree C"
#print "The measurement was performed at", Temp25, "degree C"

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
#Curr15 = MAT_DATA15[:,2]
#Curr16 = MAT_DATA16[:,2]
#Curr17 = MAT_DATA17[:,2]
#Curr18 = MAT_DATA18[:,2]
#Curr19 = MAT_DATA19[:,2]
#Curr20 = MAT_DATA20[:,2]
#Curr21 = MAT_DATA21[:,2]
#Curr22 = MAT_DATA22[:,2]
#Curr23 = MAT_DATA23[:,2]
#Curr24 = MAT_DATA24[:,2]
#Curr25 = MAT_DATA25[:,2]

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
Max_plot_1 = 111                                    # Region of plot, upper limit
Vr_plot_1 = np.array(Vol1[Min_plot_1:Max_plot_1])   # Voltage region
Cur_plot_1 = np.array(Curr1[Min_plot_1:Max_plot_1]) # Current region

###############################################################

###### Optimization for V_back = 0 V ###########################

Min_1 = 28                                          # Region of fit, lower limit
Max_1 = 111                                         # Region of fit, upper limit
w_1 = 0.002                                         # Weight for the data fit
Vr_1 = np.array(Vol1[Min_1:Max_1])                  # Voltage region
Cur_1 = np.array(Curr1[Min_1:Max_1])                # Current region             
                                         
#################################################################


########### Plot for V_back = -0.5 V #############################

Min_plot_2 = 0                                      # Region of plot, lower limit
Max_plot_2 = 111                                    # Region of plot, upper limit
Vr_plot_2 = np.array(Vol2[Min_plot_2:Max_plot_2])   # Voltage region
Cur_plot_2 = np.array(Curr2[Min_plot_2:Max_plot_2]) # Current region

###############################################################

###### Optimization for V_back = -0.5 V ###########################

Min_2 = 28                                          # Region of fit, lower limit
Max_2 = 111                                         # Region of fit, upper limit
w_2 = 0.002                                         # Weight for the data fit
Vr_2 = np.array(Vol2[Min_2:Max_2])                  # Voltage region
Cur_2 = np.array(Curr2[Min_2:Max_2])                # Current region             
                                         
#################################################################


########### Plot for V_back = -1.0 V #############################

Min_plot_3 = 0                                      # Region of plot, lower limit
Max_plot_3 = 111                                    # Region of plot, upper limit
Vr_plot_3 = np.array(Vol3[Min_plot_3:Max_plot_3])   # Voltage region
Cur_plot_3 = np.array(Curr3[Min_plot_3:Max_plot_3]) # Current region

###############################################################

###### Optimization for V_back = -1.0 V ###########################

Min_3 = 28                                          # Region of fit, lower limit
Max_3 = 111                                         # Region of fit, upper limit
w_3 = 0.002                                         # Weight for the data fit
Vr_3 = np.array(Vol3[Min_3:Max_3])                  # Voltage region
Cur_3 = np.array(Curr3[Min_3:Max_3])                # Current region             
                                         
#################################################################


########### Plot for V_back = -2.0 V #############################

Min_plot_4 = 0                                      # Region of plot, lower limit
Max_plot_4 = 111                                    # Region of plot, upper limit
Vr_plot_4 = np.array(Vol4[Min_plot_4:Max_plot_4])   # Voltage region
Cur_plot_4 = np.array(Curr4[Min_plot_4:Max_plot_4]) # Current region

###############################################################

###### Optimization for V_back = -2.0 V ###########################

Min_4 = 28                                          # Region of fit, lower limit
Max_4 = 111                                         # Region of fit, upper limit
w_4 = 0.002                                         # Weight for the data fit
Vr_4 = np.array(Vol4[Min_4:Max_4])                  # Voltage region
Cur_4 = np.array(Curr4[Min_4:Max_4])                # Current region             
                                         
#################################################################

########### Plot for V_back = -3.0 V #############################

Min_plot_5 = 0                                      # Region of plot, lower limit
Max_plot_5 = 111                                    # Region of plot, upper limit
Vr_plot_5 = np.array(Vol5[Min_plot_5:Max_plot_5])   # Voltage region
Cur_plot_5 = np.array(Curr5[Min_plot_5:Max_plot_5]) # Current region

###############################################################

###### Optimization for V_back = -3.0 V ###########################

Min_5 = 28                                          # Region of fit, lower limit
Max_5 = 111                                         # Region of fit, upper limit
w_5 = 0.002                                         # Weight for the data fit
Vr_5 = np.array(Vol5[Min_5:Max_5])                  # Voltage region
Cur_5 = np.array(Curr5[Min_5:Max_5])                # Current region             
                                         
#################################################################

########### Plot for V_back = -5.0 V #############################

Min_plot_6 = 0                                      # Region of plot, lower limit
Max_plot_6 = 121                                    # Region of plot, upper limit
Vr_plot_6 = np.array(Vol6[Min_plot_6:Max_plot_6])   # Voltage region
Cur_plot_6 = np.array(Curr6[Min_plot_6:Max_plot_6]) # Current region

###############################################################

###### Optimization for V_back = -5.0 V ###########################

Min_6 = 38                                          # Region of fit, lower limit
Max_6 = 121                                         # Region of fit, upper limit
w_6 = 0.002                                         # Weight for the data fit
Vr_6 = np.array(Vol6[Min_6:Max_6])                  # Voltage region
Cur_6 = np.array(Curr6[Min_6:Max_6])                # Current region             
                                         
#################################################################

########### Plot for V_back = -10.0 V #############################

Min_plot_7 = 0                                      # Region of plot, lower limit
Max_plot_7 = 146                                    # Region of plot, upper limit
Vr_plot_7 = np.array(Vol7[Min_plot_7:Max_plot_7])   # Voltage region
Cur_plot_7 = np.array(Curr7[Min_plot_7:Max_plot_7]) # Current region

###############################################################

###### Optimization for V_back = -10.0 V ###########################

Min_7 = 63                                         # Region of fit, lower limit
Max_7 = 146                                         # Region of fit, upper limit
w_7 = 0.002                                         # Weight for the data fit
Vr_7 = np.array(Vol7[Min_7:Max_7])                  # Voltage region
Cur_7 = np.array(Curr7[Min_7:Max_7])                # Current region             
                                         
#################################################################


############ Curve fit function V_back = 0.0 V ###########################

#sigma_1 = np.array(Curr1[Min_1:Max_1]*w_1)
#p0_1 = np.array([1.52e-06, 0.0146, 2.341])
#print "Parameters Vback = 0 V", optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)

#Alpha_1 = optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[0][0]
#beta_1 = optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[0][1]
#V_th_1 =   optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -0.5 V ###########################

#sigma_2 = np.array(Curr2[Min_2:Max_2]*w_2)
#p0_2 = np.array([1.52e-06, 0.0146, 2.341])
#print "Parameters Vback = -0.5 V", optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)

#Alpha_2 = optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[0][0]
#beta_2 = optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[0][1]
#V_th_2 =   optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -1.0 V ###########################

#sigma_3 = np.array(Curr3[Min_3:Max_3]*w_3)
#p0_3 = np.array([1.52e-06, 0.0146, 2.341])
#print "Parameters Vback = -1.0 V", optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)

#Alpha_3 = optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[0][0]
#beta_3 = optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[0][1]
#V_th_3 =   optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -2.0 V ###########################

#sigma_4 = np.array(Curr4[Min_4:Max_4]*w_4)
#p0_4 = np.array([1.52e-06, 0.0146, 2.341])
#print "Parameters Vback = -2.0 V", optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)

#Alpha_4 = optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[0][0]
#beta_4 = optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[0][1]
#V_th_4 =   optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[0][2]

###########################################################################


############ Curve fit function V_back = -3.0 V ###########################

#sigma_5 = np.array(Curr5[Min_5:Max_5]*w_5)
#p0_5 = np.array([1.52e-06, 0.0146, 2.341])
#print "Parameters Vback = -3.0 V", optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)

#Alpha_5 = optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[0][0]
#beta_5 = optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[0][1]
#V_th_5 =   optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -5.0 V ###########################

#sigma_6 = np.array(Curr6[Min_6:Max_6]*w_6)
#p0_6 = np.array([ 1.52930864e-06,   2.42926553e-02,  -1.41510893e+00])
#print "Parameters Vback = -5.0 V", optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)

#Alpha_6 = optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[0][0]
#beta_6 = optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[0][1]
#V_th_6 =   optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[0][2]

###########################################################################

############ Curve fit function V_back = -10.0 V ###########################

#sigma_7 = np.array(Curr7[Min_7:Max_7]*w_7)
#p0_7 = np.array([ 1.52930864e-06,   2.42926553e-02,  -1.41510893e+00])
#print "Parameters Vback = -10.0 V", optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7, sigma_7, absolute_sigma=True)

#Alpha_7 = optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[0][0]
#beta_7 = optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[0][1]
#V_th_7 =   optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[0][2]

###########################################################################

######### Mobility for dif. backside voltage ############################## 

#print "The m_0 - Vback=0 V is:", Alpha_1*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
#print "The m_0 - Vback=-0.5 V is:", Alpha_2*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
#print "The m_0 - Vback=-1.0 V is:", Alpha_3*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
#print "The m_0 - Vback=-2.0 V is:", Alpha_4*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
#print "The m_0 - Vback=-3.0 V is:", Alpha_5*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
#print "The m_0 - Vback=-5.0 V is:", Alpha_6*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."
#print "The m_0 - Vback=-10.0 V is:", Alpha_7*Rev_fac_circ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[1][0][0])*Rev_fac_circ,"."

###########################################################################

########## V1/2 for dif. backside voltage #################################

#print "The V_1/2 - Vback=0 V is:", 1.0/beta_1,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[1][1][1])*(1.0/beta_1)**2,"."
#print "The V_1/2 - Vback=-0.5 V is:", 1.0/beta_2,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[1][1][1])*(1.0/beta_2)**2,"."
#print "The V_1/2 - Vback=-1.0 V is:", 1.0/beta_3,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[1][1][1])*(1.0/beta_3)**2,"."
#print "The V_1/2 - Vback=-2.0 V is:", 1.0/beta_4,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[1][1][1])*(1.0/beta_4)**2,"."
#print "The V_1/2 - Vback=-3.0 V is:", 1.0/beta_5,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[1][1][1])*(1.0/beta_5)**2,"."
#print "The V_1/2 - Vback=-5.0 V is:", 1.0/beta_6,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[1][1][1])*(1.0/beta_6)**2,"."
#print "The V_1/2 - Vback=-10.0 V is:", 1.0/beta_7,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[1][1][1])*(1.0/beta_7)**2,"."

###########################################################################

########## Vth for dif. backside voltage #################################

#print "The Vth - Vback=0 V is:", V_th_1 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[1][2][2]),"."
#print "The Vth - Vback=-0.5 V is:", V_th_2 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[1][2][2]),"."
#print "The Vth - Vback=-1.0 V is:", V_th_3 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[1][2][2]),"."
#print "The Vth - Vback=-2.0 V is:", V_th_4 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[1][2][2]),"."
#print "The Vth - Vback=-3.0 V is:", V_th_5 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[1][2][2]),"."
#print "The Vth - Vback=-5.0 V is:", V_th_6 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[1][2][2]),"."
#print "The Vth - Vback=-10.0V is:", V_th_7 ,"+-", sqrt(optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[1][2][2]),"."


#V_th_1_err = sqrt(optimization.curve_fit(Id_fit, Vr_1, Cur_1, p0_1 ,sigma_1, absolute_sigma=True)[1][2][2])
#V_th_2_err = sqrt(optimization.curve_fit(Id_fit, Vr_2, Cur_2, p0_2 ,sigma_2, absolute_sigma=True)[1][2][2])
#V_th_3_err = sqrt(optimization.curve_fit(Id_fit, Vr_3, Cur_3, p0_3 ,sigma_3, absolute_sigma=True)[1][2][2])
#V_th_4_err = sqrt(optimization.curve_fit(Id_fit, Vr_4, Cur_4, p0_4 ,sigma_4, absolute_sigma=True)[1][2][2])
#V_th_5_err = sqrt(optimization.curve_fit(Id_fit, Vr_5, Cur_5, p0_5 ,sigma_5, absolute_sigma=True)[1][2][2])
#V_th_6_err = sqrt(optimization.curve_fit(Id_fit, Vr_6, Cur_6, p0_6 ,sigma_6, absolute_sigma=True)[1][2][2])
#V_th_7_err = sqrt(optimization.curve_fit(Id_fit, Vr_7, Cur_7, p0_7 ,sigma_7, absolute_sigma=True)[1][2][2])
###########################################################################

################ Chi square ###############################################

#print "The chi_square - Vback= 0 V is:", Chi_sq(Cur_1,Id_plot(Vr_1,Alpha_1,beta_1,V_th_1),w_1), "."     
#print "The chi_square - Vback= -0.5 V is:", Chi_sq(Cur_2,Id_plot(Vr_2,Alpha_2,beta_2,V_th_2),w_2), "." 
#print "The chi_square - Vback= -1.0 V is:", Chi_sq(Cur_3,Id_plot(Vr_3,Alpha_3,beta_3,V_th_1),w_3), "."         
#print "The chi_square - Vback= -2.0 V is:", Chi_sq(Cur_4,Id_plot(Vr_4,Alpha_4,beta_4,V_th_4),w_4), "."   
#print "The chi_square - Vback= -3.0 V is:", Chi_sq(Cur_5,Id_plot(Vr_5,Alpha_5,beta_5,V_th_5),w_5), "."   
#print "The chi_square - Vback= -5.0 V is:", Chi_sq(Cur_6,Id_plot(Vr_6,Alpha_6,beta_6,V_th_6),w_6), "."   
#print "The chi_square - Vback= -10.0 V is:", Chi_sq(Cur_7,Id_plot(Vr_7,Alpha_7,beta_7,V_th_7),w_7), "."   

###########################################################################



######## Fit dunction #######################################################


#V_back = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
#m0 = np.array([Alpha_1, Alpha_2, Alpha_3, Alpha_4, Alpha_5, Alpha_6, Alpha_7]*Rev_fac_circ)
#V_half = np.array([1.0/beta_1, 1.0/beta_2, 1.0/beta_3, 1.0/beta_4, 1.0/beta_5, 1.0/beta_6, 1.0/beta_7])
#V_th = np.array([V_th_1, V_th_2, V_th_3, V_th_4, V_th_5, V_th_6, V_th_7])
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
PLOT_IV = plot(Vol1, Curr1*1e6, "-", label = "$V_{back}$ = 0 V",  markersize = 5, color='b', linewidth = 2.0)
PLOT_IV = plot(Vol2, Curr2*1e6, "-", label = "$V_{back}$ = -0.5 V",  markersize = 5, color='r', linewidth = 2.0)
PLOT_IV = plot(Vol3, Curr3*1e6, "-", label = "$V_{back}$ = -1.0 V",  markersize = 5, color='c', linewidth = 2.0)
PLOT_IV = plot(Vol4, Curr4*1e6, "-", label = "$V_{back}$ = -2.0 V",  markersize = 5, color='g', linewidth = 2.0)
PLOT_IV = plot(Vol5, Curr5*1e6, "-", label = "$V_{back}$ = -3.0 V",  markersize = 5, color='m', linewidth = 2.0)
PLOT_IV = plot(Vol6, Curr6*1e6, "-", label = "$V_{back}$ = -5.0 V",  markersize = 5, color='black', linewidth = 2.0)
PLOT_IV = plot(Vol7, Curr7*1e6, "-", label = "$V_{back}$ = -10.0 V",  markersize = 5, color='y', linewidth = 2.0)
PLOT_IV = plot(Vol8, Curr8*1e6, "-", label = "$V_{back}$ = -20.0 V",  markersize = 5, color='brown', linewidth = 2.0)
PLOT_IV = plot(Vol9, Curr9*1e6, "-", label = "$V_{back}$ = -30.0 V",  markersize = 5, color='gray', linewidth = 2.0)
PLOT_IV = plot(Vol10, Curr10*1e6, "-", label = "$V_{back}$ = 0.1 V",  markersize = 5, color='olive', linewidth = 2.0)
PLOT_IV = plot(Vol11, Curr11*1e6, "--", label = "$V_{back}$ = 0.2 V",  markersize = 5, color='b', linewidth = 2.0)
PLOT_IV = plot(Vol12, Curr12*1e6, "--", label = "$V_{back}$ = 0.3 V",  markersize = 5, color='r', linewidth = 2.0)
PLOT_IV = plot(Vol13, Curr13*1e6, "--", label = "$V_{back}$ = 0.4 V",  markersize = 5, color='c', linewidth = 2.0)
PLOT_IV = plot(Vol14, Curr14*1e6, "--", label = "$V_{back}$ = 0.5 V",  markersize = 5, color='g', linewidth = 2.0)
#PLOT_IV = plot(Vol15, Curr15*1e6, "-", label = "$V_{back}$ = -70 V",  markersize = 5, color='m', linewidth = 2.0)
#PLOT_IV = plot(Vol16, Curr16*1e6, "-", label = "$V_{back}$ = -75 V",  markersize = 5, color='black', linewidth = 2.0)
#PLOT_IV = plot(Vol17, Curr17*1e6, "-", label = "$V_{back}$ = -80 V",  markersize = 5, color='y', linewidth = 2.0)
#PLOT_IV = plot(Vol18, Curr18*1e6, "-", label = "$V_{back}$ = -85 V",  markersize = 5, color='brown', linewidth = 2.0)
#PLOT_IV = plot(Vol19, Curr19*1e6, "-", label = "$V_{back}$ = -90 V",  markersize = 5, color='gray', linewidth = 2.0)
#PLOT_IV = plot(Vol20, Curr20*1e6, "-", label = "$V_{back}$ = -95 V",  markersize = 5, color='olive', linewidth = 2.0)
#PLOT_IV = plot(Vol21, Curr21*1e6, "p--", label = "$V_{back}$ = -100 V",  markersize = 5, color='b', linewidth = 2.0)
#PLOT_IV = plot(Vol22, Curr22*1e6, "*--", label = "$V_{back}$ = -105 V",  markersize = 5, color='r', linewidth = 2.0)
#PLOT_IV = plot(Vol23, Curr23*1e6, ">--", label = "$V_{back}$ = -110 V",  markersize = 5, color='c', linewidth = 2.0)
#PLOT_IV = plot(Vol24, Curr24*1e6, "p--", label = "$V_{back}$ = -115 V",  markersize = 5, color='g', linewidth = 2.0)
#PLOT_IV = plot(Vol25, Curr25*1e6, "*--", label = "$V_{back}$ = -120 V",  markersize = 5, color='m', linewidth = 2.0)

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



#fig = plt.figure(2)

#ax = fig.add_subplot(111)
#gs = gridspec.GridSpec(4,1)

#PLOT_CAP = plot(Vr_plot_1, Cur_plot_1*1e6, "o", label = "data $V_{back}$ = 0 V", markersize = 4, color='b', linewidth = 1.8)
#PLOT_CAP = plot(Vr_1, Id_plot(Vr_1,Alpha_1,beta_1,V_th_1)*1e6, "-", label = "fit $V_{back}$ = 0 V", markersize = 4, color='b', linewidth = 1.8)

#PLOT_CAP = plot(Vr_plot_2, Cur_plot_2*1e6, "^", label = "data $V_{back}$ = -0.5 V", markersize = 4, color='r', linewidth = 1.8)
#PLOT_CAP = plot(Vr_2, Id_plot(Vr_2,Alpha_2,beta_2,V_th_2)*1e6, "-", label = "fit $V_{back}$ = -0.5 V", markersize = 4, color='r', linewidth = 1.8)

#PLOT_CAP = plot(Vr_plot_3, Cur_plot_3*1e6, "s", label = "data $V_{back}$ = -1.0 V", markersize = 4, color='c', linewidth = 1.8)
#PLOT_CAP = plot(Vr_3, Id_plot(Vr_3,Alpha_3,beta_3,V_th_3)*1e6, "-", label = "fit $V_{back}$ = -1.0 V", markersize = 4, color='c', linewidth = 1.8)

#PLOT_CAP = plot(Vr_plot_4, Cur_plot_4*1e6, "<", label = "data $V_{back}$ = -2.0 V", markersize = 4, color='g', linewidth = 1.8)
#PLOT_CAP = plot(Vr_4, Id_plot(Vr_4,Alpha_4,beta_4,V_th_4)*1e6, "-", label = "fit $V_{back}$ = -2.0 V", markersize = 4, color='g', linewidth = 1.8)

#PLOT_CAP = plot(Vr_plot_5, Cur_plot_5*1e6, ">", label = "data $V_{back}$ = -3.0 V", markersize = 4, color='m', linewidth = 1.8)
#PLOT_CAP = plot(Vr_5, Id_plot(Vr_5,Alpha_5,beta_5,V_th_5)*1e6, "-", label = "fit $V_{back}$ = -3.0 V", markersize = 4, color='m', linewidth = 1.8)

#PLOT_CAP = plot(Vr_plot_6, Cur_plot_6*1e6, "p", label = "data $V_{back}$ = -5.0 V", markersize = 4, color='black', linewidth = 1.8)
#PLOT_CAP = plot(Vr_6, Id_plot(Vr_6,Alpha_6,beta_6,V_th_6)*1e6, "-", label = "fit $V_{back}$ = -5.0 V", markersize = 4, color='black', linewidth = 1.8)

#PLOT_CAP = plot(Vr_plot_7, Cur_plot_7*1e6, "*", label = "data $V_{back}$ = -10.0 V", markersize = 4, color='y', linewidth = 1.8)
#PLOT_CAP = plot(Vr_7, Id_plot(Vr_7,Alpha_7,beta_7,V_th_7)*1e6, "-", label = "fit $V_{back}$ = -10.0 V", markersize = 4, color='y', linewidth = 1.8)

#title(TITLE, fontsize = 18, fontproperties = fp)
#xlabel("$-V_{gate}$ [$V$]", fontsize = 12, fontproperties = fp, fontweight='bold')
#ylabel("$I_{ds}$ [$\mu A$]", fontsize = 12, fontproperties = fp, fontweight='bold')
#legend(loc=2, frameon=True, prop = FontProperties(size=7.5, family = FONT, fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"), numpoints=1)
#grid(True)
#yscale("log")
# Setup the range of two axis
#xlim(0,-4.0)
#ax.set_position(gs[0:3].get_position(fig))
#ax.set_subplotspec(gs[0:3]) 

#xticks(fontsize=1)
#yticks(fontsize=16)
#xlim(-7,20.0)
#ylim(-1.0, 22.0)

#fig.add_subplot(gs[3:4])

#PLOT_CAP = plot(Vr_1, np.array(Id_plot(Vr_1,Alpha_1,beta_1,V_th_1)/Cur_1), "-", label = "$V_{back}$ = 0 V", markersize = 4, color='b', linewidth = 1.8)
#PLOT_CAP = plot(Vr_2, np.array(Id_plot(Vr_2,Alpha_2,beta_2,V_th_2)/Cur_2), "-", label = "$V_{back}$ = -0.5 V", markersize = 4, color='r', linewidth = 1.8)
#PLOT_CAP = plot(Vr_3, np.array(Id_plot(Vr_3,Alpha_3,beta_3,V_th_3)/Cur_3), "-", label = "$V_{back}$ = -1.0 V", markersize = 4, color='g', linewidth = 1.8)
#PLOT_CAP = plot(Vr_4, np.array(Id_plot(Vr_4,Alpha_4,beta_4,V_th_4)/Cur_4), "-", label = "$V_{back}$ = -2.0 V", markersize = 4, color='g', linewidth = 1.8)

#PLOT_CAP = plot(Vr_5, np.array(Id_plot(Vr_5,Alpha_5,beta_5,V_th_5)/Cur_5), "-", label = "$V_{back}$ = -3.0 V", markersize = 4, color='r', linewidth = 1.8)
#PLOT_CAP = plot(Vr_6, np.array(Id_plot(Vr_6,Alpha_6,beta_6,V_th_6)/Cur_6), "-", label = "$V_{back}$ = -5.0 V", markersize = 4, color='g', linewidth = 1.8)
#PLOT_CAP = plot(Vr_7, np.array(Id_plot(Vr_7,Alpha_7,beta_7,V_th_7)/Cur_7), "-", label = "$V_{back}$ = -10.0 V", markersize = 4, color='g', linewidth = 1.8)

#title(TITLE, fontsize = 18, fontproperties = fp)
#xlabel("$V_{gate}$ [$V$]", fontsize = 12, fontproperties = fp, fontweight='bold')
#ylabel('fit/data', fontsize = 12, fontproperties = fp, fontweight='bold')
#legend(loc=2, frameon=True, prop = FontProperties(size=7, family = FONT, fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"), numpoints=1)
#grid(True)
#yscale("log")
# Setup the range of two axis
#xticks(fontsize=16)
#yticks(fontsize=16)
#plt.subplots_adjust(hspace=0.2, bottom=0.125)
#pyplot.locator_params(axis = 'y', nbins = 5)
#xlim(-7,20.0)
#ylim(0.98, 1.03)


#figure(3)
#PLOT_IV = plot(sqrt(2.0*f_f(Nd) + V_back), V_th, "o", label = "data",  markersize = 5, color='b', linewidth = 2.0)
#PLOT_IV = plot(sqrt(2.0*f_f(Nd) + V_back), Vt_func(sqrt(2.0*f_f(Nd) + V_back), Vt_func_s, Vt_func_i), "-", label = "fit",  markersize = 5, color='r', linewidth = 2.0)

#title(TITLE, fontsize = 18, fontproperties = fp)
#legend(loc=2, frameon=True, prop = FontProperties(size=14, family = FONT, fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"), numpoints=1)
#xlabel("$\sqrt{(2 \phi_{f} +V_{back}}$ [V]", fontsize = 18, fontproperties = fp, fontweight='bold')
#ylabel("V$_{th}$ [V]", fontsize = 18, fontproperties = fp, fontweight='bold')
#grid(True)
#ylim(-3.0, 0)
#yscale("log")
# Setup the range of two axis
#xticks(fontsize=16)
#yticks(fontsize=16)

#figure(4)
#PLOT_IV = plot(V_back, m0, "o-", label = "",  markersize = 5, color='b', linewidth = 2.0)

#title(TITLE, fontsize = 18, fontproperties = fp)
#legend(loc=3, frameon=True, prop = FontProperties(size=14, family = FONT, fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"), numpoints=1)
#xlabel("V$_{vback}$ [V]", fontsize = 18, fontproperties = fp, fontweight='bold')
#ylabel("$\mu_{0}$ [$cm^2$/V*sec]", fontsize = 18, fontproperties = fp, fontweight='bold')
#grid(True)
#ylim(0, 1450)
#yscale("log")
# Setup the range of two axis
#xticks(fontsize=16)
#yticks(fontsize=16)

#figure(5)
#PLOT_IV = plot(V_back, V_half, "o-", label = "",  markersize = 5, color='b', linewidth = 2.0)
#title(TITLE, fontsize = 18, fontproperties = fp)
#legend(loc=2, frameon=True, prop = FontProperties(size=14, family = FONT, fname="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"), numpoints=1)
#xlabel("V$_{back}$ [V]", fontsize = 18, fontproperties = fp, fontweight='bold')
#ylabel("V$_{1/2}$ [V]", fontsize = 18, fontproperties = fp, fontweight='bold')
#grid(True)
#ylim(0, 50)
#yscale("log")
# Setup the range of two axis
#xticks(fontsize=16)
#yticks(fontsize=16)


show()





