import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import re
from scipy.interpolate import CubicSpline
from scipy.special import erf
from scipy.optimize import curve_fit
import math
#import sympy as sym

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--isHamburg',             action='store',      default=False,   type=bool,      help="Is it Hamburg file? By default Vienna")
argParser.add_argument('--Vienna_Set2',           action='store',      default=False,   type=bool,      help="Is it Set 2 of Vienna measurements? By default Vienna Set 1 (old)")
argParser.add_argument('--Vienna_PSS_Set2',           action='store',      default=False,   type=bool,      help="Is it Set 2 of Vienna measurements? By default Vienna Set 1 (old)")
argParser.add_argument('--Vienna_HGC_OTLL_Set2',           action='store',      default=False,   type=bool,      help="Is it Set 2 of Vienna measurements? By default Vienna Set 1 (old)")
argParser.add_argument('--Vienna_HGC_OTLL_Set2_100mV',           action='store',      default=False,   type=bool,      help="Is it Set 2 of Vienna measurements? By default Vienna Set 1 (old)")
argParser.add_argument('--useELM',             action='store',      default=False,   type=bool,      help="Use Electrometer current?")

args = argParser.parse_args()

#some specifications for plotting

if args.isHamburg:
    #Hamburg ranges
    small_vrange = 2.5
    large_vrange = 30
    left_vrange = -0.5
else:
    #Vienna ranges
    if args.Vienna_Set2 or args.Vienna_PSS_Set2 or args.Vienna_HGC_OTLL_Set2 or args.Vienna_HGC_OTLL_Set2_100mV:
        small_vrange = 30.
        left_vrange = -0.5
    else:
        small_vrange = 30
        left_vrange = 0.
    large_vrange = 300

#end od spec.

def mufunc(x, mu0, vp5):
    return mu0 * 1./(1. + x/vp5)

def mufunc_abs(x, mu0, vp5):
    return mu0 * x * 1./(1. + x/vp5)

def mufunc_abs_VG(x, mu0, vp5, vg):
    return mu0 * (x-vg) * 1./(1. + (x-vg)/vp5)

def polfunc2(x, par0, par1, par2):
    return (par0 + par1*x + par2*x*x)

def gaus(x, mu, sig, A):
    return (A*np.exp(-((x-mu)*(x-mu))*1./(2*(sig*sig))))

def gausmod(x, mu, sig0, sig1, A, B):
    #return (A*math.exp(-math.pow((x-mu),2)*1./(2*(sig0*sig0 + sig1*sig1*math.pow((x-mu),2)))) + B)
    return (A*np.exp(-((x-mu)*(x-mu))*1./(2*(sig0*sig0 + sig1*sig1*(x-mu)*(x-mu)))) + B)

def erfunc(x, a, b, z, f):
    return a * erf((x - z)*f) + b

def expofunc(x, a, b, c):
    return (a * np.exp(b*x +c))

def expo2func(x, a, b, c, d, e):
    return a+np.exp(b*x+c)+np.exp(d*x+e)

def logfunc(x, a, b, c):
    return (a + c*np.log(b*x))

def log2func(x, a, b, c, d, e):
    return (a + c*np.log(b*x) + d*np.log(e*x))

def pol4(x,a,b,c):
    return (a+b*x+c*x*x)

def calculate_threshold(vs,isds):

   v_th = np.nan
   a = b = spl_dev = -1

   for i in range(1,len(vs)):
   # get spline fit, requires strictlty increasing array
        isds[i] = isds[i] - isds[0]  # we are havng offset problems
   isds[0] = 0
   y_norm = []
   for i in range(len(vs)):
    y_norm.append(isds[i]/ np.max(np.abs(isds)))
   x_norm = np.arange(len(y_norm))
   spl = CubicSpline(x_norm, y_norm)
   spl_dev = spl(x_norm, 1)

   # get tangent at max. of 1st derivative
   maximum = np.argmax(spl_dev)
   i_0 = isds[maximum]
   v_0 = vs[maximum]
   a = (isds[maximum] - isds[maximum - 1]) / (vs[maximum] - vs[maximum - 1])
   b = i_0 - a*v_0
   v_th = -b/a
   v_th = round(v_th,2)

   return v_th


def plot_biasvsthreshold(bias_vol_nums,bd_voltages,plotdir):
   
    new_bias = []
    new_bd = []
    #new_bias = bias_vol_floats
    for ivl in range(len(bias_vol_floats)):
        if bias_vol_floats[ivl]<1000 and bias_vol_floats[ivl]>=1:
            new_bias.append(bias_vol_floats[ivl])
            new_bd.append(bd_voltages[ivl])
    #new_bias.remove(0)

    err = []
    for vl in bd_voltages:
        err.append(1.*vl)

    #poptx, pcovx = curve_fit(log2func, np.array(new_bias), np.array(new_bd), maxfev=5000, p0=[9,40,2,10,1])#, sigma=err)
    #bd_exp = log2func(np.array(new_bias),*poptx)

    #poptx, pcovx = curve_fit(expo, np.array(new_bias), np.array(new_bd), maxfev=5000)
    #bd_exp = expo(np.array(new_bias),*poptx)

    plt.figure("Fig_VbvsVth_lin",figsize=(14.0, 10.5))
    fig, axs = plt.subplots(1, 1)
    axs.plot(bias_vol_nums,bd_voltages, '-o')
    axs.set_xlabel('V$_{bias}$ [V]', size=0.75*fontsize)
    axs.set_ylabel('V$_{threshold}$ [V]', size=0.75*fontsize)
    axs.set_ylim(bottom=-2)
    axs.grid()
    fig.tight_layout()
    plt.draw()
    savefile = plotdir+"/thresholdvsbias_lin.png"
    plt.savefig(savefile,format='png')
    plt.close()

    plt.figure("Fig_VbvsVth_log",figsize=(14.0, 10.5))
    fig, axs = plt.subplots(1, 1)
    bias_vol_nums_shifted = [bv+0.1 for bv in bias_vol_nums]
    axs.plot(bias_vol_nums_shifted,bd_voltages, '-o')
    axs.set_xlabel('V$_{bias}$+0.1 [V]', size=0.75*fontsize)
    axs.set_ylabel('V$_{threshold}$ [V]', size=0.75*fontsize)
    axs.set_xscale('log')
    axs.set_ylim(bottom=-2)
    axs.grid()
    fig.tight_layout()
    plt.draw()
    savefile = plotdir+"/thresholdvsbias_log.png"
    plt.savefig(savefile,format='png')
    plt.close()

def plot_params(counter,vols_all,curs_all,V_offset,par_product,bias_voltages,colors,mu0_vals,vp5_vals,vth_off):

    plt.figure("Fig2",figsize=(14.0, 10.5))

    for icol in range(counter):
    
        vols_sub2, curs_sub2, mus_2 = [], [], []

        for ix in range(len(vols_all[icol])):
            if(vols_all[icol][ix]>V_offset):
                vols_sub2.append(vols_all[icol][ix])
                curs_sub2.append(curs_all[icol][ix])
                mus_2.append(curs_all[icol][ix]*1./(vols_all[icol][ix]))

        for ix in range(len(curs_sub2)):
            curs_sub2[ix] = curs_sub2[ix]*1./par_product
            mus_2[ix] = mus_2[ix]*1./par_product

        plt.title(title, size=fontsize)

        #popt, pcov = curve_fit(mufunc, vols_sub2, mus_2, p0=[0.000003,6], maxfev=5000)
        #popt, pcov = curve_fit(mufunc_abs, vols_sub2, curs_sub2, p0=[0.000003*1./par_product,6], maxfev=5000)
        if args.Vienna_Set2 or args.Vienna_PSS_Set2 or args.Vienna_HGC_OTLL_Set2 or args.Vienna_HGC_OTLL_Set2_100mV:
	    popt, pcov = curve_fit(mufunc_abs_VG, vols_sub2, curs_sub2, p0=[0.00003*1./par_product,6,0], maxfev=50000)
	else:
	    popt, pcov = curve_fit(mufunc_abs_VG, vols_sub2, curs_sub2, p0=[0.000003*1./par_product,6,0], maxfev=5000)
        #print popt
        mu0_vals.append(popt[0])
        vp5_vals.append(popt[1])
        vth_off.append(popt[2])
        #print popt[2]

        stdev = 1.
        exp = mufunc_abs_VG(vols_sub2, *popt)
        r = mus_2 - exp
        chisq = np.sum((r/stdev)**2)
        df = len(vols_sub2)-2

        plt.plot(vols_sub2,exp,'-o', label=bias_voltages[icol], color=colors[icol%29])

        plt.xlabel('V$_{SG}$ - V$_{th}$ [V]', size=fontsize)
        plt.ylabel('${\mu}_e$ [cm$^{2}$ / V s]', size=fontsize)

        #print "chi2/df: ",chisq,"/",df

        del vols_sub2, curs_sub2, mus_2


    plt.legend(loc=1,ncol=2,fontsize=0.85*fontsize)
    plt.grid()
    plt.xlim([0.1, 20])
    plt.draw()
    savefile = plotdir+"/mue_vs_vth_subtracted.png"
    plt.savefig(savefile,format='png')
    plt.close()

def plot_param_values(bias_vol_nums,mu0_vals,vp5_vals,plotdir,mu0_max,v1by2_min,v1by2_max):

    for ivar in range(2):

        plt.figure("Fig_mu",figsize=(14.0, 10.5))
        fig, axs = plt.subplots(1, 1)
        axs.plot(bias_vol_nums,mu0_vals, '-o')
        axs.set_xlabel('V$_{bias}$ [V]', size=0.75*fontsize)
        axs.set_ylabel('${\mu}_0$', size=0.75*fontsize)
        axs.set_ylim(bottom=0,top=mu0_max)
        if ivar==0:
            axs.set_xlim(left=left_vrange,right=small_vrange)
        else:
            axs.set_xlim(left=left_vrange,right=large_vrange)
        axs.grid()
        fig.tight_layout()
        plt.draw()
        savefile = plotdir+"/paramsvsbias_mu_"+str(ivar+1)+".png"
        plt.savefig(savefile,format='png')
        plt.close()

        plt.figure("Fig_v1by2",figsize=(14.0, 10.5))
        fig, axs = plt.subplots(1, 1)
        axs.plot(bias_vol_nums,vp5_vals, '-o')
        axs.set_xlabel('V$_{bias}$ [V]', size=0.75*fontsize)
        axs.set_ylabel('V$_{1/2}$ [V]', size=0.75*fontsize)
        axs.set_ylim(bottom=v1by2_min,top=v1by2_max)
        if ivar==0:
            axs.set_xlim(left=left_vrange,right=small_vrange)
        else:
            axs.set_xlim(left=left_vrange,right=large_vrange)
        axs.grid()
        fig.tight_layout()
        plt.draw()
        savefile = plotdir+"/paramsvsbias_v1by2_"+str(ivar+1)+".png"
        plt.savefig(savefile,format='png')
        plt.close()

def plot_thresholdvsbias(bd_voltages,bias_vol_floats,csv,plotdir):

    new_bd, new_bias = [], []

    for iv in range(len(bias_vol_floats)):
        if bias_vol_floats[iv]<=20:
            new_bias.append(bias_vol_floats[iv])
            new_bd.append(bd_voltages[iv])
     
    poptx, pcovx = curve_fit(expofunc, np.array(new_bd),np.array(new_bias), maxfev=50000,p0=[0.5,0.5,-5])
    bias_vol_floats_exp = expofunc(np.array(new_bd),*poptx)

    print "thresholdvsbias"
    for par in poptx:
        print par

    plt.figure(figsize=(24.0, 16))
    figc, axcs = plt.subplots(2, 1)
    axcs[0].plot(bd_voltages,bias_vol_floats, '-o')
    axcs[0].plot(bd_voltages,csv(bd_voltages), 'k', color='green')
    axcs[0].plot(new_bd,bias_vol_floats_exp, 'k', color='red')
    #axcs[0].plot(bd_voltages,csv(bd_voltages,1), 'k', color='yellow')
    #axcs[0].plot(bd_voltages,csv(bd_voltages,2), 'k', color='magenta')
    axcs[0].set_ylabel('V$_{bias}$ [V]', size=0.75*fontsize)
    axcs[0].set_xlabel('V$_{threshold}$ [V]', size=0.75*fontsize)
    axcs[0].grid()
    axcs[0].set_ylim(top=30)
    axcs[1].plot(bd_voltages,bias_vol_floats, '-o')
    axcs[1].plot(bd_voltages,csv(bd_voltages), 'k', color='green')
    axcs[1].plot(new_bd,bias_vol_floats_exp, 'k', color='red')
    #axcs[1].plot(bd_voltages,csv(bd_voltages,1), 'k', color='yellow')
    #axcs[1].plot(bd_voltages,csv(bd_voltages,2), 'k', color='magenta')
    axcs[1].set_ylabel('V$_{bias}$ [V]', size=0.75*fontsize)
    axcs[1].set_xlabel('V$_{threshold}$ [V]', size=0.75*fontsize)
    axcs[1].grid()
    axcs[1].set_ylim(top=large_vrange)
    figc.tight_layout()
    plt.draw()
    savefile = plotdir+"/threshold_fit.png"
    plt.savefig(savefile,format='png')
    plt.close()

def plot_vbdvsx(bd_voltages,x_vals,plotdir,x_vals_exp):

    plt.figure(figsize=(24.0, 16))
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(bd_voltages,x_vals, '-o')
    axs[0].plot(bd_voltages,x_vals_exp, 'k', color='red')
    axs[0].set_xlabel('V$_{threshold}$ [V]', size=0.75*fontsize)
    axs[0].set_ylabel('x [um]', size=0.75*fontsize)
    axs[0].grid()
    axs[0].set_ylim(top=10)
    axs[1].plot(bd_voltages,x_vals, '-o')
    axs[1].plot(bd_voltages,x_vals_exp, 'k', color='red')
    axs[1].set_xlabel('V$_{threshold}$ [V]', size=0.75*fontsize)
    axs[1].set_ylabel('x [um]', size=0.75*fontsize)
    axs[1].grid()
    fig.tight_layout()
    plt.draw()
    savefile = plotdir+"/threshold_vs_x.png"
    plt.savefig(savefile,format='png')
    plt.close()

def plot_doping(bias_vol_nums,x_vals,px_vals,plotdir,bias_vol_floats):

    #poptx, pcovx = curve_fit(erfunc, bias_vol_floats, x_vals, maxfev=5000)
    #x_vals_exp = erfunc(bias_vol_floats, *poptx)

    for ivar in range(2):

        # plot w/x vs V_bias

        plt.figure("Fig_xvsv",figsize=(24.0, 16))
        fig, axs = plt.subplots(1, 1)
        axs.plot(bias_vol_nums, x_vals, 'o')
        #axs.plot(bias_vol_nums, x_vals_exp, 'k', color='red')
        axs.set_xlabel('V$_{bias}$ [V]', size=0.75*fontsize)
        axs.set_ylabel('w (= x) [um]',size=0.75*fontsize)
        if ivar==0:
            axs.set_xlim(left=left_vrange,right=small_vrange)
            axs.set_ylim(top=200,bottom=0)
        else:
            axs.set_xlim(left=left_vrange,right=large_vrange)
            if args.isHamburg:
                axs.set_ylim(top=120,bottom=0)
                yticks=np.arange(0,120,20)
            else:
                axs.set_ylim(top=300,bottom=0)
                yticks=np.arange(0,300,30)
            axs.set_yticks(yticks)
        axs.grid()
        
        fig.tight_layout()
        plt.draw()
        savefile = plotdir+"/profile_"+str(ivar+1)+"_xvsV.png"
        plt.savefig(savefile,format='png')
        plt.close()

        # plot p vs V_bias

        plt.figure("Fig_pvsv",figsize=(24.0, 16))
        fig, axs = plt.subplots(1, 1)
        axs.plot(bias_vol_nums,px_vals, 'o')
        axs.set_xlabel('V$_{bias}$ [V]', size=0.75*fontsize)
        axs.set_ylabel('p(x) [cm$^{-3}$]', size=0.75*fontsize)
        axs.set_ylim(bottom=1.e+11,top=1.e+16)
        #yticks=np.arange(1.e+12, 1.e+15, 1.e+13)
        #axs[1].set_yticks(yticks * 10, ['%d'%val for val in yticks])
        if ivar==0:
            axs.set_xlim(left=left_vrange,right=small_vrange)
        else:
            axs.set_xlim(left=left_vrange,right=large_vrange)
        axs.grid()
        axs.set_yscale('log')

        fig.tight_layout()
        plt.draw()
        savefile = plotdir+"/profile_"+str(ivar+1)+"_pvsV.png"
        plt.savefig(savefile,format='png')
        plt.close()
        
        # p(x) vs x

        plt.figure("Fig_pvsx",figsize=(24.0, 16))
        fig, axs = plt.subplots(1, 1)
        
        axs.plot(x_vals,px_vals, 'o')
        axs.set_ylim(bottom=1.e+11,top=1.e+16)

        #fit p(x) vs x:

        # Gauusian fit
        #popt_x, pcov_x = curve_fit(gaus, x_vals, px_vals, p0=[1,0.1,1.e+15], maxfev=5000)
        #px_val_exp = gaus(x_vals, *popt_x)
        # Modified Gaussian fit
        #popt_x, pcov_x = curve_fit(gausmod, x_vals, px_vals, p0=[2,30,0.1,1.e+16,-1.e+14], maxfev=5000)
        #px_val_exp = gausmod(x_vals, *popt_x)

        #axs.plot(x_vals,px_val_exp, 'k')

        #print "Bulk doping",popt_x[3]*np.exp(-0.5/(popt_x[2]*popt_x[2]))+popt_x[4]

        axs.set_xlabel('x [um]', size=0.75*fontsize)
        axs.set_ylabel('p(x) [cm$^{-3}$]', size=0.75*fontsize)
        if ivar==0:
            if args.isHamburg:
                axs.set_xlim(left=0.,right=2.5)#10)
            else:
                axs.set_xlim(left=0.,right=10)
        else:
            if args.isHamburg:
                axs.set_xlim(left=0.,right=120)
            else:
                axs.set_xlim(left=0.,right=300)
        axs.grid()
        axs.set_yscale('log')
        
        fig.tight_layout()
        plt.draw()
        savefile = plotdir+"/profile_"+str(ivar+1)+"_pvsx.png"
        plt.savefig(savefile,format='png')
        plt.close()


folder = '/home/suman/CMS/HGCal/MOSEFT/Data/HGCal_OTLL_waiting_time/'
plotdir = 'HGCal_OTLL_waiting_time'
title = 'MOSFET (HGcal OTLL round)'
file_names = [
'test_bias_minus_0p0V_WT_5p0sec-2023-07-18T23-32-53.txt',
'test_bias_minus_0p0V_WT_2p0sec-2023-07-19T10-56-08.txt',
'test_bias_minus_0p0V_WT_1p0sec-2023-07-18T18-39-40.txt',
'test_bias_minus_0p0V_WT_0p5sec-2023-07-19T11-20-31.txt'

#'test_bias_minus_5p0V_WT_5p0sec-2023-07-18T22-45-45.txt',
#'test_bias_minus_5p0V_WT_2p0sec-2023-07-18T21-28-09.txt',
#'test_bias_minus_5p0V_WT_1p0sec-2023-07-17T17-19-46.txt',
#'test_bias_minus_5p0V_WT_0p5sec-2023-07-18T20-56-14.txt',

#'test_bias_minus_10p0V_WT_5p0sec-2023-07-19T12-39-47.txt',
#'test_bias_minus_10p0V_WT_2p0sec-2023-07-19T12-18-47.txt',
#'test_bias_minus_10p0V_WT_1p0sec-2023-07-17T20-29-34.txt',
#'test_bias_minus_10p0V_WT_0p5sec-2023-07-19T12-08-08.txt'
]
wts = [5.0,2.0,1.0,0.5]
VGS_max=35
VGS_min=0

files = [folder + fl for fl in file_names]

plotdir = 'Figures/'+plotdir
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

sep = '/'

bias_voltages = []
bd_voltages = []

#dielectric const
Si_DC = 11.67   
SiO2_DC = 3.9
#vacuum permittivity
epsilon_0 = 8.854 *1.e-12 *1.e-2 # last 1.e-2 to change from F/m to F/cm 

#define numerical values of parameters

if args.isHamburg:
    #Hamburg numbers
    WbyL = 4.964
    C_ox = 4.933e-09
    V_ds = 50.*1.e-3
    #tuneable parameters used for fit
    V_offset = 1.5
else:
    ##Vienna numbers
    rD = 650  #diameter-drain
    rS = 110  #diameter-source
    rG = 360  #diameter-gate
    WbyL = 2*math.pi*1./math.log(rD/rS)
    rD = rD*0.5*1.e-6
    rS = rS*0.5*1.e-6
    #C_ox = 82*(1.e-12)*1./(math.pi*math.pow(rD*100,2))  #capacitance / area in cm2 100 to go from m to cm #5*1.e-9
    d_ox = 720*1.e-9*1.e+2  #thickness in nm, 1.e+2 to get in cm
    C_ox =  SiO2_DC*epsilon_0*1./d_ox
    #C_ox = 4.8e-09
    V_ds = 100.*1.e-3
    #tuneable parameters used for fit
    V_offset = 2.5

print "C_ox ",C_ox," WbyL ",WbyL
epsilon = Si_DC * epsilon_0
q0 = 1.6e-19

colors = ['aqua','grey','sienna','black','blue','brown','chartreuse','coral','crimson','pink',
        'darkgreen','navy','gold','green','orange','indigo','khaki','darkgoldenrod','magenta','olive',
        'cadetblue','darkorchid','papayawhip','tan','maroon','lightcoral','firebrick','salmon','red' 
        ]

fontsize = 20

plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['axes.titlesize'] = fontsize

fig, ax = plt.subplots()

plt.figure("Fig1",figsize=(14.0, 10.5))

#fig, ax = plt.subplots()

sequence = 48
counter=0

curs_all = []
vols_all = []
mus_all = []

bias_vol_floats = []

index = 0

for f in files:
   print(f)

   # plotting current and voltage #

   #read data
   strs = f.split('/')
   #bias_voltage = ((strs[len(strs)-1][16:22]).split('-')[0]).split('_')[0]
   #print "bias_voltage",bias_voltage
   #temp = re.compile("([0-9]+)([a-zA-Z]+)([0-9]+)([a-zA-Z]+)")
   #res = temp.match(bias_voltage).groups()
   #bias_voltage = res[0]+"."+res[2]
   #bv_float = int(res[0])+0.1*int(res[2])
   #if res[3].startswith("N"):
   #    bv_float *= -1.0
   #    bias_voltage = "-"+res[0]+"."+res[2]
   #bv_float = int(res[0])+0.1*int(res[2])
   #bias_vol_floats.append(bv_float)

   if args.isHamburg:
    #Hamburg convention
    data = np.genfromtxt(f, skip_header=37 , skip_footer=1, delimiter = '\t', dtype=None)#, max_rows=1000)
    df = pd.DataFrame(data, columns = ['bias', 'T', 'Current', 'Guard Current'])
    vgs_index=0
    ids_index=2
    # put data in plot
    #plt.plot(df['voltage'], df['i_smu2'], '-o', label=''.format(sequence), color=colors[counter])
    plt.plot(df['bias'], df['Current'], '-o', label=bias_voltage, color=colors[counter%29])
    #print df['bias'], df['Current']
   else:
    ##Vienna convention
    data = np.genfromtxt(f, skip_header=10 , delimiter = '\t', dtype=None, max_rows=1000)
    df = pd.DataFrame(data, columns = ['timestamp', 'voltage', 'i_smu', 'i_smu2', 'i_elm', 'temperature'])
    vgs_index=1
    #ids_index=3
    # put data in plot
    if args.useELM:
        ids_index=4
        plt.plot(df['voltage'], df['i_elm'], '-o', label=wts[counter], color=colors[counter%29])
    else:
        ids_index=3
        plt.plot(df['voltage'], df['i_smu2'], '-o', label=wts[counter], color=colors[counter%29])
   
   plt.title(title, size=fontsize)
   plt.xlabel('V$_{SG}$ [V]', size=fontsize)
   plt.ylabel('I$_{SD}$ [A]', size=fontsize)
   #for label
   #bias_voltages.append(bias_voltage)
   # counter for color
   counter += 1

   # Get threshold voltage here #
   vals = df.values.tolist()
   vols, curs, curs_store = [], [], []
   for iv in range(len(vals)):
       vols.append(vals[iv][vgs_index])
       curs.append(vals[iv][ids_index])
       curs_store.append(vals[iv][ids_index])
   #print vols,curs
   v_th = calculate_threshold(vols,curs)
   bd_voltages.append(v_th)
   print v_th

   curs_all.append(curs_store)

#ax.xaxis.set_tick_params(labelsize=2*fontsize)
#ax.yaxis.set_tick_params(labelsize=20*fontsize)

plt.legend(loc=2,ncol=2,fontsize=0.85*fontsize)
plt.grid()
plt.xlim([VGS_min, VGS_max])
#plt.ylim(bottom=-1.e-12,top=1.e-9)

plt.draw()
#plt.show()
savefile = plotdir+"/ISDvsVSG.png"
plt.savefig(savefile,format='png')
plt.close()

# plot threshold voltage as a function of bias voltage

#plot_biasvsthreshold(bias_vol_floats,bd_voltages,plotdir)
