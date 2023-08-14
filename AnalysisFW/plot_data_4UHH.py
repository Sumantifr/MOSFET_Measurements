import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import re
from scipy.interpolate import CubicSpline
from scipy.special import erf
from scipy.optimize import curve_fit
import statsmodels.api as sm
import scipy.fftpack as fftpack
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.signal import savgol_filter
import math
#import sympy as sym

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--isHamburg',             action='store',      default=False,   type=bool,      help="Is it Hamburg file? By default Vienna")
argParser.add_argument('--Vienna_Set2',           action='store',      default=False,   type=bool,      help="Is it Set 2 of Vienna measurements? By default Vienna Set 1 (old)")
argParser.add_argument('--Vienna_PSS_Set2',           action='store',      default=False,   type=bool,      help="Is it Set 2 of Vienna measurements? By default Vienna Set 1 (old)")
argParser.add_argument('--Vienna_HGC_OTLL_Set2',           action='store',      default=False,   type=bool,      help="Is it Set 2 of Vienna measurements? By default Vienna Set 1 (old)")
argParser.add_argument('--Vienna_HGC_OTLL_Set2_100mV',           action='store',      default=False,   type=bool,      help="Is it Set 2 of Vienna measurements? By default Vienna Set 1 (old)")
argParser.add_argument('--Vienna_HGC_nopstop_Set2',           action='store',      default=False,   type=bool,      help="Is it Set 2 of Vienna measurements? By default Vienna Set 1 (old)")
argParser.add_argument('--useELM',             action='store',      default=False,   type=bool,      help="Use Electrometer current?")
argParser.add_argument('--isVienna',           action='store',      default=False,   type=bool,      help="Is Vienna measurements?")
argParser.add_argument('--nopstop',             action='store',      default=False,   type=bool,      help="Structure without p-stop/p-spray?")
argParser.add_argument('--HGC_Common',             action='store',      default=False,   type=bool,      help="HGC common structure?")
argParser.add_argument('--LargeSet',             action='store',      default=False,   type=bool,      help="Structure without p-stop/p-spray?")
argParser.add_argument('--verbose',             action='store',      default=False,   type=bool,      help="Print many numbers?")
argParser.add_argument('--usePolFit',            action='store',      default=False,   type=bool,      help="Polynomial fit (for x)?")
argParser.add_argument('--useSmoothing',            action='store',      default=False,   type=bool,      help="Use smoothing (for x)?")
argParser.add_argument('--repeated',            action='store',      default=False,   type=bool,      help="Repeated measurements?")
args = argParser.parse_args()

#some specifications for plotting

if args.isHamburg:
    #Hamburg ranges
    small_vrange = 1.75
    medium_vrange = 30
    large_vrange = 120
    left_vrange = -0.75
else:
    #Vienna ranges
    #if args.Vienna_Set2 or args.Vienna_PSS_Set2 or args.Vienna_HGC_OTLL_Set2 or args.Vienna_HGC_OTLL_Set2_100mV or args.Vienna_HGC_nopstop_Set2:
    #    small_vrange = 1.75 #30.
    #    left_vrange = -0.75 #-0.5
    #else:
    #    small_vrange = 30
    #    left_vrange = 0.
    small_vrange = 1.75
    medium_vrange = 30
    large_vrange = 200
    left_vrange = -0.75

#end od spec.

def smoothTriangle(data, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]
    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

def smoothData(data, degree):
    box = np.ones(degree)/degree
    smoothed = np.convolve(data, box, mode='same')
    return smoothed

def smooth_data_convolve_my_average(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")

    # The "my_average" part: shrinks the averaging window on the side that 
    # reaches beyond the data, keeps the other side the same size as given 
    # by "span"
    re[0] = np.average(arr[:span])
    for i in range(1, span + 1):
        re[i] = np.average(arr[:i + span])
        re[-i] = np.average(arr[-i - span:])
    return re

def smooth_data_np_average(arr, span):  # my original, naive approach
    return [np.average(arr[val - span:val + span + 1]) for val in range(len(arr))]

def smooth_data_np_convolve(arr, span):
    return np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")

def smooth_data_np_cumsum_my_average(arr, span):
    cumsum_vec = np.cumsum(arr)
    moving_average = (cumsum_vec[2 * span:] - cumsum_vec[:-2 * span]) / (2 * span)

    # The "my_average" part again. Slightly different to before, because the
    # moving average from cumsum is shorter than the input and needs to be padded
    front, back = [np.average(arr[:span])], []
    for i in range(1, span):
        front.append(np.average(arr[:i + span]))
        back.insert(0, np.average(arr[-i - span:]))
    back.insert(0, np.average(arr[-2 * span:]))
    return np.concatenate((front, moving_average, back))

def smooth_data_kernel_regression(arr, span):
    # "span" smoothing parameter is ignored. If you know how to 
    # incorporate that with kernel regression, please comment below.
    kr = KernelReg(arr, np.linspace(0, 1, len(arr)), 'c')
    return kr.fit()[0]

def smooth_data_savgol_0(arr, span):  
    return savgol_filter(arr, span * 2 + 1, 0)

def smooth_data_savgol_1(arr, span):  
    return savgol_filter(arr, span * 2 + 1, 1)

def smooth_data_savgol_2(arr, span):  
    return savgol_filter(arr, span * 2 + 1, 2)

def smooth_data_fft(arr, span):  # the scaling of "span" is open to suggestions
    w = fftpack.rfft(arr)
    spectrum = w ** 2
    cutoff_idx = spectrum < (spectrum.max() * (1 - np.exp(-span / 2000)))
    w[cutoff_idx] = 0
    return fftpack.irfft(w)

def mufunc(x, mu0, vp5, vg):
    return mu0 * 1./(1. + (x-vg)/vp5)

def mufunc_abs(x, mu0, vp5):
    return mu0 * x * 1./(1. + x/vp5)

def mufunc_abs_VG(x, mu0, vp5, vg):
    return mu0 * (x-vg) * 1./(1. + (x-vg)/vp5)

def mufunc_o2(x, mu0, vp5, vg, a2):
    return mu0 * 1./(1. + (x-vg)/vp5 + a2*(x-vg)*(x-vg))

def mufunc_abs_VG_o2(x, mu0, vp5, vg, a2):
    return mu0 * (x-vg) * 1./(1. + (x-vg)/vp5 + a2*(x-vg)*(x-vg))

def polfunc0(x, par0):
    return par0 + 0*x

def polfunc2(x, par0, par1, par2):
    return (par0 + par1*x + par2*x*x)

def polfunc4(x, par0, par1, par2, par3, par4):
    return (par0 + par1*x + par2*x*x + par3*x*x*x + par4*x*x*x*x)

def gaus(x, mu, sig, A):
    return (A*np.exp(-((x-mu)*(x-mu))*1./(2*(sig*sig))))

def gausmod(x, mu, sig0, sig1, A, B):
    #return (A*math.exp(-math.pow((x-mu),2)*1./(2*(sig0*sig0 + sig1*sig1*math.pow((x-mu),2)))) + B)
    return (A*np.exp(-((x-mu)*(x-mu))*1./(2*(sig0*sig0 + sig1*sig1*(x-mu)*(x-mu)))) + B)

def erfunc(x, a, b, z, f):
    return a * erf((x - z)*f) + b

def mod_erfunc(x, eps_max, mu_mod, sig_mod):
    Mod_err_func = eps_max*(1.0-erf((x-mu_mod)/sig_mod))
    return Mod_err_func

def expofunc(x, a, b):
    return (a * np.exp(b*x))

def expo2func(x, a, b, c, d, e):
    return a+np.exp(b*x+c)+np.exp(d*x+e)

def expoplusgausfunc(x, ea, eb, mu, sig, A):
    return (A*np.exp(-((x-mu)*(x-mu))*1./(2*(sig*sig))) + ea * np.exp(-eb*x))

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

#def plot_params(counter,vols_all,curs_all,vth_all,V_offset,par_product,bias_voltages,colors,mu0_vals,vp5_vals,vth_off,vth_err,bias_voltages_float):
def plot_params(counter,vols_all,curs_all,vth_all,V_offset,par_product,bias_voltages_float,colors,mu0_vals,vp5_vals,vth_off,vth_err):

    #plt.figure("Fig2",figsize=(14.0, 10.5))

    if args.isHamburg:
        cur_unc = 0.001
    else:
        cur_unc = 0.002

    mu_vals_sum = []
    Ed_vals_sum = []

    for icol in range(counter):
    
        vols_sub2, curs_sub2, mus_2 = [], [], []
        sigma_curr = []

        Voffset = V_offset
        if args.isHamburg and bias_voltages_float[icol]<0:
            Voffset = 2*V_offset

        for ix in range(len(vols_all[icol])):

            #if args.isVienna and args.HGC_Common:
            #    Voffset = V_offset + 0.005*(ix+1)
            
            if((vols_all[icol][ix])>Voffset):
                vols_sub2.append(vols_all[icol][ix])
                curs_sub2.append(curs_all[icol][ix])
                sigma_curr.append(cur_unc*curs_all[icol][ix])
                mus_2.append(curs_all[icol][ix]*1./(vols_all[icol][ix]))

        for ix in range(len(curs_sub2)):
            curs_sub2[ix] = curs_sub2[ix] #*1./par_product
            sigma_curr[ix] = sigma_curr[ix] #*1./par_product
            mus_2[ix] = mus_2[ix] #*1./par_product

        plt.figure("Fig2",figsize=(14.0, 10.5))
        plt.title(title, size=fontsize)

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

        if (args.Vienna_Set2 or args.Vienna_PSS_Set2 or args.isVienna):
            ##popt, pcov = curve_fit(mufunc_abs_VG_o2, vols_sub2, curs_sub2, p0=[7.e+3/par_product,6.,-0.1,-2.5e-3], sigma=sigma_curr, absolute_sigma=True, maxfev=50000)
            if args.HGC_Common:
                popt, pcov = curve_fit(mufunc_abs_VG_o2, vols_sub2, curs_sub2, p0=[ 5.e-06,10,-1.0,-5.0e-04], sigma=sigma_curr, absolute_sigma=True, maxfev=50000)
            else:
                popt, pcov = curve_fit(mufunc_abs_VG_o2, vols_sub2, curs_sub2, p0=[ 1.e-06,06,-0.1,-2.5e-03], sigma=sigma_curr, absolute_sigma=True, maxfev=50000) #works for HCal ATLL & nopstop
        else:
            popt, pcov = curve_fit(mufunc_abs_VG, vols_sub2, curs_sub2, p0=[1.5e-3,60.,0.], sigma=sigma_curr, absolute_sigma=True, maxfev=5000)
            ##popt, pcov = curve_fit(mufunc_abs_VG_o2, vols_sub2, curs_sub2, p0=[1.52e-06,60.,0.,-1.4e-04], sigma=sigma_curr, absolute_sigma=True, maxfev=5000)

        if args.verbose:
            print "Bias",bias_voltages_float[icol]
            print popt
            print "Error on Vth ",np.sqrt(np.diag(pcov)[2])

        mu0_vals.append(popt[0]*1./par_product)
        vp5_vals.append(popt[1])
        vth_off.append(popt[2])
        #vth_err.append(np.sqrt(np.diag(pcov)[2])) 
        vth_err[icol] =  np.sqrt(max(np.diag(pcov)[2],1.e-6))

        stdev = 1.
        if (args.Vienna_Set2 or args.Vienna_PSS_Set2 or args.isVienna):
            exp = mufunc_abs_VG_o2(vols_sub2, *popt)
        else:
            exp = mufunc_abs_VG(vols_sub2, *popt)
        r = curs_sub2 - exp
        chisq = np.sum((r/sigma_curr)**2)
        df = len(vols_sub2)-len(popt)


        #variables for mu vs electric field plots#
        mu_vals = []
        vols_sub2_by_d = []
        bias_labels = []

        dx = 720*1.e-9*1.e+2

        for vl in vols_sub2:
            vols_sub2_by_d.append(1.e-6*vl/dx)  #1.e-6 to go to MV/cm from V/cm
            #Ed_vals_sum.append(1.e-6*vl/dx)

        if args.isVienna:
            mu_vals = mufunc_o2(vols_sub2, mu0_vals[icol],vp5_vals[icol],vth_off[icol],popt[3])
        if args.isHamburg:
            mu_vals = mufunc(vols_sub2, mu0_vals[icol],vp5_vals[icol],vth_off[icol])

        #for muval in mu_vals:
        #    mu_vals_sum.append(muval)

        Ed_vals_sum.append(vols_sub2_by_d)
        mu_vals_sum.append(mu_vals)
        bias_labels.append(str(bias_voltages_float[icol]))

        # end of mu vs E variables 

        ratio_expvsobs = []
        for iel in range(len(exp)):
            ratio_expvsobs.append(curs_sub2[iel]/exp[iel])

        #plt.plot(vols_sub2,curs_sub2,'-k', label=bias_voltages[icol], color=colors[icol%29])
        #plt.plot(vols_sub2,exp,'-o', label=bias_voltages[icol], color=colors[icol%29])

        axs[0].plot(vols_sub2,curs_sub2,'o',vols_sub2,exp,'-')
        axs[1].plot(vols_sub2,ratio_expvsobs,'o')
        #plt.legend({'data','curve-fit'})

        plt.xlabel('V$_{SG}$ - V$_{th}$ [V]', size=0.8*fontsize)
        #axs[0].set_ylabel('${\mu}_e$ [cm$^{2}$ / V s]', size=0.75*fontsize)
        axs[0].set_ylabel('I$_{SD}$ [A]', size=0.75*fontsize)
        axs[1].set_ylabel('Obs/Exp', size=0.75*fontsize)
       
        axs[0].set_ylim(bottom=0.0)
        axs[1].set_ylim(bottom=0.99,top=1.01)

        axs[0].set_xlim(left=0,right=20)#vols_sub2[0],right=vols_sub2[len(vols_sub2)-1])
        axs[1].set_xlim(left=0,right=20)#vols_sub2[0],right=vols_sub2[len(vols_sub2)-1])

        if args.verbose:
            print "chi2/df: ",chisq,"/",df

        del vols_sub2, curs_sub2, mus_2

        #plt.legend(loc=1,ncol=2,fontsize=0.85*fontsize)
        plt.grid()
        plt.xlim([0.1, 20])
        plt.draw()
        savefile = plotdir+"/Ids_vs_Vgs_vth_subtracted_bias"+str(bias_voltages_float[icol])+".png"
        plt.savefig(savefile,format='png')
        plt.close()

        #if args.isVienna:
        plt.figure("Fig_mu",figsize=(14.0, 10.5))
        plt.title(title + "    V$_{bias}$ = "+str(bias_voltages_float[icol])+" [V]", size=fontsize)
        plt.plot(vols_sub2_by_d,mu_vals,'-o')
        plt.ylabel('${\mu}_{e}$ [cm$^2$ /  (V s)]')
        plt.xlabel('(V$_{SG}$ - V$_{th}$) / d [MV/cm]')
        plt.ylim(0,2000)
        plt.draw()
        savefile = plotdir+"/mue_vs_Efield_bias"+str(bias_voltages_float[icol])+".png"
        plt.savefig(savefile,format='png')
        plt.close()

    # sum of mobiliy plots #

    plt.figure("Fig_mu_all",figsize=(14.0, 10.5))
    plt.title(title + "    all V$_{bias}$ values", size=fontsize)
    for iv in range(len(Ed_vals_sum)):
        if bias_voltages_float[iv]>=0.0 and bias_voltages_float[iv]<=5.0:
            plt.plot(Ed_vals_sum[iv],mu_vals_sum[iv],'-o',label=bias_voltages_float[iv],color=colors[iv%29])
    plt.ylabel('${\mu}_{e}$ [cm$^2$ /  (V s)]')
    plt.xlabel('(V$_{SG}$ - V$_{th}$) / d [MV/cm]')
    plt.ylim(0,2000)
    plt.legend(loc=1,ncol=2)
    plt.draw()
    savefile = plotdir+"/mue_vs_Efield_all_bias.png"
    plt.savefig(savefile,format='png')
    plt.close()


def plot_param_values(bias_vol_nums,mu0_vals,vp5_vals,plotdir,mu0_max,mu0_min,v1by2_min,v1by2_max):

    for ivar in range(3):

        plt.figure("Fig_mu",figsize=(14.0, 10.5))
        fig, axs = plt.subplots(1, 1)
        axs.plot(bias_vol_nums,mu0_vals, '-o')
        axs.set_xlabel('V$_{bias}$ [V]', size=0.75*fontsize)
        axs.set_ylabel('${\mu}_0$ [cm$^2$ /  (V s)]', size=0.75*fontsize)
        axs.set_ylim(bottom=mu0_min,top=mu0_max)
        if ivar==0:
            axs.set_xlim(left=left_vrange,right=small_vrange)
        elif ivar==1:
            axs.set_xlim(left=left_vrange,right=medium_vrange)
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
        elif ivar==1:
            axs.set_xlim(left=left_vrange,right=medium_vrange)
        else:
            axs.set_xlim(left=left_vrange,right=large_vrange)
        axs.grid()
        fig.tight_layout()
        plt.draw()
        savefile = plotdir+"/paramsvsbias_v1by2_"+str(ivar+1)+".png"
        plt.savefig(savefile,format='png')
        plt.close()

def plot_thresholdvsbias(bd_voltages,bias_vol_floats,csv,plotdir,coeffs,do_fits):

    new_bd, new_bias = [], []

    for iv in range(len(bias_vol_floats)):
        if bias_vol_floats[iv]>=10:
            new_bias.append(math.log(bias_vol_floats[iv]+0.35))
            new_bd.append(bd_voltages[iv])
     
    ratio_expvsobs = []

    if args.verbose:
        print "thresholdvsbias"

    if do_fits:

        poptx, pcovx = curve_fit(polfunc4, np.array(new_bd),np.array(new_bias), maxfev=50000,p0=[100,50,10.,1,0.1])
        bias_vol_floats_exp = polfunc4(np.array(new_bd),*poptx)
    
        for iel in range(len(bias_vol_floats_exp)):
            ratio_expvsobs.append(new_bias[iel]/bias_vol_floats_exp[iel])
    
        for par in poptx:
            if args.verbose:
                print par
            coeffs.append(par)

    plt.figure(figsize=(24.0, 16))
    figc, axcs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
    axcs[0].plot(bd_voltages,bias_vol_floats, '-o')
    #axcs[0].plot(bd_voltages,csv(bd_voltages), 'k', color='green')
    if do_fits:
        axcs[0].plot(new_bd,bias_vol_floats_exp, 'k', color='red')
    #axcs[0].plot(bd_voltages,csv(bd_voltages,1), 'k', color='yellow')
    #axcs[0].plot(bd_voltages,csv(bd_voltages,2), 'k', color='magenta')
    axcs[0].set_ylabel('V$_{bias}$ [V]', size=0.75*fontsize)
    axcs[0].set_xlabel('V$_{threshold}$ [V]', size=0.75*fontsize)
    axcs[0].grid()
    #axcs[0].set_ylim(top=30)
    axcs[0].set_ylim(top=large_vrange)

    ##axcs[1].plot(bd_voltages,bias_vol_floats, '-o')
    ##axcs[1].plot(bd_voltages,csv(bd_voltages), 'k', color='green')
    ##axcs[1].plot(new_bd,bias_vol_floats_exp, 'k', color='red')
    #axcs[1].plot(bd_voltages,csv(bd_voltages,1), 'k', color='yellow')
    #axcs[1].plot(bd_voltages,csv(bd_voltages,2), 'k', color='magenta')
    #axcs[1].set_ylabel('V$_{bias}$ [V]', size=0.75*fontsize)
    #axcs[1].plot(bd_voltages,ratio_expvsobs, '-o')
    axcs[1].set_ylabel('Obs/Exp')
    axcs[1].set_xlabel('V$_{threshold}$ [V]', size=0.75*fontsize)
    axcs[1].grid()
    #axcs[1].set_ylim(top=large_vrange)
    axcs[1].set_ylim(top=2,bottom=0)
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

def plot_doping(bias_vol_nums,x_vals,px_vals,plotdir):

    x_vals_10 = []
    for ix in range(len(x_vals)):
        x_vals_10.append(10*x_vals[ix])

    for ivar in range(3):

        # plot w/x vs V_bias

        plt.figure("Fig_xvsv",figsize=(24.0, 16))
        fig, axs = plt.subplots(1, 1)
        if ivar==0 and not args.nopstop:
            axs.plot(bias_vol_nums, x_vals_10, 'o')
        else:
            axs.plot(bias_vol_nums, x_vals, 'o')
        #axs.plot(bias_vol_nums, x_vals_exp, 'k', color='red')
        axs.set_xlabel('V$_{bias}$ [V]', size=0.75*fontsize)
        if ivar==0 and not args.nopstop:
            axs.set_ylabel('w (= x) x 10 [um]',size=0.75*fontsize)
        else:
            axs.set_ylabel('w (= x) [um]',size=0.75*fontsize)
        if ivar==0:
            axs.set_xlim(left=left_vrange,right=small_vrange)
            axs.set_ylim(top=40,bottom=0)
        elif ivar==1:
            axs.set_xlim(left=left_vrange,right=medium_vrange)
            axs.set_ylim(top=120,bottom=0)
        else:
            axs.set_xlim(left=left_vrange,right=large_vrange)
            if args.isHamburg:
                axs.set_ylim(top=300,bottom=0)
                yticks=np.arange(0,300,30)
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
        if args.isHamburg:
            axs.set_ylim(bottom=1.e+12,top=1.e+16)
        else:
            axs.set_ylim(bottom=1.e+11,top=1.e+16)
        #yticks=np.arange(1.e+12, 1.e+15, 1.e+13)
        #axs[1].set_yticks(yticks * 10, ['%d'%val for val in yticks])
        if ivar==0:
            axs.set_xlim(left=left_vrange,right=small_vrange)
        elif ivar==1:
            axs.set_xlim(left=left_vrange,right=medium_vrange)
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
        if args.isHamburg:
            if ivar==0:
                axs.set_ylim(bottom=1.e+14,top=1.e+16)
            else:
                axs.set_ylim(bottom=1.e+12,top=1.e+16)
        else:
            if ivar==0:
                axs.set_ylim(bottom=1.e+14,top=1.e+16)
            else:
                axs.set_ylim(bottom=1.e+11,top=1.e+16)

        #fit p(x) vs x:

        # Gauusian fit
        #popt_x, pcov_x = curve_fit(gaus, x_vals, px_vals, p0=[1,0.1,1.e+15], maxfev=5000)
        #px_val_exp = gaus(x_vals, *popt_x)
        # Modified Gaussian fit
      
        xvals_fit, pxvals_fit = [], []
        for ix in range(len(x_vals)):
            if ivar==0:
                if(x_vals[ix]<0.1 or x_vals[ix]>2.5):
                    continue
            elif ivar==1:
                if(x_vals[ix]<0.1 or x_vals[ix]>50):
                    continue
            else:
                if(x_vals[ix]>250):
                    continue
            xvals_fit.append(x_vals[ix])
            pxvals_fit.append(px_vals[ix])

        px_errs = []
        for px in pxvals_fit:
            px_errs.append(0.01*px)

        if not args.nopstop:
            ''' 
            popt_x, pcov_x = curve_fit(gausmod, xvals_fit, pxvals_fit, p0=[0.5,1,1,4.e+15,-1.e+15], sigma=px_errs, maxfev=50000, bounds=((0.2,0,0,1.e+15,-1.e+16),(0.8,10,10,1.e+16,1.e+16)))
            px_val_exp = gausmod(xvals_fit, *popt_x)
            
            #popt_x, pcov_x = curve_fit(expoplusgausfunc, xvals_fit, pxvals_fit, p0=[8.e+15,1.,0.5,1,3.e+15], sigma=px_errs, bounds = [(1.e+14,-100,0.2,0,1.e+15), (1.e+16,10,0.8,10,1.e+16)], maxfev=50000)
            #px_val_exp = expoplusgausfunc(np.array(xvals_fit), *popt_x)

            axs.plot(xvals_fit,px_val_exp, 'k')
            print popt_x
            peak_bulk = [0.6,200]
            print "peak and bulk dopings",gausmod(peak_bulk, *popt_x)
            '''
            popt_x, pcov_x = curve_fit(mod_erfunc, xvals_fit, pxvals_fit, p0=[3.e+15,0.5,50], maxfev=50000, bounds=((1.e+15,0.1,-5000),(1.e+16,1,5000)))
            px_val_exp = mod_erfunc(xvals_fit, *popt_x)
            #axs.plot(xvals_fit,px_val_exp, 'k')
            print popt_x

        else:
            #popt_x, pcov_x = curve_fit(polfunc2, x_vals, px_vals, p0=[3.e+12,0,0], maxfev=50000)
            #px_val_exp = polfunc2(np.array(x_vals), *popt_x)
            popt_x, pcov_x = curve_fit(polfunc0, x_vals, px_vals, p0=[3.e+12], maxfev=50000)
            px_val_exp = []
            for ix in range(len(x_vals)):
                px_val_exp.append(px_vals[0])
            axs.plot(x_vals,px_val_exp, 'k')
            print "Bulk doping (const)",px_vals[0]
        #print "Bulk doping",popt_x[3]*np.exp(-0.5/(popt_x[2]*popt_x[2]))+popt_x[4]
        
        axs.set_xlabel('x [um]', size=0.75*fontsize)
        axs.set_ylabel('p(x) [cm$^{-3}$]', size=0.75*fontsize)
        if ivar==0:
            if args.isHamburg:
                axs.set_xlim(left=0.,right=2.5)
            else:
                axs.set_xlim(left=0.,right=2.5)#10)
        elif ivar==1:
            if args.isHamburg:
                axs.set_xlim(left=0.,right=50)
            else:
                axs.set_xlim(left=0.,right=50)
        else:
            if args.isHamburg:
                if args.nopstop and args.LargeSet:
                    axs.set_xlim(left=0.,right=250)
                else:
                    axs.set_xlim(left=0.,right=120)
            else:
                axs.set_xlim(left=0.,right=250)
        axs.grid()
        axs.set_yscale('log')
        
        fig.tight_layout()
        plt.draw()
        savefile = plotdir+"/profile_"+str(ivar+1)+"_pvsx.png"
        plt.savefig(savefile,format='png')
        plt.close()


def get_derivates_from_pol(bd_voltages_sel,bias_vol_floats_sel,bd_voltages_err,first_voltage_ders,fitname):

    for iv in range(len(bd_voltages_sel)):
        
        xvals, yvals,yerrs = [], [], []
        
        if(iv>0 and iv<(len(bd_voltages_sel)-1)):

            if(iv==1 or iv==(len(bd_voltages_sel)-2)):
                yvals.append(bd_voltages_sel[iv-1])
                yvals.append(bd_voltages_sel[iv])
                yvals.append(bd_voltages_sel[iv+1])
                xvals.append(bias_vol_floats_sel[iv-1])
                xvals.append(bias_vol_floats_sel[iv])
                xvals.append(bias_vol_floats_sel[iv+1])
                yerrs.append(bd_voltages_err[iv-1])
                yerrs.append(bd_voltages_err[iv])
                yerrs.append(bd_voltages_err[iv+1])
            else:
                yvals.append(bd_voltages_sel[iv-2])
                yvals.append(bd_voltages_sel[iv-1])
                yvals.append(bd_voltages_sel[iv])
                yvals.append(bd_voltages_sel[iv+1])
                yvals.append(bd_voltages_sel[iv+2])
                xvals.append(bias_vol_floats_sel[iv-2])
                xvals.append(bias_vol_floats_sel[iv-1])
                xvals.append(bias_vol_floats_sel[iv])
                xvals.append(bias_vol_floats_sel[iv+1])
                xvals.append(bias_vol_floats_sel[iv+2])
                yerrs.append(bd_voltages_err[iv-1])
                yerrs.append(bd_voltages_err[iv-2])
                yerrs.append(bd_voltages_err[iv])
                yerrs.append(bd_voltages_err[iv+1])
                yerrs.append(bd_voltages_err[iv+2])

            if args.verbose:
                print "bias",bias_vol_floats_sel[iv]
                print "Derivative before polynomial fit ",first_voltage_ders[iv]
    
            # for vth-vbias fit
            #init_values = [-0.5,max(0,min(first_voltage_ders[iv],10)),0.05]
            #ranges = ((-10000,0,-100),(100000,10,10000))
            # for vbias-vth fit
            init_values = [-0.5,max(0,max(0.1,1./first_voltage_ders[iv])),-0.05]
            ranges = ((-10000,-1.0,-5.),(100000,10,10))

            if args.nopstop:
                ranges = ((-10000,-500,-1000),(100000,1000,10000)) 
            if fitname=="second_der":
                init_values = [0,max(0,min(first_voltage_ders[iv],10)),1]
                ranges = ((-10000,0,-10),(100000,10,10000))

            poptx_der, pcovx_der = curve_fit(polfunc2, np.array(xvals),np.array(yvals), sigma=yerrs, maxfev=5000, absolute_sigma=True, p0=init_values, bounds=ranges)
            
            if args.verbose:
                print poptx_der

            yvals_exp = polfunc2(np.array(xvals), *poptx_der)

            plt.figure("first_der_fit",figsize=(14.0, 10.5))
            #plt.title(title, size=fontsize)
            plt.plot(xvals,yvals,'-o',xvals,yvals_exp,'-')
            plt.ylabel('V$_{threshold}$ [V]')
            plt.xlabel('V$_{bias}$ [V]')
            savefile = plotdir+"/"+fitname+"_fit"+bias_voltages[iv]+".png"
            plt.savefig(savefile,format='png')
            plt.close()
            #print poptx_der
            f1, f2 = 0, 0
            f1 = poptx_der[1] + poptx_der[2]*(xvals[len(xvals)/2]*xvals[len(xvals)/2]-xvals[len(xvals)/2-1]*xvals[len(xvals)/2-1])/(xvals[len(xvals)/2]-xvals[len(xvals)/2-1])
            f2 = poptx_der[1] + poptx_der[2]*(xvals[len(xvals)/2+1]*xvals[len(xvals)/2+1]-xvals[len(xvals)/2]*xvals[len(xvals)/2])/(xvals[len(xvals)/2+1]-xvals[len(xvals)/2])
            
            #first_voltage_ders[iv] = (f1+f2)/2
            first_voltage_ders[iv] = (1./f1 + 1./f2)/2
            if args.verbose:
                print "Derivative after polynomial fit ",first_voltage_ders[iv]
        
        del xvals, yvals

files = []
'''
file_names = [
'test_bias_minus_1V_refined-2023-02-28T16-07-02.txt',
'test_bias_minus_2V_refined-2023-02-28T16-17-27.txt',
'test_bias_minus_3V_refined-2023-02-28T16-28-33.txt',
'test_bias_minus_4V_refined-2023-02-28T16-41-14.txt',
'test_bias_minus_5V_refined-2023-02-28T16-52-02.txt',
'test_bias_minus_6V_refined-2023-02-28T17-14-07.txt',
'test_bias_minus_7V_refined-2023-02-28T17-24-52.txt',
'test_bias_minus_8V_refined-2023-02-28T17-37-39.txt',
'test_bias_minus_9V_refined-2023-02-28T17-49-34.txt',
]
'''
'''
folder = '/home/suman/CMS/HGCal/MOSEFT/Data/Tracker2S_LowThreshold/'
plotdir = 'Tracker2S_LowThreshold'
title = 'MOSFET (Tracker 2S low-threshold)'
file_names = [
'test_bias_minus_0V-2023-02-28T10-41-53.txt',
#'test_bias_minus_1V-2023-02-28T10-58-45.txt',
#'test_bias_minus_2V-2023-02-28T11-07-37.txt',
#'test_bias_minus_3V-2023-02-28T11-16-01.txt',
#'test_bias_minus_4V-2023-02-28T11-24-15.txt',
#'test_bias_minus_5V-2023-02-28T11-32-18.txt',
#'test_bias_minus_6V-2023-02-28T11-40-16.txt',
#'test_bias_minus_7V-2023-02-28T11-48-19.txt',
#'test_bias_minus_8V-2023-02-28T11-56-45.txt',
#'test_bias_minus_9V-2023-02-28T12-05-13.txt',
#'test_bias_minus_10V-2023-02-28T12-13-50.txt',
'test_bias_minus_1V_round2-2023-02-28T18-00-36.txt',
'test_bias_minus_2V_round2-2023-02-28T18-08-29.txt',
'test_bias_minus_3V_round2-2023-02-28T18-17-41.txt',
'test_bias_minus_4V_round2-2023-02-28T18-26-13.txt',
'test_bias_minus_5V_repeat-2023-02-28T15-43-11.txt',
'test_bias_minus_6V_round2-2023-02-28T18-56-04.txt',
'test_bias_minus_7V_round2-2023-02-28T19-04-25.txt',
'test_bias_minus_8V_round2-2023-02-28T19-14-07.txt',
'test_bias_minus_9V_round2-2023-02-28T19-22-31.txt',
'test_bias_minus_10V_round2-2023-02-28T19-30-41.txt',
'test_bias_minus_12V-2023-02-28T15-26-45.txt',
'test_bias_minus_14V-2023-02-28T15-16-38.txt',
'test_bias_minus_16V-2023-02-28T15-06-00.txt',
'test_bias_minus_18V-2023-02-28T14-57-43.txt',
'test_bias_minus_20V-2023-02-28T12-21-58.txt',
'test_bias_minus_30V-2023-02-28T13-49-11.txt',
'test_bias_minus_40V-2023-02-28T13-57-34.txt',
'test_bias_minus_50V-2023-02-28T14-05-43.txt',
'test_bias_minus_75V-2023-02-28T14-14-07.txt',
'test_bias_minus_100V-2023-02-28T10-49-56.txt',
'test_bias_minus_150V-2023-03-06T12-12-02.txt',
'test_bias_minus_200V-2023-02-28T14-22-19.txt',
'test_bias_minus_300V-2023-02-28T14-30-33.txt',
#'test_bias_minus_500V-2023-02-28T14-47-36.txt'
]
'''
'''
folder = '/home/suman/CMS/HGCal/MOSEFT/Data/TrackerPSS_round/'
plotdir = 'TrackerPS_round'
title = 'MOSFET (Tracker PS round)'
file_names = [
'test_bias_minus_0V-2023-02-20T15-20-59.txt',
'test_bias_minus_1V-2023-02-20T11-59-02.txt',
'test_bias_minus_2V-2023-02-20T12-06-45.txt',
'test_bias_minus_3V-2023-02-20T12-13-25.txt',
'test_bias_minus_4V-2023-02-20T12-19-11.txt',
'test_bias_minus_5V-2023-02-20T12-24-58.txt',
'test_bias_minus_6V-2023-02-20T12-30-38.txt',
'test_bias_minus_7V-2023-02-20T12-35-56.txt',
'test_bias_minus_8V-2023-02-20T12-41-18.txt',
'test_bias_minus_9V-2023-02-20T12-46-45.txt',
'test_bias_minus_10V-2023-02-20T13-06-16.txt',
'test_bias_minus_20V-2023-02-20T13-53-14.txt',
'test_bias_minus_30V-2023-02-20T13-58-49.txt',
'test_bias_minus_40V-2023-02-20T14-44-57.txt',
'test_bias_minus_50V-2023-02-20T14-50-36.txt',
'test_bias_minus_100V-2023-02-20T14-56-00.txt',
'test_bias_minus_200V-2023-02-20T15-01-17.txt',
'test_bias_minus_300V-2023-02-20T15-06-49.txt',
'test_bias_minus_500V-2023-02-20T15-13-49.txt'
]
'''
'''
folder = '/home/suman/CMS/HGCal/MOSEFT/Data/300um/'
plotdir = 'HGCal_300um'
title = 'MOSFET (HGcal 300 $\mu$m)'
file_names = [
'test_bias_minus_0V-2023-01-24T17-44-41.txt',
'test_bias_minus_1V-2023-01-24T17-37-45.txt',
'test_bias_minus_2V-2023-01-24T17-30-13.txt',
'test_bias_minus_3V-2023-01-24T17-23-22.txt',
'test_bias_minus_4V-2023-01-24T17-16-31.txt',
'test_bias_minus_5V-2023-01-24T17-09-25.txt',
'test_bias_minus_6V-2023-01-24T17-02-01.txt',
#'test_bias_minus_7V-2023-01-24T16-55-00.txt',
#'test_bias_minus_8V-2023-01-24T16-47-14.txt',
'test_bias_minus_9V-2023-01-24T16-40-17.txt',
'test_bias_minus_10V-2023-01-24T16-30-40.txt',
'test_bias_minus_20V-2023-01-30T10-08-40.txt',
'test_bias_minus_30V-2023-01-30T10-18-55.txt',
'test_bias_minus_40V-2023-01-30T11-19-17.txt',
'test_bias_minus_50V-2023-01-30T10-35-06.txt',
'test_bias_minus_100V-2023-01-30T10-42-42.txt',
'test_bias_minus_200V-2023-01-30T10-49-47.txt',
'test_bias_minus_300V-2023-01-30T10-57-41.txt',
'test_bias_minus_500V-2023-01-30T11-04-42.txt',
#'test_bias_minus_1000V-2023-01-30T11-11-41.txt'
]
'''
'''
folder = '/home/suman/CMS/HGCal/MOSEFT/Data/Tracker2S/'
plotdir = 'Tracker2S'
title = 'MOSFET (Tracker 2S)'
file_names = [
'test_bias_minus_0V-2023-01-31T10-49-01.txt',
'test_bias_minus_1V-2023-01-31T11-01-09.txt',
'test_bias_minus_2V-2023-01-31T11-08-19.txt',
'test_bias_minus_3V-2023-01-31T11-15-10.txt',
'test_bias_minus_4V-2023-01-31T11-29-10.txt',
'test_bias_minus_5V-2023-01-31T11-36-49.txt',
'test_bias_minus_6V-2023-01-31T11-43-47.txt',
'test_bias_minus_7V-2023-01-31T11-50-30.txt',
'test_bias_minus_8V-2023-01-31T11-58-06.txt',
'test_bias_minus_9V-2023-01-31T12-04-54.txt',
'test_bias_minus_10V-2023-01-31T12-11-46.txt',
'test_bias_minus_20V-2023-01-31T14-24-00.txt',
'test_bias_minus_30V-2023-01-31T14-31-36.txt',
'test_bias_minus_40V-2023-01-31T14-38-44.txt',
'test_bias_minus_50V-2023-01-31T14-45-53.txt',
'test_bias_minus_100V-2023-01-31T14-52-44.txt',
'test_bias_minus_200V-2023-02-10T14-34-14.txt',
'test_bias_minus_300V-2023-02-10T14-41-57.txt',
'test_bias_minus_500V-2023-02-10T14-56-15.txt'
]
'''
'''
folder = '/home/suman/CMS/HGCal/MOSEFT/Data/Tracker2S_Round/'
plotdir = 'Tracker2S_round'
title = 'MOSFET (Tracker 2S round)'
file_names = [
'test_bias_minus_0V-2023-02-20T16-46-03.txt',
'test_bias_minus_1V-2023-02-20T16-54-35.txt',
'test_bias_minus_2V-2023-02-20T18-24-41.txt',
'test_bias_minus_3V-2023-02-20T18-31-40.txt',
'test_bias_minus_4V-2023-02-20T18-52-01.txt',
'test_bias_minus_5V-2023-02-20T18-59-04.txt',
'test_bias_minus_6V-2023-02-20T19-05-49.txt',
'test_bias_minus_7V-2023-02-20T19-12-51.txt',
'test_bias_minus_8V-2023-02-20T19-19-58.txt',
'test_bias_minus_9V-2023-02-20T19-28-04.txt',
'test_bias_minus_10V-2023-02-20T19-34-52.txt',
'test_bias_minus_20V-2023-02-20T19-42-40.txt',
'test_bias_minus_30V-2023-02-20T19-49-52.txt',
'test_bias_minus_40V-2023-02-20T19-56-31.txt',
'test_bias_minus_50V-2023-02-20T20-03-08.txt',
'test_bias_minus_100V-2023-02-20T20-09-56.txt',
'test_bias_minus_200V-2023-02-20T20-16-33.txt',
'test_bias_minus_300V-2023-02-20T20-40-48.txt',
'test_bias_minus_500V-2023-02-20T20-34-11.txt'
]
'''
'''
folder = '/home/suman/CMS/HGCal/MOSEFT/Data/Tracker2S_Round_nopstop/'
plotdir = 'Tracker2S_round_nopstop'
title = 'MOSFET (Tracker 2S round no-pstop)'
file_names = [
'test_bias_minus_0V-2023-03-13T12-22-59.txt',
'test_bias_minus_1V-2023-03-13T13-39-09.txt',
'test_bias_minus_2V-2023-03-13T13-47-38.txt',
'test_bias_minus_3V-2023-03-13T15-09-07.txt',
'test_bias_minus_4V-2023-03-13T15-18-00.txt',
'test_bias_minus_5V-2023-03-13T15-26-47.txt',
'test_bias_minus_6V-2023-03-13T15-35-02.txt',
'test_bias_minus_7V-2023-03-13T15-44-33.txt',
'test_bias_minus_8V-2023-03-13T15-53-21.txt',
'test_bias_minus_9V-2023-03-13T16-08-49.txt',
'test_bias_minus_10V-2023-03-13T12-32-03.txt',
'test_bias_minus_20V-2023-03-13T16-20-38.txt',
'test_bias_minus_30V-2023-03-13T16-29-29.txt',
'test_bias_minus_40V-2023-03-13T16-38-48.txt',
'test_bias_minus_50V-2023-03-13T16-50-19.txt',
'test_bias_minus_100V-2023-03-13T12-53-11.txt',
'test_bias_minus_200V-2023-03-13T17-10-01.txt',
'test_bias_minus_300V-2023-03-13T17-26-53.txt',
'test_bias_minus_500V-2023-03-13T17-43-25.txt'
]
'''

folder = '/home/suman/CMS/HGCal/MOSEFT/Data/HGcal_Common_Round/'
plotdir = 'HGcal_round_common'
title = 'MOSFET (HGcal round)'
file_names = [
'test_bias_minus_0p0V-2023-03-14T15-18-28.txt',
'test_bias_minus_1p0V-2023-03-14T15-54-39.txt',
'test_bias_minus_2p0V-2023-03-14T16-03-40.txt',
'test_bias_minus_3p0V-2023-03-14T16-18-24.txt',
'test_bias_minus_4p0V-2023-03-14T16-27-25.txt',
'test_bias_minus_5p0V-2023-03-14T16-36-30.txt',
'test_bias_minus_6p0V-2023-03-14T16-45-13.txt',
'test_bias_minus_7p0V-2023-03-14T16-53-46.txt',
'test_bias_minus_8p0V-2023-03-14T17-08-41.txt',
'test_bias_minus_9p0V-2023-03-14T17-16-58.txt',
'test_bias_minus_10p0V-2023-03-14T17-25-27.txt',
'test_bias_minus_12p0V-2023-03-14T17-33-39.txt',
'test_bias_minus_14p0V-2023-03-14T17-41-58.txt',
'test_bias_minus_16p0V-2023-03-14T17-50-15.txt',
'test_bias_minus_18p0V-2023-03-14T18-00-32.txt',
'test_bias_minus_20p0V-2023-03-14T18-09-37.txt',
'test_bias_minus_30p0V-2023-03-14T18-18-12.txt',
'test_bias_minus_40p0V-2023-03-14T18-26-35.txt',
'test_bias_minus_50p0V-2023-03-14T18-57-52.txt',
'test_bias_minus_75p0V-2023-03-14T19-05-52.txt',
'test_bias_minus_100p0V-2023-03-14T19-14-00.txt',
'test_bias_minus_200p0V-2023-03-14T19-22-44.txt',
'test_bias_minus_300p0V-2023-03-14T19-31-59.txt'
]
VGS_max=20
VGS_min=0
mu0_max = 25000
v1by2_max = 40
v1by2_min = 0
'''
folder = '/home/suman/CMS/HGCal/MOSEFT/Data/HGcal_200um_radiated/'
plotdir = 'HGcal_200um_radiated'
title = 'MOSFET (HGcal 200 $\mu$m) Irradiated'
file_names = [
'test_bias_minus_0V-2023-04-17T12-37-47.txt',
'test_bias_minus_1V-2023-04-17T13-53-57.txt',
'test_bias_minus_2V-2023-04-17T16-23-39.txt',
'test_bias_minus_3V-2023-04-17T16-33-53.txt',
'test_bias_minus_4V-2023-04-17T16-42-08.txt',
'test_bias_minus_5V-2023-04-17T16-50-58.txt',
#'test_bias_minus_6V-2023-04-17T16-59-28.txt',
#'test_bias_minus_7V-2023-04-17T17-08-20.txt',
#'test_bias_minus_8V-2023-04-17T17-16-48.txt',
#'test_bias_minus_9V-2023-04-17T17-25-20.txt',
'test_bias_minus_10V-2023-04-17T17-33-50.txt',
'test_bias_minus_12V-2023-04-17T17-42-06.txt',
'test_bias_minus_14V-2023-04-17T17-50-02.txt',
'test_bias_minus_16V-2023-04-17T17-57-50.txt',
'test_bias_minus_18V-2023-04-17T18-05-50.txt',
'test_bias_minus_20V-2023-04-17T18-14-06.txt',
'test_bias_minus_30V-2023-04-17T18-22-13.txt',
'test_bias_minus_40V-2023-04-17T18-30-27.txt',
'test_bias_minus_50V-2023-04-17T18-38-44.txt',
'test_bias_minus_75V-2023-04-17T18-47-47.txt',
'test_bias_minus_100V-2023-04-17T18-56-12.txt',
'test_bias_minus_150V-2023-04-17T19-04-08.txt',
'test_bias_minus_200V-2023-04-17T19-12-11.txt',
'test_bias_minus_300V-2023-04-17T19-36-06.txt',
'test_bias_minus_500V-2023-04-17T19-50-44.txt'
]
'''
'''
folder = '/home/suman/CMS/HGCal/MOSEFT/Data/Hamburg/M200_MOSFET/M200P_analysis/'
plotdir = 'UHH_M2000P'
title = 'MOSFET (MS round)'
file_names = [
'test_bias_minus_0V-2016.txt',
'test_bias_minus_5V-2016.txt',
'test_bias_minus_10V-2016.txt',
'test_bias_minus_15V-2016.txt',
'test_bias_minus_20V-2016.txt',
'test_bias_minus_25V-2016.txt',
'test_bias_minus_30V-2016.txt',
'test_bias_minus_35V-2016.txt',
'test_bias_minus_40V-2016.txt',
'test_bias_minus_45V-2016.txt',
'test_bias_minus_50V-2016.txt',
'test_bias_minus_55V-2016.txt',
'test_bias_minus_60V-2016.txt',
'test_bias_minus_65V-2016.txt',
'test_bias_minus_70V-2016.txt',
'test_bias_minus_75V-2016.txt',
'test_bias_minus_80V-2016.txt',
'test_bias_minus_85V-2016.txt',
'test_bias_minus_90V-2016.txt',
'test_bias_minus_95V-2016.txt',
'test_bias_minus_100V-2016.txt',
'test_bias_minus_105V-2016.txt',
'test_bias_minus_110V-2016.txt',
'test_bias_minus_115V-2016.txt',
'test_bias_minus_120V-2016.txt',
]
'''
if args.isHamburg:
    if args.nopstop:
        if args.LargeSet:
            folder = '/home/suman/CMS/HGCal/MOSEFT/Data/Hamburg/M200_MOSFET/M200P_analysis/'
            plotdir = 'UHH_M2000P_LargeSet'
            title = 'MOSFET (MS round) no p-spray'
            file_names = [
            'test_bias_minus_0p0V-2016.txt',
            'test_bias_minus_5p0V-2016.txt',
            'test_bias_minus_10p0V-2016.txt',
            'test_bias_minus_15p0V-2016.txt',
            'test_bias_minus_20p0V-2016.txt',
            'test_bias_minus_25p0V-2016.txt',
            'test_bias_minus_30p0V-2016.txt',
            'test_bias_minus_35p0V-2016.txt',
            'test_bias_minus_40p0V-2016.txt',
            'test_bias_minus_45p0V-2016.txt',
            'test_bias_minus_50p0V-2016.txt',
            'test_bias_minus_55p0V-2016.txt',
            'test_bias_minus_60p0V-2016.txt',
            'test_bias_minus_65p0V-2016.txt',
            'test_bias_minus_70p0V-2016.txt',
            'test_bias_minus_75p0V-2016.txt',
            'test_bias_minus_80p0V-2016.txt',
            'test_bias_minus_85p0V-2016.txt',
            'test_bias_minus_90p0V-2016.txt',
            'test_bias_minus_95p0V-2016.txt',
            'test_bias_minus_100p0V-2016.txt',
            'test_bias_minus_105p0V-2016.txt',
            'test_bias_minus_110p0V-2016.txt',
            'test_bias_minus_115p0V-2016.txt',
            'test_bias_minus_120p0V-2016.txt',
            ]
        else:
            folder = '/home/suman/CMS/HGCal/MOSEFT/Data/Hamburg/M200_MOSFET/M200P_analysis/Set1/'
            plotdir = 'UHH_M2000P'
            title = 'MOSFET (MS round) no p-spray'
            file_names = [
            'test_bias_minus_0p0V-2016.txt',
            'test_bias_minus_0p5V-2016.txt',
            'test_bias_minus_1p0V-2016.txt',
            'test_bias_minus_2p0V-2016.txt',
            'test_bias_minus_3p0V-2016.txt',
            'test_bias_minus_5p0V-2016.txt',
            'test_bias_minus_10p0V-2016.txt',
            'test_bias_minus_20p0V-2016.txt',
            'test_bias_minus_30p0V-2016.txt',
            ]
    else:
        folder = '/home/suman/CMS/HGCal/MOSEFT/Data/Hamburg/M200_MOSFET/M200Y_analysis/Set4/'
        plotdir = 'UHH_M2000Y_Set4'
        title = 'MOSFET (MS round) p-spray'
        file_names = [
        'test_bias_minus_0p5NegV-2016.txt',
        'test_bias_minus_0p4NegV-2016.txt',
        'test_bias_minus_0p3NegV-2016.txt',
        'test_bias_minus_0p2NegV-2016.txt',
        'test_bias_minus_0p1NegV-2016.txt',
        'test_bias_minus_0p0V-2016.txt',
        'test_bias_minus_0p1V-2016.txt',
        'test_bias_minus_0p2V-2016.txt',
        'test_bias_minus_0p3V-2016.txt',
        'test_bias_minus_0p5V-2016.txt',
        'test_bias_minus_0p7V-2016.txt',
        'test_bias_minus_0p9V-2016.txt',
        'test_bias_minus_1p1V-2016.txt',
        'test_bias_minus_1p3V-2016.txt',
        'test_bias_minus_1p5V-2016.txt',
        'test_bias_minus_1p7V-2016.txt',
        'test_bias_minus_1p9V-2016.txt',
        'test_bias_minus_2p1V-2016.txt',
        'test_bias_minus_2p3V-2016.txt',
        'test_bias_minus_2p5V-2016.txt',
        'test_bias_minus_3p0V-2016.txt',
        'test_bias_minus_3p5V-2016.txt',
        'test_bias_minus_4p0V-2016.txt',
        'test_bias_minus_4p5V-2016.txt',
        'test_bias_minus_5p0V-2016.txt',
        'test_bias_minus_6p0V-2016.txt',
        'test_bias_minus_7p0V-2016.txt',
        'test_bias_minus_9p0V-2016.txt',
        'test_bias_minus_12p0V-2016.txt',
        'test_bias_minus_16p0V-2016.txt',
        'test_bias_minus_20p0V-2016.txt',
        'test_bias_minus_25p0V-2016.txt',
        'test_bias_minus_30p0V-2016.txt',
        ]
    VGS_max=20
    VGS_min=0
    mu0_max = 1400
    mu0_min = 900
    v1by2_max = 80
    v1by2_min = 20
else:
    if args.isVienna:
        if args.nopstop:
            folder = '/home/suman/CMS/HGCal/MOSEFT/Data/HGcal_nostop_Round-ELM/'
            plotdir = 'HGcal_nostop_round_ELM'
            title = 'MOSFET (HGcal round no p-stop)'
            file_names = [
            'test_bias_minus_0p3NegV-2023-08-03T23-34-29.txt',
            'test_bias_minus_0p2NegV-2023-08-03T23-03-53.txt',
            #'test_bias_minus_0p1NegV-2023-08-03T22-45-45.txt',
            'test_bias_minus_0p0V-2023-08-02T10-05-30.txt',
            'test_bias_minus_0p1V-2023-08-02T10-40-51.txt',
            'test_bias_minus_0p2V-2023-08-02T11-00-36.txt',
            'test_bias_minus_0p3V-2023-08-02T11-16-06.txt',
            'test_bias_minus_0p4V-2023-08-02T11-36-20.txt',
            'test_bias_minus_0p5V-2023-08-02T11-49-24.txt',
            'test_bias_minus_0p6V-2023-08-02T12-02-49.txt',
            'test_bias_minus_0p7V-2023-08-02T12-22-45.txt',
            'test_bias_minus_0p8V-2023-08-02T12-36-55.txt',
            'test_bias_minus_0p9V-2023-08-02T12-53-39.txt',
            'test_bias_minus_1p0V-2023-08-02T13-06-04.txt',
            'test_bias_minus_1p1V-2023-08-02T14-16-04.txt',
            'test_bias_minus_1p2V-2023-08-02T14-28-45.txt',
            'test_bias_minus_1p4V-2023-08-02T14-45-34.txt',
            'test_bias_minus_1p6V-2023-08-02T15-45-27.txt',
            'test_bias_minus_1p8V-2023-08-02T16-04-09.txt',
            'test_bias_minus_2p0V-2023-08-02T16-17-53.txt',
            'test_bias_minus_2p2V-2023-08-02T16-30-12.txt',
            'test_bias_minus_2p4V-2023-08-02T16-47-43.txt',
            'test_bias_minus_2p6V-2023-08-02T17-03-55.txt',
            'test_bias_minus_2p8V-2023-08-02T17-16-45.txt',
            'test_bias_minus_3p0V-2023-08-02T17-29-35.txt',
            'test_bias_minus_3p5V-2023-08-02T18-30-39.txt',
            'test_bias_minus_4p0V-2023-08-02T18-49-52.txt',
            'test_bias_minus_4p5V-2023-08-02T19-02-23.txt',
            'test_bias_minus_5p0V-2023-08-02T19-21-54.txt',
            'test_bias_minus_5p5V-2023-08-02T19-34-34.txt',
            'test_bias_minus_6p0V-2023-08-02T19-50-54.txt',
            'test_bias_minus_6p5V-2023-08-02T21-32-38.txt',
            'test_bias_minus_7p0V-2023-08-02T22-18-39.txt',
            'test_bias_minus_7p5V-2023-08-02T22-38-23.txt',
            'test_bias_minus_8p0V-2023-08-02T22-54-01.txt',
            'test_bias_minus_9p0V-2023-08-02T23-09-02.txt',
            'test_bias_minus_10p0V-2023-08-02T23-47-24.txt',
            'test_bias_minus_11p0V-2023-08-03T00-02-30.txt',
            'test_bias_minus_12p0V-2023-08-03T00-18-53.txt',
            'test_bias_minus_13p0V-2023-08-03T00-31-35.txt',
            'test_bias_minus_14p0V-2023-08-03T10-34-26.txt',
            'test_bias_minus_15p0V-2023-08-03T11-00-27.txt',
            'test_bias_minus_16p0V-2023-08-03T11-21-48.txt',
            'test_bias_minus_17p0V-2023-08-03T11-34-30.txt',
            'test_bias_minus_18p0V-2023-08-03T11-47-14.txt',
            'test_bias_minus_19p0V-2023-08-03T11-59-47.txt',
            'test_bias_minus_20p0V-2023-08-03T12-18-32.txt',
            'test_bias_minus_22p0V-2023-08-03T12-30-56.txt',
            'test_bias_minus_24p0V-2023-08-03T12-43-40.txt',
            'test_bias_minus_26p0V-2023-08-03T12-56-39.txt',
            'test_bias_minus_28p0V-2023-08-03T13-10-57.txt',
            'test_bias_minus_30p0V-2023-08-03T14-09-34.txt',
            'test_bias_minus_35p0V-2023-08-03T14-39-52.txt',
            'test_bias_minus_40p0V-2023-08-03T14-56-34.txt',
            'test_bias_minus_45p0V-2023-08-03T15-30-38.txt',
            'test_bias_minus_50p0V-2023-08-04T16-25-01.txt',
            'test_bias_minus_60p0V-2023-08-03T16-13-02.txt',
            'test_bias_minus_70p0V-2023-08-03T16-34-14.txt',
            'test_bias_minus_80p0V-2023-08-03T16-51-18.txt',
            'test_bias_minus_90p0V-2023-08-03T17-09-52.txt',
            'test_bias_minus_100p0V-2023-08-03T18-15-33.txt',
            'test_bias_minus_110p0V-2023-08-03T18-28-51.txt',
            'test_bias_minus_120p0V-2023-08-03T19-29-19.txt',
            'test_bias_minus_130p0V-2023-08-03T19-41-42.txt',
            'test_bias_minus_140p0V-2023-08-03T19-55-59.txt',
            'test_bias_minus_150p0V-2023-08-03T20-21-34.txt',
            'test_bias_minus_160p0V-2023-08-03T20-38-47.txt',
            'test_bias_minus_170p0V-2023-08-03T20-51-30.txt',
            'test_bias_minus_180p0V-2023-08-03T21-04-04.txt',
            'test_bias_minus_190p0V-2023-08-03T21-18-49.txt',
            'test_bias_minus_200p0V-2023-08-03T21-31-25.txt',
            'test_bias_minus_250p0V-2023-08-03T22-10-31.txt',
            'test_bias_minus_300p0V-2023-08-03T22-23-55.txt'
        ]
        elif args.repeated:
            folder = '/home/suman/CMS/HGCal/MOSEFT/Data/HGcal_OTLL_Round_nopstop_Repeat/'
            plotdir = 'HGcal_OTLL_round_ELM_nopstop_Repeated'
            title = 'MOSFET (HGcal OTLL round)'
            file_names = [
            'test_bias_minus_0p0V-repeat1-2023-08-07T09-12-40.txt',
            'test_bias_minus_0p0V-repeat2-2023-08-07T09-25-05.txt',
            'test_bias_minus_0p0V-repeat3-2023-08-07T15-02-11.txt',
            'test_bias_minus_0p0V-repeat4-2023-08-07T15-43-11.txt',
            'test_bias_minus_0p0V-repeat5-2023-08-07T15-57-35.txt',
            'test_bias_minus_0p0V-repeat6-2023-08-07T16-14-10.txt',
            'test_bias_minus_0p0V-repeat7-2023-08-07T17-00-22.txt',
            'test_bias_minus_0p0V-repeat8-2023-08-07T17-21-38.txt',
            'test_bias_minus_0p0V-repeat9-2023-08-07T17-47-10.txt',
            'test_bias_minus_0p0V-repeat10-2023-08-07T18-15-05.txt'
            ]
        elif args.HGC_Common:
            folder = '/home/suman/CMS/HGCal/MOSEFT/Data/HGcal_Common_Round-ELM/'
            plotdir = 'HGcal_Common_round_ELM'
            title = 'MOSFET (HGcal Common round)'
            file_names = [
                #'test_bias_minus_0p6NegV-2023-08-12T19-04-51.txt',
                #'test_bias_minus_0p5NegV-2023-08-12T18-49-14.txt',
                #'test_bias_minus_0p4NegV-2023-08-12T18-30-48.txt',
                'test_bias_minus_0p3NegV-2023-08-12T18-17-21.txt',
                'test_bias_minus_0p2NegV-2023-08-12T18-04-12.txt',
                'test_bias_minus_0p1NegV-2023-08-12T17-50-55.txt',
                'test_bias_minus_0p0V-2023-08-10T14-38-27.txt',
                'test_bias_minus_0p1V-2023-08-10T14-54-04.txt',
                'test_bias_minus_0p2V-2023-08-10T15-07-21.txt',
                'test_bias_minus_0p3V-2023-08-10T15-20-24.txt',
                'test_bias_minus_0p4V-2023-08-10T15-33-58.txt',
                'test_bias_minus_0p5V-2023-08-10T15-47-15.txt',
                'test_bias_minus_0p6V-2023-08-10T16-00-22.txt',
                'test_bias_minus_0p7V-2023-08-10T16-15-12.txt',
                'test_bias_minus_0p8V-2023-08-10T16-40-57.txt',
                'test_bias_minus_0p9V-2023-08-10T16-53-48.txt',
                'test_bias_minus_1p0V-2023-08-10T17-06-57.txt',
                'test_bias_minus_1p1V-2023-08-10T17-23-16.txt',
                'test_bias_minus_1p2V-2023-08-10T17-37-54.txt',
                'test_bias_minus_1p3V-2023-08-10T17-50-52.txt',
                #'test_bias_minus_1p4V-2023-08-10T18-08-56.txt',
                'test_bias_minus_1p4V-2023-08-12T21-22-06.txt',
                'test_bias_minus_1p5V-2023-08-12T22-06-23.txt',
                #'test_bias_minus_1p6V-2023-08-10T18-28-50.txt',
                'test_bias_minus_1p6V-2023-08-12T22-20-03.txt',
                'test_bias_minus_1p8V-2023-08-10T18-42-05.txt',
                'test_bias_minus_2p0V-2023-08-10T20-42-19.txt',
                'test_bias_minus_2p2V-2023-08-10T21-10-14.txt',
                'test_bias_minus_2p4V-2023-08-10T21-41-39.txt',
                'test_bias_minus_2p6V-2023-08-10T21-56-16.txt',
                #'test_bias_minus_2p8V-2023-08-10T22-11-48.txt',
                'test_bias_minus_3p0V-2023-08-10T22-41-07.txt',
                'test_bias_minus_3p5V-2023-08-10T22-54-10.txt',
                'test_bias_minus_4p0V-2023-08-10T23-36-42.txt',
                'test_bias_minus_4p5V-2023-08-10T23-51-59.txt',
                'test_bias_minus_5p0V-2023-08-11T00-08-13.txt',
                'test_bias_minus_5p5V-2023-08-11T10-08-03.txt',
                'test_bias_minus_6p0V-2023-08-11T10-28-35.txt',
                'test_bias_minus_6p5V-2023-08-11T11-07-27.txt',
                'test_bias_minus_7p0V-2023-08-11T11-20-37.txt',
                'test_bias_minus_7p5V-2023-08-11T11-35-13.txt',
                'test_bias_minus_8p0V-2023-08-11T11-49-40.txt',
                'test_bias_minus_9p0V-2023-08-11T12-06-53.txt',
                'test_bias_minus_10p0V-2023-08-11T12-19-56.txt',
                'test_bias_minus_11p0V-2023-08-11T12-37-33.txt',
                'test_bias_minus_12p0V-2023-08-11T14-19-51.txt',
                'test_bias_minus_13p0V-2023-08-11T14-40-32.txt',
                'test_bias_minus_14p0V-2023-08-11T14-57-57.txt',
                'test_bias_minus_15p0V-2023-08-11T15-11-56.txt',
                'test_bias_minus_16p0V-2023-08-11T15-25-17.txt',
                'test_bias_minus_17p0V-2023-08-11T15-48-32.txt',
                'test_bias_minus_18p0V-2023-08-11T16-15-58.txt',
                #'test_bias_minus_19p0V-2023-08-11T16-40-50.txt',
                'test_bias_minus_20p0V-2023-08-11T16-54-37.txt',
                'test_bias_minus_22p0V-2023-08-11T17-08-22.txt',
                'test_bias_minus_24p0V-2023-08-11T17-29-43.txt',
                #'test_bias_minus_26p0V-2023-08-11T17-54-21.txt',
                'test_bias_minus_28p0V-2023-08-11T18-09-36.txt',
                'test_bias_minus_30p0V-2023-08-11T18-28-29.txt',
                'test_bias_minus_35p0V-2023-08-11T22-36-18.txt',
                'test_bias_minus_40p0V-2023-08-11T22-56-09.txt',
                'test_bias_minus_45p0V-2023-08-11T23-26-53.txt',
                'test_bias_minus_50p0V-2023-08-12T00-04-10.txt',
                'test_bias_minus_60p0V-2023-08-12T00-39-57.txt',
                'test_bias_minus_70p0V-2023-08-12T00-53-48.txt',
                'test_bias_minus_80p0V-2023-08-12T01-06-57.txt',
                'test_bias_minus_90p0V-2023-08-12T01-21-58.txt',
                'test_bias_minus_100p0V-2023-08-12T11-39-39.txt',
                'test_bias_minus_110p0V-2023-08-12T12-05-16.txt',
                'test_bias_minus_120p0V-2023-08-12T12-43-04.txt',
                'test_bias_minus_130p0V-2023-08-12T12-56-59.txt',
                'test_bias_minus_140p0V-2023-08-12T13-13-35.txt',
                'test_bias_minus_150p0V-2023-08-12T13-43-14.txt',
                'test_bias_minus_160p0V-2023-08-12T14-04-14.txt',
                'test_bias_minus_170p0V-2023-08-12T14-43-56.txt',
                #'test_bias_minus_180p0V-2023-08-12T15-47-14.txt',
                'test_bias_minus_190p0V-2023-08-12T16-00-32.txt',
                'test_bias_minus_200p0V-2023-08-12T16-16-39.txt',
                'test_bias_minus_225p0V-2023-08-12T16-31-44.txt',
                'test_bias_minus_250p0V-2023-08-12T16-45-16.txt',
                'test_bias_minus_275p0V-2023-08-12T17-03-16.txt',
                'test_bias_minus_300p0V-2023-08-12T17-27-15.txt'
            ]
        else:
            folder = '/home/suman/CMS/HGCal/MOSEFT/Data/HGcal_OTLL_Round-ELM/'
            plotdir = 'HGcal_OTLL_round_ELM'
            title = 'MOSFET (HGcal OTLL round)'
            file_names = [
                'test_bias_minus_0p3NegV-2023-07-18T20-38-51.txt',
                'test_bias_minus_0p2NegV-2023-07-18T20-09-35.txt',
                'test_bias_minus_0p1NegV-2023-08-02T09-14-00.txt',
                'test_bias_minus_0p0V-2023-07-18T18-39-40.txt',
                'test_bias_minus_0p1V-2023-07-31T20-12-00.txt',
                'test_bias_minus_0p2V-2023-07-31T20-37-51.txt',
                'test_bias_minus_0p3V-2023-07-31T21-34-13.txt',
                'test_bias_minus_0p4V-2023-07-31T21-49-34.txt',
                'test_bias_minus_0p5V-2023-07-31T22-05-29.txt',
                'test_bias_minus_0p6V-2023-07-31T22-19-52.txt',
                'test_bias_minus_0p7V-2023-07-31T23-06-01.txt',
                'test_bias_minus_0p8V-2023-07-31T23-27-57.txt',
                'test_bias_minus_0p9V-2023-07-31T23-41-39.txt',
                'test_bias_minus_1p0V-2023-07-31T23-54-46.txt',
                'test_bias_minus_1p1V-2023-08-01T10-01-51.txt',
                'test_bias_minus_1p2V-2023-08-01T10-29-32.txt',
                'test_bias_minus_1p4V-2023-08-01T12-01-51.txt',
                'test_bias_minus_1p6V-2023-08-01T12-22-56.txt',
                'test_bias_minus_1p8V-2023-08-01T12-57-08.txt',
                'test_bias_minus_2p0V-2023-08-01T14-02-17.txt',
                'test_bias_minus_2p2V-2023-08-01T14-27-44.txt',
                'test_bias_minus_2p4V-2023-08-01T14-42-13.txt',
                'test_bias_minus_2p6V-2023-08-01T14-58-50.txt',
                'test_bias_minus_2p8V-2023-08-01T15-13-35.txt',
                'test_bias_minus_3p0V-2023-08-01T15-32-39.txt',
                'test_bias_minus_3p5V-2023-08-01T15-46-17.txt',
                'test_bias_minus_4p0V-2023-08-01T16-00-50.txt',
                'test_bias_minus_4p5V-2023-08-01T16-16-21.txt',
                'test_bias_minus_5p0V-2023-07-17T17-19-46.txt',
                'test_bias_minus_5p5V-2023-07-17T17-35-30.txt',
                'test_bias_minus_6p0V-2023-07-17T18-07-08.txt',
                'test_bias_minus_6p5V-2023-07-17T18-22-00.txt',
                'test_bias_minus_7p0V-2023-07-17T18-40-16.txt',
                'test_bias_minus_7p5V-2023-07-17T18-55-04.txt',
                'test_bias_minus_8p0V-2023-07-17T19-08-31.txt',
                'test_bias_minus_9p0V-2023-07-17T20-06-52.txt',
                'test_bias_minus_10p0V-2023-08-01T16-31-34.txt',
                'test_bias_minus_11p0V-2023-07-17T21-02-10.txt',
                'test_bias_minus_12p0V-2023-07-17T21-26-51.txt',
                'test_bias_minus_13p0V-2023-07-17T22-02-17.txt',
                #'test_bias_minus_14p0V-2023-07-17T22-21-58.txt',
                'test_bias_minus_15p0V-2023-08-01T16-49-00.txt',
                'test_bias_minus_16p0V-2023-07-17T23-07-05.txt',
                'test_bias_minus_17p0V-2023-07-17T23-32-47.txt',
                'test_bias_minus_18p0V-2023-07-17T23-54-38.txt',
                #'test_bias_minus_19p0V-2023-07-18T00-08-55.txt',
                'test_bias_minus_20p0V-2023-07-18T00-22-46.txt',
                'test_bias_minus_22p0V-2023-07-18T09-27-57.txt',
                'test_bias_minus_24p0V-2023-07-18T10-31-00.txt',
                'test_bias_minus_26p0V-2023-07-18T10-44-43.txt',
                'test_bias_minus_28p0V-2023-07-18T11-03-59.txt',
                'test_bias_minus_30p0V-2023-07-18T11-18-36.txt',
                'test_bias_minus_35p0V-2023-07-18T11-36-37.txt',
                'test_bias_minus_40p0V-2023-07-18T12-17-27.txt',
                'test_bias_minus_45p0V-2023-07-18T12-33-48.txt',
                'test_bias_minus_50p0V-2023-07-18T12-59-04.txt',
                'test_bias_minus_60p0V-2023-07-18T13-13-10.txt',
                'test_bias_minus_70p0V-2023-07-18T14-09-08.txt',
                'test_bias_minus_80p0V-2023-07-18T14-23-10.txt',
                'test_bias_minus_90p0V-2023-07-18T14-38-42.txt',
                'test_bias_minus_100p0V-2023-07-18T14-54-31.txt',
                'test_bias_minus_110p0V-2023-07-18T16-33-25.txt',
                'test_bias_minus_120p0V-2023-07-18T16-46-55.txt',
                'test_bias_minus_130p0V-2023-07-18T17-02-07.txt',
                'test_bias_minus_140p0V-2023-07-18T17-16-46.txt',
                #'test_bias_minus_125p0V-2023-07-18T15-18-48.txt',
                'test_bias_minus_150p0V-2023-07-18T15-32-51.txt',
                'test_bias_minus_160p0V-2023-07-18T17-41-13.txt',
                'test_bias_minus_170p0V-2023-07-18T17-55-25.txt',
                'test_bias_minus_180p0V-2023-07-18T18-08-51.txt',
                'test_bias_minus_190p0V-2023-07-18T18-22-35.txt',
                'test_bias_minus_200p0V-2023-07-18T15-48-15.txt',
                'test_bias_minus_250p0V-2023-07-18T16-03-09.txt',
                'test_bias_minus_300p0V-2023-07-18T16-18-02.txt'
            ]

        VGS_max=40
        VGS_min=0
        mu0_max = 15000
        mu0_min = 3000
        v1by2_max = 40
        v1by2_min = 0

    else:
        if args.Vienna_Set2:
            folder = '/home/suman/CMS/HGCal/MOSEFT/Data/HGcal_Common_Round-Set2/'
            plotdir = 'HGcal_round_common_Set2'
            title = 'MOSFET (HGcal round)'
            file_names = [
            'test_bias_minus_0p4NegV-2023-06-06T20-27-38.txt',
            'test_bias_minus_0p3NegV-2023-06-06T20-43-19.txt',
            'test_bias_minus_0p2NegV-2023-06-06T20-16-18.txt',
            'test_bias_minus_0p1NegV-2023-06-06T20-10-55.txt',
            #'test_bias_minus_0p0V-2023-06-06T19-53-55.txt',
            'test_bias_minus_0p001V-2023-06-06T19-59-43.txt',
            'test_bias_minus_0p01V-2023-06-06T20-05-17.txt',
            'test_bias_minus_0p1V-2023-06-06T11-53-35.txt',
            'test_bias_minus_0p2V-2023-06-06T11-58-58.txt',
            'test_bias_minus_0p3V-2023-06-06T12-04-25.txt',
            'test_bias_minus_0p4V-2023-06-06T12-09-56.txt',
            'test_bias_minus_0p5V-2023-06-06T12-16-41.txt',
            'test_bias_minus_0p6V-2023-06-06T12-22-51.txt',
            'test_bias_minus_0p7V-2023-06-06T12-28-00.txt',
            'test_bias_minus_0p8V-2023-06-06T12-33-21.txt',
            'test_bias_minus_0p9V-2023-06-06T12-38-40.txt',
            'test_bias_minus_1p0V-2023-06-06T12-44-02.txt',
            'test_bias_minus_1p2V-2023-06-06T13-51-52.txt',
            'test_bias_minus_1p4V-2023-06-06T13-57-26.txt',
            'test_bias_minus_1p6V-2023-06-06T14-02-50.txt',
            'test_bias_minus_1p8V-2023-06-06T14-08-16.txt',
            'test_bias_minus_2p0V-2023-06-06T14-14-15.txt',
            'test_bias_minus_2p2V-2023-06-06T14-20-38.txt',
            'test_bias_minus_2p4V-2023-06-06T14-29-35.txt',
            'test_bias_minus_2p6V-2023-06-06T14-35-00.txt',
            'test_bias_minus_2p8V-2023-06-06T14-40-41.txt',
            'test_bias_minus_3p0V-2023-06-06T14-46-07.txt',
            'test_bias_minus_3p3V-2023-06-06T14-52-17.txt',
            'test_bias_minus_3p6V-2023-06-06T14-57-42.txt',
            'test_bias_minus_4p0V-2023-06-06T15-03-21.txt',
            'test_bias_minus_4p5V-2023-06-06T15-09-01.txt',
            'test_bias_minus_5p0V-2023-06-06T15-14-28.txt',
            'test_bias_minus_5p5V-2023-06-06T15-20-19.txt',
            'test_bias_minus_6p0V-2023-06-06T15-25-32.txt',
            'test_bias_minus_6p5V-2023-06-06T15-30-55.txt',
            'test_bias_minus_7p0V-2023-06-06T15-36-19.txt',
            'test_bias_minus_7p5V-2023-06-06T15-41-44.txt',
            'test_bias_minus_8p0V-2023-06-06T15-46-55.txt',
            'test_bias_minus_9p0V-2023-06-06T15-52-19.txt',
            'test_bias_minus_10p0V-2023-06-06T15-59-52.txt',
            'test_bias_minus_11p0V-2023-06-06T16-05-13.txt',
            'test_bias_minus_12p0V-2023-06-06T17-00-59.txt',
            'test_bias_minus_13p0V-2023-06-06T17-06-42.txt',
            'test_bias_minus_14p0V-2023-06-06T17-12-04.txt',
            'test_bias_minus_15p0V-2023-06-06T17-17-45.txt',
            'test_bias_minus_16p0V-2023-06-06T17-23-39.txt',
            'test_bias_minus_17p0V-2023-06-06T17-29-07.txt',
            'test_bias_minus_18p0V-2023-06-06T17-34-13.txt',
            'test_bias_minus_19p0V-2023-06-06T17-41-11.txt',
            'test_bias_minus_20p0V-2023-06-06T17-46-21.txt',
            'test_bias_minus_22p0V-2023-06-06T17-53-19.txt',
            'test_bias_minus_24p0V-2023-06-06T17-58-46.txt',
            'test_bias_minus_26p0V-2023-06-06T18-03-54.txt',
            'test_bias_minus_30p0V-2023-06-06T18-11-31.txt',
            'test_bias_minus_35p0V-2023-06-06T18-16-47.txt',
            'test_bias_minus_40p0V-2023-06-06T18-22-38.txt',
            'test_bias_minus_45p0V-2023-06-06T18-28-17.txt',
            'test_bias_minus_50p0V-2023-06-06T18-33-40.txt',
            'test_bias_minus_60p0V-2023-06-06T18-38-54.txt',
            'test_bias_minus_70p0V-2023-06-06T18-44-34.txt',
            'test_bias_minus_80p0V-2023-06-06T18-50-13.txt',
            'test_bias_minus_100p0V-2023-06-06T18-55-53.txt',
            'test_bias_minus_125p0V-2023-06-06T19-01-16.txt',
            'test_bias_minus_150p0V-2023-06-06T19-07-04.txt',
            'test_bias_minus_200p0V-2023-06-06T19-12-23.txt',
            'test_bias_minus_250p0V-2023-06-06T19-17-31.txt',
            'test_bias_minus_300p0V-2023-06-06T19-23-03.txt'
            ]
        elif args.Vienna_PSS_Set2:
            folder = '/home/suman/CMS/HGCal/MOSEFT/Data/TrackerPSS_Round-Set2/'
            plotdir = 'TrackerPSS_round_Set2'
            title = 'MOSFET (Tracker PSS round)'
            file_names = [
            'test_bias_minus_0p3NegV-2023-07-12T18-36-24.txt',
            'test_bias_minus_0p2NegV-2023-07-12T18-29-18.txt',
            'test_bias_minus_0p1NegV-2023-07-12T18-22-02.txt',
            'test_bias_minus_0p0V-2023-07-12T18-10-23.txt',
            'test_bias_minus_0p1V-2023-07-11T17-47-07.txt',
            'test_bias_minus_0p2V-2023-07-11T18-01-11.txt',
            'test_bias_minus_0p3V-2023-07-11T18-12-01.txt',
            'test_bias_minus_0p4V-2023-07-11T18-25-43.txt',
            'test_bias_minus_0p5V-2023-07-11T18-53-40.txt',
            'test_bias_minus_0p6V-2023-07-11T19-32-40.txt',
            'test_bias_minus_0p7V-2023-07-11T19-43-17.txt',
            'test_bias_minus_0p8V-2023-07-11T19-53-22.txt',
            'test_bias_minus_0p9V-2023-07-11T20-27-39.txt',
            'test_bias_minus_1p0V-2023-07-11T17-30-43.txt',
            'test_bias_minus_1p1V-2023-07-11T20-39-56.txt',
            'test_bias_minus_1p2V-2023-07-11T20-52-29.txt',
            'test_bias_minus_1p3V-2023-07-11T21-12-01.txt',
            'test_bias_minus_1p4V-2023-07-11T21-26-00.txt',
            'test_bias_minus_1p5V-2023-07-11T21-44-29.txt',
            'test_bias_minus_1p6V-2023-07-11T21-52-11.txt',
            'test_bias_minus_1p8V-2023-07-11T22-04-24.txt',
            'test_bias_minus_2p0V-2023-07-11T22-36-06.txt',
            'test_bias_minus_2p2V-2023-07-11T22-45-58.txt',
            'test_bias_minus_2p4V-2023-07-11T23-08-33.txt',
            'test_bias_minus_2p6V-2023-07-11T23-17-00.txt',
            'test_bias_minus_2p8V-2023-07-11T23-24-38.txt',
            'test_bias_minus_3p0V-2023-07-11T23-33-10.txt',
            'test_bias_minus_3p5V-2023-07-11T23-42-14.txt',
            'test_bias_minus_4p0V-2023-07-11T23-52-50.txt',
            'test_bias_minus_4p5V-2023-07-12T00-01-20.txt',
            'test_bias_minus_5p0V-2023-07-12T00-09-50.txt',
            'test_bias_minus_5p5V-2023-07-12T10-32-03.txt',
            'test_bias_minus_6p0V-2023-07-12T10-41-41.txt',
            'test_bias_minus_6p5V-2023-07-12T10-56-55.txt',
            'test_bias_minus_7p0V-2023-07-12T11-16-37.txt',
            'test_bias_minus_7p5V-2023-07-12T11-25-48.txt',
            'test_bias_minus_8p0V-2023-07-12T11-34-16.txt',
            'test_bias_minus_9p0V-2023-07-12T11-50-52.txt',
            'test_bias_minus_10p0V-2023-07-12T11-59-55.txt',
            'test_bias_minus_11p0V-2023-07-12T12-07-48.txt',
            'test_bias_minus_12p0V-2023-07-12T12-15-37.txt',
            'test_bias_minus_13p0V-2023-07-12T12-24-14.txt',
            'test_bias_minus_14p0V-2023-07-12T13-41-37.txt',
            'test_bias_minus_15p0V-2023-07-12T13-53-15.txt',
            'test_bias_minus_16p0V-2023-07-12T14-02-43.txt',
            'test_bias_minus_17p0V-2023-07-12T14-28-00.txt',
            'test_bias_minus_18p0V-2023-07-12T14-59-09.txt',
            'test_bias_minus_19p0V-2023-07-12T15-09-20.txt',
            'test_bias_minus_20p0V-2023-07-12T15-30-28.txt',
            'test_bias_minus_22p0V-2023-07-12T19-17-25.txt',
            'test_bias_minus_24p0V-2023-07-12T20-00-21.txt',
            'test_bias_minus_26p0V-2023-07-12T20-08-24.txt',
            'test_bias_minus_28p0V-2023-07-12T20-22-25.txt',
            'test_bias_minus_30p0V-2023-07-12T20-38-38.txt',
            'test_bias_minus_35p0V-2023-07-12T20-46-37.txt',
            'test_bias_minus_40p0V-2023-07-12T20-57-11.txt',
            'test_bias_minus_45p0V-2023-07-12T21-29-41.txt',
            'test_bias_minus_50p0V-2023-07-12T21-43-30.txt',
            'test_bias_minus_60p0V-2023-07-12T22-11-49.txt',
            'test_bias_minus_70p0V-2023-07-12T22-20-07.txt',
            'test_bias_minus_80p0V-2023-07-12T22-48-52.txt',
            'test_bias_minus_100p0V-2023-07-12T23-08-56.txt',
            'test_bias_minus_125p0V-2023-07-12T23-19-05.txt',
            'test_bias_minus_150p0V-2023-07-12T23-40-51.txt',
            'test_bias_minus_200p0V-2023-07-12T23-53-51.txt',
            'test_bias_minus_250p0V-2023-07-13T00-02-28.txt',
            'test_bias_minus_300p0V-2023-07-13T00-10-35.txt'
            ]
        elif args.Vienna_HGC_OTLL_Set2_100mV:
            folder = '/home/suman/CMS/HGCal/MOSEFT/Data/HGcal_OTLL_VSD_100mv_ELM/'
            plotdir = 'HGcal_OTLL_VSD_100mv_ELM'
            title = 'MOSFET (HGcal OTLL round)'
            file_names = [
            'test_bias_minus_0p0V-2023-07-19T13-44-18.txt',
            'test_bias_minus_0p1V-2023-07-19T13-56-36.txt',
            'test_bias_minus_0p2V-2023-07-19T14-11-41.txt',
            'test_bias_minus_0p3V-2023-07-19T14-27-19.txt',
            'test_bias_minus_0p4V-2023-07-19T15-19-00.txt',
            'test_bias_minus_0p5V-2023-07-19T15-54-23.txt',
            'test_bias_minus_0p6V-2023-07-19T16-06-48.txt',
            'test_bias_minus_0p7V-2023-07-19T16-19-12.txt',
            'test_bias_minus_0p8V-2023-07-19T16-32-31.txt',
            'test_bias_minus_0p9V-2023-07-19T16-43-13.txt',
            'test_bias_minus_1p0V-2023-07-19T16-53-45.txt',
            'test_bias_minus_1p1V-2023-07-19T17-06-07.txt',
            'test_bias_minus_1p2V-2023-07-19T18-39-23.txt',
            'test_bias_minus_1p4V-2023-07-19T18-49-47.txt',
            'test_bias_minus_1p6V-2023-07-20T11-12-12.txt',
            'test_bias_minus_1p8V-2023-07-20T11-23-00.txt',
            'test_bias_minus_2p0V-2023-07-20T11-34-27.txt',
            'test_bias_minus_2p2V-2023-07-20T11-47-36.txt',
            'test_bias_minus_2p4V-2023-07-20T12-18-34.txt',
            'test_bias_minus_2p6V-2023-07-20T12-44-28.txt',
            'test_bias_minus_2p8V-2023-07-20T12-54-58.txt',
            'test_bias_minus_3p0V-2023-07-20T14-20-51.txt',
            'test_bias_minus_3p5V-2023-07-20T14-32-35.txt',
            'test_bias_minus_4p0V-2023-07-20T14-44-33.txt',
            'test_bias_minus_4p5V-2023-07-20T14-55-55.txt',
            'test_bias_minus_5p0V-2023-07-20T15-06-15.txt',
            'test_bias_minus_5p5V-2023-07-20T15-40-32.txt',
            'test_bias_minus_6p0V-2023-07-20T15-57-04.txt',
            'test_bias_minus_6p5V-2023-07-20T16-13-31.txt',
            'test_bias_minus_7p0V-2023-07-20T19-34-50.txt',
            'test_bias_minus_7p5V-2023-07-20T23-16-33.txt',
            'test_bias_minus_8p0V-2023-07-20T23-42-24.txt',
            'test_bias_minus_9p0V-2023-07-21T00-11-35.txt',
            'test_bias_minus_10p0V-2023-07-21T00-30-43.txt',
            'test_bias_minus_12p0V-2023-07-21T11-19-51.txt',
            'test_bias_minus_13p0V-2023-07-21T11-51-00.txt',
            'test_bias_minus_14p0V-2023-07-21T12-19-08.txt',
            'test_bias_minus_15p0V-2023-07-21T12-32-26.txt',
            'test_bias_minus_16p0V-2023-07-21T12-48-57.txt',
            'test_bias_minus_17p0V-2023-07-21T14-03-46.txt',
            'test_bias_minus_18p0V-2023-07-21T14-17-32.txt',
            'test_bias_minus_19p0V-2023-07-21T14-32-48.txt',
            'test_bias_minus_20p0V-2023-07-21T14-49-49.txt',
            #'test_bias_minus_22p0V-2023-07-21T15-18-50.txt',
            'test_bias_minus_24p0V-2023-07-21T15-47-32.txt',
            'test_bias_minus_26p0V-2023-07-21T16-02-33.txt',
            #'test_bias_minus_28p0V-2023-07-21T16-22-46.txt',
            'test_bias_minus_30p0V-2023-07-21T17-51-06.txt',
            'test_bias_minus_35p0V-2023-07-21T18-30-52.txt',
            'test_bias_minus_40p0V-2023-07-21T20-55-12.txt',
            'test_bias_minus_45p0V-2023-07-21T22-25-26.txt',
            'test_bias_minus_50p0V-2023-07-21T22-55-10.txt',
            'test_bias_minus_60p0V-2023-07-21T23-36-43.txt',
            'test_bias_minus_70p0V-2023-07-22T00-35-30.txt',
            'test_bias_minus_80p0V-2023-07-31T10-06-54.txt',
            'test_bias_minus_90p0V-2023-07-31T10-40-16.txt',
            'test_bias_minus_100p0V-2023-07-31T11-04-49.txt',
            'test_bias_minus_110p0V-2023-07-31T11-26-37.txt',
            'test_bias_minus_120p0V-2023-07-31T11-45-41.txt',
            'test_bias_minus_130p0V-2023-07-31T12-00-38.txt',
            'test_bias_minus_140p0V-2023-07-31T12-16-58.txt',
            'test_bias_minus_150p0V-2023-07-31T12-45-53.txt'
            ]
        else:
            folder = '/home/suman/CMS/HGCal/MOSEFT/Data/HGcal_Common_Round/'
            plotdir = 'HGcal_round_common'
            title = 'MOSFET (HGcal round)'
            file_names = [
            'test_bias_minus_0p0V-2023-03-14T15-18-28.txt',
            'test_bias_minus_1p0V-2023-03-14T15-54-39.txt',
            'test_bias_minus_2p0V-2023-03-14T16-03-40.txt',
            'test_bias_minus_3p0V-2023-03-14T16-18-24.txt',
            'test_bias_minus_4p0V-2023-03-14T16-27-25.txt',
            'test_bias_minus_5p0V-2023-03-14T16-36-30.txt',
            'test_bias_minus_6p0V-2023-03-14T16-45-13.txt',
            'test_bias_minus_7p0V-2023-03-14T16-53-46.txt',
            'test_bias_minus_8p0V-2023-03-14T17-08-41.txt',
            'test_bias_minus_9p0V-2023-03-14T17-16-58.txt',
            'test_bias_minus_10p0V-2023-03-14T17-25-27.txt',
            'test_bias_minus_12p0V-2023-03-14T17-33-39.txt',
            'test_bias_minus_14p0V-2023-03-14T17-41-58.txt',
            'test_bias_minus_16p0V-2023-03-14T17-50-15.txt',
            'test_bias_minus_18p0V-2023-03-14T18-00-32.txt',
            'test_bias_minus_20p0V-2023-03-14T18-09-37.txt',
            'test_bias_minus_30p0V-2023-03-14T18-18-12.txt',
            'test_bias_minus_40p0V-2023-03-14T18-26-35.txt',
            'test_bias_minus_50p0V-2023-03-14T18-57-52.txt',
            'test_bias_minus_75p0V-2023-03-14T19-05-52.txt',
            'test_bias_minus_100p0V-2023-03-14T19-14-00.txt',
            'test_bias_minus_200p0V-2023-03-14T19-22-44.txt',
            'test_bias_minus_300p0V-2023-03-14T19-31-59.txt'
            ]

    VGS_max=40
    VGS_min=0
    mu0_max = 4500 #3000
    mu0_min = 500#3000
    v1by2_max = 40
    v1by2_min = 0

files = [folder + fl for fl in file_names]

plotdir = 'Figures/'+plotdir
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

sep = '/'

bias_voltages = []
bd_voltages = []

#dielectric const
Si_DC = 11.68 
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
    V_offset = 2.5
else:
    ##Vienna numbers
    rD = 150 #0.5*650  #0.5*diameter-drain
    rS = 100 #0.5*110  #0.5*diameter-source
    #rG = 0.5*360  #0.5*diameter-gate
    rD = rD*1.e-6
    rS = rS*1.e-6
    #rG = rG*1.e-6
    WbyL = 2*math.pi*1./math.log(rD/rS)
    #C_ox = 82*(1.e-12)*1./(math.pi*math.pow(rD*100,2))  #capacitance / area in cm2 100 to go from m to cm #5*1.e-9
    d_ox = 720*1.e-9*1.e+2  #thickness in nm, 1.e+2 to get in cm
    C_ox =  SiO2_DC*epsilon_0*1./d_ox
    V_ds = 50.*1.e-3
    #tuneable parameters used for fit
    V_offset = 3.0
    #software Klayout

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

for f in files:
   print(f)

   # plotting current and voltage #

   #read data
   strs = f.split('/')
   bias_voltage = ((strs[len(strs)-1][16:22]).split('-')[0]).split('_')[0]
   #print "bias_voltage",bias_voltage
   temp = re.compile("([0-9]+)([a-zA-Z]+)([0-9]+)([a-zA-Z]+)")
   res = temp.match(bias_voltage).groups()
   bias_voltage = res[0]+"."+res[2]
   bv_float = int(res[0])+0.1*int(res[2])
   if res[3].startswith("N"):
       bv_float *= -1.0
       bias_voltage = "-"+res[0]+"."+res[2]
   #bv_float = int(res[0])+0.1*int(res[2])
   bias_vol_floats.append(bv_float)

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
        plt.plot(df['voltage'], df['i_elm'], '-o', label=bias_voltage, color=colors[counter%29])
    else:
        ids_index=3
        plt.plot(df['voltage'], df['i_smu2'], '-o', label=bias_voltage, color=colors[counter%29])
   
   plt.title(title, size=fontsize)
   plt.xlabel('V$_{SG}$ [V]', size=fontsize)
   plt.ylabel('I$_{SD}$ [A]', size=fontsize)
   #for label
   bias_voltages.append(bias_voltage)
   # counter for color
   counter += 1

   # Get threshold voltage here #
   vals = df.values.tolist()
   vols, curs, curs_store = [], [], []
   for iv in range(len(vals)):
       if vals[iv][vgs_index]>=VGS_max:
           continue
       vols.append(vals[iv][vgs_index])
       curs.append(vals[iv][ids_index])
       curs_store.append(vals[iv][ids_index])

   print "bias_voltage ",bias_voltage#,type(bv_float)
   if ((bv_float)<(-0.3)):
    for iv in range(1,len(vols)):
        curs[iv] -= curs[0]
        curs_store[iv] -= curs_store[0]
    curs[0] = 0
    curs_store[0] =0

   #print vols,curs
   v_th = calculate_threshold(vols,curs)
   bd_voltages.append(v_th)
   print bias_voltage, v_th

   vols_sub, mus = [], []
   for iv in range(len(vols)):#vals)):
        vols_sub.append(vols[iv]-v_th)
        mus.append(curs_store[iv]*1./(1.e-6+vols_sub[iv]))

   vols_all.append(vols_sub)
   curs_all.append(curs_store)
   mus_all.append(mus)

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

### mu vs V_GS - V_th ###

par_product = V_ds * C_ox * WbyL
if args.verbose:
    print "par_product",par_product

mu0_vals, vp5_vals = [], []
bd_off, bd_err = [], []

for iv in range(len(bd_voltages)):
    bd_err.append(0)

#plot_params(counter,vols_all,curs_all,bd_voltages,V_offset,par_product,bias_voltages,colors,mu0_vals,vp5_vals,bd_off,bd_err,bias_vol_floats)
plot_params(counter,vols_all,curs_all,bd_voltages,V_offset,par_product,bias_vol_floats,colors,mu0_vals,vp5_vals,bd_off,bd_err)

plot_param_values(bias_vol_floats,mu0_vals,vp5_vals,plotdir,mu0_max,mu0_min,v1by2_min,v1by2_max)

#exit()

## Calculate doping depth and concentration ##

## Double derivative: d^2 V_{back} / d V_{th}^2

first_voltage_ders = []
second_voltage_ders = []
x_vals = []
px_vals = []

#change threshold by hand

for iv in range(len(bd_voltages)):
    print "bias",bias_vol_floats[iv]
    if args.verbose:
        print "before",bd_voltages[iv]
        print "offset",bd_off[iv]
    if args.isHamburg:
        bd_voltages[iv] += bd_off[iv]
    else:
        bd_voltages[iv] += bd_off[iv]
    print "after",bd_voltages[iv]

for iv in range(len(bd_voltages)-1):
    if bd_voltages[iv+1]<bd_voltages[iv]:
        print "problem at ",bias_vol_floats[iv]

if args.verbose:
    for iv in range(len(bd_voltages)):
        print bias_vol_floats[iv],bd_voltages[iv]

if args.repeated:
    exit()

# plot threshold voltage as a function of bias voltage
plot_biasvsthreshold(bias_vol_floats,bd_voltages,plotdir)

csv = CubicSpline(bd_voltages,bias_vol_floats)
vthvsbias_fit_coeff = []

if args.isHamburg and args.nopstop:
    plot_thresholdvsbias(bd_voltages,bias_vol_floats,csv,plotdir,vthvsbias_fit_coeff,0)
else:
    plot_thresholdvsbias(bd_voltages,bias_vol_floats,csv,plotdir,vthvsbias_fit_coeff,1)

bias_vol_floats_sel, bd_voltages_sel, bias_vol_floats_sel = [], [], []
bd_voltages_sel_err = []

for iv in range(len(bias_vol_floats)):
    if(bias_vol_floats[iv]>=left_vrange and bias_vol_floats[iv]<=2000):
        bias_vol_floats_sel.append(bias_vol_floats[iv])
        bd_voltages_sel.append(bd_voltages[iv])
        bd_voltages_sel_err.append(bd_err[iv]) #0.04

#if args.isVienna and args.nopstop:
bd_voltages_sel = sm.nonparametric.lowess(bd_voltages_sel, bias_vol_floats_sel, frac=0.075)[:, 1]

first_voltage_ders = bd_voltages_sel

first_voltage_ders = csv(bd_voltages_sel,1)

first_voltage_ders[0:-1] = np.diff(bias_vol_floats_sel)/np.diff(bd_voltages_sel)
first_voltage_ders[-1]   = (bias_vol_floats_sel[-1]-bias_vol_floats_sel[-2])*1./(bd_voltages_sel[-1] - bd_voltages_sel[-2])

if args.usePolFit:
    get_derivates_from_pol(bd_voltages_sel,bias_vol_floats_sel,bd_voltages_sel_err,first_voltage_ders,"first_der")

if args.useSmoothing:
    #first_voltage_ders = smoothTriangle(first_voltage_ders,3)
    #first_voltage_ders = smoothData(first_voltage_ders,2)
    #first_voltage_ders = sm.nonparametric.lowess(first_voltage_ders, bd_voltages_sel, frac=0.2)[:, 1]
    #first_voltage_ders = smooth_data_fft(first_voltage_ders,25)
    #first_voltage_ders = smooth_data_np_convolve(first_voltage_ders,5)
    #first_voltage_ders = smooth_data_np_cumsum_my_average(first_voltage_ders,5)
    #first_voltage_ders = smooth_data_kernel_regression(first_voltage_ders,50)
    #first_voltage_ders = smooth_data_savgol_0(first_voltage_ders,5)
    first_voltage_ders = smooth_data_savgol_1(first_voltage_ders,5)
    #first_voltage_ders = smooth_data_savgol_2(first_voltage_ders,5)
    #first_voltage_ders = smooth_data_convolve_my_average(first_voltage_ders,5)

for iv in range(len(bias_vol_floats_sel)):
    x_vals.append(first_voltage_ders[iv] * (epsilon / C_ox) * 1.e+4)  #1.e+4 for cm -> um

csv_1 = CubicSpline(bd_voltages_sel,first_voltage_ders)

plot_vbdvsx(bd_voltages_sel,x_vals,plotdir,csv_1(bd_voltages_sel) * (epsilon / C_ox) * 1.e+4)

second_voltage_ders = csv_1(bd_voltages_sel,1)

#second_voltage_ders = first_voltage_ders
second_voltage_ders[0:-1] =np.diff(first_voltage_ders)/np.diff(bd_voltages_sel)
#second_voltage_ders[-1] = (first_voltage_ders[-1]-first_voltage_ders[-2])*1./(bd_voltages_sel[-1]-bd_voltages_sel[-2])

#if args.useSmoothing:
#   second_voltage_ders = smooth_data_kernel_regression(second_voltage_ders,50)
#   second_voltage_ders = smoothTriangle(second_voltage_ders,3)
#   second_voltage_ders = smooth_data_np_cumsum_my_average(second_voltage_ders,1)
#   second_voltage_ders = sm.nonparametric.lowess(second_voltage_ders, bias_vol_floats_sel, frac=0.1)[:, 1]

for iv in range(len(bias_vol_floats_sel)):
    if second_voltage_ders[iv]<1.e-6:
        px_vals.append(0)
    else:
        px_vals.append((1./second_voltage_ders[iv]) * C_ox * C_ox * 1./(q0*epsilon) )  #1.e-6 for m-3 -> cm-3

for ix in range(len(first_voltage_ders)):
    print bias_vol_floats_sel[ix],bd_voltages_sel[ix],x_vals[ix],px_vals[ix]

# remove first and last two points (to avoid effects of end points in calculation of derivatives)
npoints_toremove = 0
if args.isVienna:
    if args.nopstop:
        npoints_toremove = 1
    else:
        npoints_toremove = 2
if args.isVienna:# and not args.nopstop:
    for iter in range(npoints_toremove):
        bias_vol_floats_sel = bias_vol_floats_sel[:-1]
        x_vals = x_vals[:-1]
        px_vals = px_vals[:-1]
    for iter in range(npoints_toremove):
        bias_vol_floats_sel = bias_vol_floats_sel[1:]
        x_vals = x_vals[1:]
        px_vals = px_vals[1:]

plot_doping(bias_vol_floats_sel,x_vals,px_vals,plotdir)
