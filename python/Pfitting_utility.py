'''
Author: Rodrigo Angulo (rangulo1@jhu.edu)
Utility python script for fitting flux profile of infrared echo found with JWST.

'''

import sys, os
import glob, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from jwst.datamodels import ImageModel

from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import Regions, PixCoord, PolygonSkyRegion, PolygonPixelRegion, CircleSkyRegion, CirclePixelRegion, RectangleSkyRegion, RectanglePixelRegion

from scipy.interpolate import interp1d
import scipy.optimize as opt

import emcee
import corner
import dynesty
from dynesty import plotting as dyplot

from scipy.stats import linregress
from scipy.stats import chi2 as sc_chi2

from scipy.ndimage import gaussian_filter1d, minimum_filter1d, percentile_filter

from celerite2 import GaussianProcess, terms


# For calcualting apparent motion

def str8line(x,m,b):
    """
    Simple definition of a straight line
    """
    y = m*x+b
    return(y)


def get_appmotion(t, y_as, y_err):
    """
    Measure apparent motion of light echo across epochs
    """
    line_fit = opt.curve_fit(str8line, t, y_as, sigma=y_err, absolute_sigma=True)
    
    lnfi = line_fit[0]
    lnfi_err = np.sqrt(np.diag(line_fit[1]))
    
    
    fig, ax = plt.subplots(dpi = 100, figsize = [6,4])
    ax.minorticks_on()
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(which='major',bottom='on',top='on',left='on',right='on',length=12)
    ax.tick_params(which='minor',bottom='on',top='on',left='on',right='on',length=5)
    ax.set_xlabel('MJD', fontsize = 14)
    ax.set_ylabel('Arcsecs', fontsize = 14)
    ax.ticklabel_format(axis='both', style='plain')
        
    ax.errorbar(t,y_as, yerr=y_err, fmt='o', color='royalblue')
    ax.axline((0,lnfi[1]),slope=lnfi[0], color='darkblue', label=f'm={np.round(lnfi[0],3)} arcsec/day')
    
    ax.set_xlim(min(t)-5, max(t)+5)
    ax.set_ylim(min(y_as)-0.2, max(y_as)+0.2)
    
    ax.legend(fontsize=14)

    return([lnfi, lnfi_err], fig, ax)


def get_hwhm(x, y, plot=False):
    """
    Measure hwhm of curve
    """
    max_ind = np.where(y==max(y))[0][0]
    max_x = x[max_ind]
    max_y = y[max_ind]
    hm_y = max_y/2.
    
    tr1 = x[:max_ind]
    m_fs1 = y[:max_ind]
    tr2 = x[max_ind:]
    m_fs2 = y[max_ind:]
    
    inters_f1 = m_fs1 - hm_y
    inters_f2 = m_fs2 - hm_y
    int_inds1 = np.argsort(abs(inters_f1))[:2]
    int_inds2 = np.argsort(abs(inters_f2))[:2]
    
    hwhm1 = (hm_y - m_fs1[int_inds1[0]])/(m_fs1[int_inds1[1]] - m_fs1[int_inds1[0]]) * (tr1[[int_inds1[1]]] - tr1[[int_inds1[0]]]) + tr1[int_inds1[0]]
    hwhm2 = (hm_y - m_fs2[int_inds2[0]])/(m_fs2[int_inds2[1]] - m_fs2[int_inds2[0]]) * (tr2[[int_inds2[1]]] - tr2[[int_inds2[0]]]) + tr2[int_inds2[0]]

    trise = abs(max_x-hwhm1[0])
    tfall = abs(max_x-hwhm2[0])

    if plot is True:
        plt.subplots(figsize=(8, 4))
        plt.scatter(x, y, alpha=0.1) 
        plt.scatter(max_x, max_y, color='red')
        plt.axvline(max_x)
        y_line = [hm_y]*len(x)
        plt.plot(x, y_line, color='red', ls='--')
        plt.axvline(hwhm1, color='red')
        plt.axvline(hwhm2, color='red')
        # plt.xlim(-2,2)
        plt.xlabel("t")
        plt.ylabel("Flux")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return(trise, tfall)
    

## For peak fitting

def optimized_GP(df, bkgsub=True):
    """
    Calculate Gaussian processed best-fit curve -- can be helpful for visualization (not necessary for fitting)
    """

    if bkgsub is True:
        yax = 'flux_bkgsub'
    elif bkgsub is False:
        yax = 'flux'
        
    x, y, yerr = df['xax_arcsec'], df[yax], df['flux_err']

    def build_gp(params):
        sigma, rho = np.exp(params)
        kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=0.5)
        gp = GaussianProcess(kernel)
        gp.compute(x, yerr=yerr)
        return(gp)

    def neg_log_like(params):
        gp = build_gp(params)
        return(-gp.log_likelihood(y))

    p0 = np.log([np.std(y), 500])
    bounds0 = [(np.log(1e-3), np.log(5*np.std(y))), (np.log(10), np.log(1000))]
    
    result = opt.minimize(neg_log_like, p0, method="L-BFGS-B", bounds=bounds0)
    gpr = build_gp(result.x)
    
    x_range = np.linspace(min(x), max(x), 500)
    mu, var = gpr.predict(y, x_range, return_var=True)
    std = np.sqrt(var)

    gp_r = pd.DataFrame({'xax_arcsec':x_range, yax: mu, 'flux_err':std})
    
    return(gp_r)


def range4_bckgrnd(df, window=100, plot=False):
    """
    Defining range for background subtraction -- finds flat part of curve
    """

    # x, y, yerr = df['arcsec'], df['Flux'], df['Flux_err']
    x, y, yerr = df['xax_arcsec'], df['flux'], df['flux_err']
    
    slopes = []
    chi2s = []
    
    for i in range(len(x)-window):
        result = linregress(x[i:i+window],y[i:i+window])
    
        str8_line = result.slope * x[i:i+window] + result.intercept
        resid = y[i:i+window] - str8_line
        chi2 = resid**2/yerr[i:i+window]**2
    
        slopes.append(result.slope)
        chi2s.append(np.sum(chi2))
    
    slopes=np.array(slopes)
    chi2s=np.array(chi2s)

    m_inds = np.where(abs(slopes) < 1e-3)[0]
    c_ind = m_inds[np.where(chi2s[m_inds] == min(chi2s[m_inds]))[0][0]]

    if plot is True:
        plt.scatter(x, y)
        plt.scatter(x[c_ind: c_ind+window], y[c_ind: c_ind+window], color='red')
        plt.show()

    return(c_ind, c_ind+window)


def background_rolling_percentile(x, flux, window_size=20, percentile=10, smooth=True):
    """
    Estimate background using rolling percentile.
    Good for data with peaks above a smooth background.
    """
    # Sort by x to ensure proper ordering
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    flux_sorted = flux[sort_idx]
    
    # Rolling percentile (low percentile captures background between peaks)
    background = percentile_filter(flux_sorted, percentile=percentile, size=window_size)
    
    # Optional: smooth the result
    if smooth is True:
        background = gaussian_filter1d(background, sigma=window_size/4)
    
    # Restore original order
    restore_idx = np.argsort(sort_idx)
    background = background[restore_idx]
    
    return background


def input_model(t, t0, hwhm_rise, hwhm_delta):
    """
    Define input model for fitting process -- piecewise function defined by rising exponential and a decaying exponential
    """

    sigma_r = hwhm_rise / np.sqrt(2*np.log(2))
    hwhm_fall = hwhm_rise + hwhm_delta
    sigma_f = hwhm_fall / np.log(2)
    
    model = np.zeros_like(t)
    model[t < t0] = 1.0 * np.exp(-1*(t[t<t0] - t0)**2/(2*sigma_r**2))
    model[t >= t0] = 1.0 * np.exp(-1*(t[t>=t0] - t0)/sigma_f)

    return(model)


def optimal_linear_params(conv_model, data, errors):
    """
    Analytically solve for optimal normalization and offset -- faster solution for emcee/dynesty runs
    """
    # Solving: data = norm * model + offset
    
    w = 1.0 / errors**2
    
    # Design matrix approach
    S = np.sum(w)
    Sm = np.sum(w * conv_model)
    Smm = np.sum(w * conv_model**2)
    Sd = np.sum(w * data)
    Smd = np.sum(w * conv_model * data)
    
    det = S * Smm - Sm**2
    
    if abs(det) < 1e-10:
        return 1.0, 0.0  # fallback
    
    norm = (S * Smd - Sm * Sd) / det
    offset = (Smm * Sd - Sm * Smd) / det
    
    return norm, offset
    


def emcee_fit(df, hwhm_rise, hwhm_delta, fit4nuisance=False):
    """
    Emcee fitting of peaks with defined input rise and fall times
    Parameters: sigma - of Gaussian, t0 - peak location
        if hwhm's not specified: hwhm_rise - rise time, hwhm_delta - diff of fall and rise times (> 0 -- longer fall time than rise)
        if fitting for nuisance parameters: A - normalization, B - offset; otherwise is linearly solved (saves processing time)
    """

    fit_inds = np.where(df['gix_mask'].eq(True))[0]
    fit_ts = df.loc[fit_inds,'xax_days'].values
    fit_fs = df.loc[fit_inds,'flux_bkgsub'].values
    fit_ferrs = df.loc[fit_inds,'flux_err'].values

    trange = np.linspace(min(fit_ts), max(fit_ts), 10*(len(fit_ts)-1)+1)   # make model have finer grid space
    dt = np.average(np.diff(trange))

    if hwhm_rise is not None and hwhm_delta is not None:
        if fit4nuisance is False:
            # sigma, t0 = params
            initial = np.array([5.0, 0.0])
            params_type = 'a'               # arbitrary labeling for setting up emcee run for proper parameter space
        elif fit4nuisance is True:
            # sigma, t0, A, B = params
            initial = np.array([5.0, 0.0, 1.0, 0.0])
            params_type = 'b'
    elif hwhm_rise is None and hwhm_delta is None:
        if fit4nuisance is False:
            # sigma, t0, hwhm_r, hwhm_d = params
            initial = np.array([5.0, 0.0, 0.1, 1.0])
            params_type = 'c'
        elif fit4nuisance is True:
            # sigma, t0, hwhm_r, hwhm_d, A, B = params
            initial = np.array([5.0, 0.0, 0.1, 1.0, 1.0, 0.0])
            params_type = 'd'
    
    def log_like(params):
        
        if params_type == 'a':
            sigma, t0 = params
            model = input_model(trange, t0, hwhm_rise, hwhm_delta)
            
        elif params_type == 'b':
            sigma, t0, A, B = params
            model = input_model(trange, t0, hwhm_rise, hwhm_delta)
            
        elif params_type == 'c':
            sigma, t0, hwhm_r, hwhm_d = params
            model = input_model(trange, t0, hwhm_r, hwhm_d)
            
        elif params_type == 'd':
            sigma, t0, hwhm_r, hwhm_d, A, B = params
            model = input_model(trange, t0, hwhm_r, hwhm_d)
        
        
        conv_model = gaussian_filter1d(model, sigma/dt)
        conv_model = np.interp(fit_ts, trange, conv_model)

        if fit4nuisance is False:
            A, B = optimal_linear_params(conv_model, fit_fs, fit_ferrs)
        
        model_f = A*conv_model+B
        chi2 = np.sum(((fit_fs-model_f)/fit_ferrs)**2)
        return(-0.5*(chi2+np.sum(np.log(2*np.pi*fit_ferrs**2))))
    
    def log_prior(params):
        
        if params_type == 'a':
            sigma, t0 = params
            if (1e-3 < sigma < 20.0) and (-10 < t0 < 10):
                return(0.0)
            else:
                return(-np.inf)
                
        elif params_type == 'b':
            sigma, t0, A, B = params
            if (1e-3 < sigma < 20.0) and (-10 < t0 < 10) and (1e-3 < A < 50.0) and (-1 < B < 1):
                return(0.0)
            else:
                return(-np.inf)
                
        elif params_type == 'c':
            sigma, t0, hwhm_r, hwhm_d = params
            if (1e-3 < sigma < 20.0) and (-10 < t0 < 10) and (1e-3 < hwhm_r < 5.0) and (0.0 < hwhm_d < 10.0):
                return(0.0)
            else:
                return(-np.inf)
                
        elif params_type == 'd':
            sigma, t0, hwhm_r, hwhm_d, A, B = params
            if (1e-3 < sigma < 20.0) and (-10 < t0 < 10) and (1e-3 < hwhm_r < 5.0) and (0.0 < hwhm_d < 10.0) and (1e-3 < A < 50.0) and (-1 < B < 1):
                return(0.0)
            else:
                return(-np.inf)
    
    def log_probability(params):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return(-np.inf)
        else:
            return(lp + log_like(params))

    ndim = len(initial)
    nwalkers = 32
    
    pos = (initial + 1e-4 * np.random.randn(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability)
    
    sampler.run_mcmc(pos, 5000, progress=True)

    samples = sampler.get_chain(discard=1000, thin=10, flat=True)

    return(samples)


def emcee_results(samples, df, hwhm_rise, hwhm_delta, fit4nuisance=False, percentile=False):

    params_16 = np.percentile(samples, 16, axis=0)
    params_50 = np.percentile(samples, 50, axis=0)
    params_84 = np.percentile(samples, 84, axis=0)

    param_tiles = []
    param_results = []
    
    for i in range(len(samples[0])):
        ptiles_i = [params_16[i], params_50[i], params_84[i]]
        pres_i = [ptiles_i[1], ptiles_i[2]-ptiles_i[1], ptiles_i[1]-ptiles_i[0]]
        param_tiles.append(ptiles_i)
        param_results.append(pres_i)

    fit_inds = np.where(df['gix_mask'].eq(True))[0]
    fit_ts = df.loc[fit_inds,'xax_days'].values
    fit_fs = df.loc[fit_inds,'flux_bkgsub'].values
    fit_ferrs = df.loc[fit_inds,'flux_err'].values
    
    trange = np.linspace(min(fit_ts), max(fit_ts), 10*(len(fit_ts)-1)+1)
    dt = np.average(np.diff(trange))

    if hwhm_rise is None and hwhm_delta is None:
        hwhm_rise = param_results[2][0]
        hwhm_delta = param_results[3][0]

    model = input_model(trange, param_results[1][0], hwhm_rise, hwhm_delta)
    conv_model = gaussian_filter1d(model, param_results[0][0]/dt)
    conv_model = np.interp(fit_ts, trange, conv_model)

    if fit4nuisance is False:
        A, B = optimal_linear_params(conv_model, fit_fs, fit_ferrs)
    elif fit4nuisance is True:
        A = param_results[-2][0]
        B = param_results[-1][0]
        
    model_f = A*conv_model+B
    
    residuals = fit_fs - model_f
    chi2 = np.sum((residuals / fit_ferrs)**2)
    dof = len(fit_ts) - len(samples[0])
    chi2red = chi2 / dof

    
    if percentile is True:
        return(param_tiles, param_results, chi2red, residuals)
    else:
        return(param_results, chi2red, residuals)
    

def plot_emcee(samples, df, hwhm_rise, hwhm_delta, fit4nuisance=False, figure=None, color_q='blue'):

    param_tiles, param_results, chi2red, resids = emcee_results(samples=samples, df=df, hwhm_rise=hwhm_rise, hwhm_delta=hwhm_delta, percentile=True, fit4nuisance=fit4nuisance)

    fit_inds = np.where(df['gix_mask'].eq(True))[0]
    fit_ts = df.loc[fit_inds,'xax_days'].values
    fit_fs = df.loc[fit_inds,'flux_bkgsub'].values
    fit_ferrs = df.loc[fit_inds,'flux_err'].values

    trange = np.linspace(min(fit_ts), max(fit_ts), 10*(len(fit_ts)-1)+1)
    dt = np.average(np.diff(trange))

    if hwhm_rise is None and hwhm_delta is None:
        hwhm_rise = param_results[2][0]
        hwhm_delta = param_results[3][0]

    ## For plotting
    lcmodel_50 = input_model(trange, param_results[1][0], hwhm_rise, hwhm_delta)

    convmod_50 = gaussian_filter1d(lcmodel_50, param_results[0][0]/dt)

    if fit4nuisance is False:
        A, B = optimal_linear_params(np.interp(fit_ts, trange, convmod_50), fit_fs, fit_ferrs)
    elif fit4nuisance is True:
        A = param_results[-2][0]
        B = param_results[-1][0]
        
    convmod_50 = A*convmod_50+B

    print(f"sigma: {param_results[0][0]:.2f} +{param_results[0][1]:.2f} -{param_results[0][2]:.2f}")
    print(f"t0: {param_results[1][0]:.2f} +{param_results[1][1]:.2f} -{param_results[1][2]:.2f}")
    print(f"A: {A},   B:{B}")
    print(f'Ave resids: {np.average(resids)}')
    
    ## Plot
    if figure is None:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axs = axs.flatten()
        color1='blue'
        color2='red'
    elif figure is not None:
        fig, axs = figure
        color1 = color_q
        color2 = color_q
        
    
    ## Top panel: Data and model
    axs[0].errorbar(fit_ts, fit_fs, yerr=fit_ferrs, fmt="o", color="gray", ms=3, alpha=0.8, capsize=2, zorder=0)#, label=f"Data - {df['filename'][0].partition('.')[0]}")

    axs[0].plot(trange, lcmodel_50, color=color1, linestyle="--", linewidth=1., label=f"Input ($t_f$={hwhm_rise+hwhm_delta:.1f})", zorder=2)    
    axs[0].plot(trange, convmod_50, color=color2, linestyle="-", linewidth=1., label=f"Convolved ($\\chi^2$={chi2red:.1f})", zorder=2)
    
    axs[0].axhline(0, color="k", linestyle="--", alpha=0.3)

    axs[0].set_title(f"Rise time of {hwhm_rise:.2f} days")
    axs[0].set_ylabel("Background-Subtracted Flux", fontsize=12)
    axs[0].legend(fontsize=10, ncol=2)
    axs[0].set_ylim(-0.1, 1.5)
    axs[0].grid(alpha=0.3)

    ## Bottom panel: Residuals
    axs[1].errorbar(fit_ts, resids, yerr=None, fmt="o", color=color2, ms=3, alpha=0.7, capsize=2)
    axs[1].axhline(0, color="k", linestyle="--", alpha=0.5)
    axs[1].set_xlabel('Days', fontsize=12)
    axs[1].set_ylabel("Residuals", fontsize=12)
    axs[1].grid(alpha=0.3)
    fig.subplots_adjust(wspace=0, hspace=0)

    if figure is None:
        plt.show()
    else:
        return(fig, axs)



def dynesty_fit(df, fit4nuisance=False):
    """
    Dynamic nested sampling for fitting LE peak
    Parameters: hwhm rise time, delta of rise and fall times, sigma of Gaussian, and peak location
    Nuiscance parameters: normalization and offset
    """

    fit_inds = np.where(df['gix_mask'].eq(True))[0]
    fit_ts = df.loc[fit_inds,'xax_days'].values
    fit_fs = df.loc[fit_inds,'flux_bkgsub'].values
    fit_ferrs = df.loc[fit_inds,'flux_err'].values

    trange = np.linspace(min(fit_ts), max(fit_ts), 10*(len(fit_ts)-1)+1)   # make model have finer grid space
    dt = np.average(np.diff(trange))

    log_norm_const = np.sum(np.log(2*np.pi*fit_ferrs**2))

    if fit4nuisance is False:
        initial = np.array([0.1, 1.0, 8.0, 0.0])
        bounds_lower = [1e-3, 0.0, 1e-3, -10]
        bounds_upper = [5, 10, 20.0, 10.0]
    elif fit4nuisance is True:
        initial = np.array([0.1, 1.0, 8.0, 0.0, 1.0, 0.0])
        bounds_lower = [1e-3, 0.0, 1e-3, -10, 1e-3, -1.0]
        bounds_upper = [5, 10, 20.0, 10.0, 50.0, 1.0]

    def log_likelihood(params):
        if fit4nuisance is False:
            hwhm_trise, hwhm_delta, sigma, t0 = params
        elif fit4nuisance is True:
            hwhm_trise, hwhm_delta, sigma, t0, A, B = params

        model = input_model(trange, t0, hwhm_trise, hwhm_delta)
        conv_model = gaussian_filter1d(model, sigma/dt)
        conv_model = np.interp(fit_ts, trange, conv_model)

        if fit4nuisance is False:
            A, B = optimal_linear_params(conv_model, fit_fs, fit_ferrs)
        
        model_f = A * conv_model + B
        chi2 = np.sum(((fit_fs-model_f)/fit_ferrs)**2)
        
        return(-0.5*(chi2+log_norm_const))

    def prior_transform(u):
        lower = np.asarray(bounds_lower)
        upper = np.asarray(bounds_upper)
        params = lower + u * (upper - lower)
        return(params)

    ndim = len(initial)
    sampler_dynesty = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim=ndim, sample='rwalk')

    sampler_dynesty.run_nested()

    results_dynesty = sampler_dynesty.results

    return(results_dynesty)


def dynesty_results(res, df, percentile=False, fit4nuisance=False):

    weights = np.exp(res.logwt - res.logz[-1])
    weights /= weights.sum()
    samples = dynesty.utils.resample_equal(res.samples, weights)

    params_16 = np.percentile(samples, 16, axis=0)
    params_50 = np.percentile(samples, 50, axis=0)
    params_84 = np.percentile(samples, 84, axis=0)

    param_tiles = []
    param_results = []
    
    for i in range(len(samples[0])):
        ptiles_i = [params_16[i], params_50[i], params_84[i]]
        pres_i = [ptiles_i[1], ptiles_i[2]-ptiles_i[1], ptiles_i[1]-ptiles_i[0]]
        param_tiles.append(ptiles_i)
        param_results.append(pres_i)

    fit_inds = np.where(df['gix_mask'].eq(True))[0]
    fit_ts = df.loc[fit_inds,'xax_days'].values
    fit_fs = df.loc[fit_inds,'flux_bkgsub'].values
    fit_ferrs = df.loc[fit_inds,'flux_err'].values
    trange = np.linspace(min(fit_ts), max(fit_ts), 10*(len(fit_ts)-1)+1)
    dt = np.average(np.diff(trange))
    model = input_model(trange, param_results[3][0], param_results[0][0], param_results[1][0])
    conv_model = gaussian_filter1d(model, param_results[2][0]/dt)
    conv_model = np.interp(fit_ts, trange, conv_model)

    if fit4nuisance is False:
        A, B = optimal_linear_params(conv_model, fit_fs, fit_ferrs)
    elif fit4nuisance is True:
        A = param_results[5][0]
        B = param_results[6][0]

    convmod_f = A*conv_model+B

    residuals = fit_fs - convmod_f
    chi2 = np.sum((residuals/fit_ferrs)**2)
    dof = len(fit_ts) - len(samples[0])
    chi2red = chi2 / dof

    if percentile is True:
        return(param_tiles, param_results, chi2red, residuals)
    else:
        return(param_results, chi2red, residuals)


def plot_dynesty(res, df, fit4nuisance=False):

    param_tiles, param_results, chi2red, resids = dynesty_results(res, df, percentile=True, fit4nuisance=fit4nuisance)

    fit_inds = np.where(df['gix_mask'].eq(True))[0]
    fit_ts = df.loc[fit_inds,'xax_days'].values
    fit_fs = df.loc[fit_inds,'flux_bkgsub'].values
    fit_ferrs = df.loc[fit_inds,'flux_err'].values

    trange = np.linspace(min(fit_ts), max(fit_ts), 10*(len(fit_ts)-1)+1)
    dt = np.average(np.diff(trange))

    # For plotting
    # lcmodel_16 = input_model(trange, param_tiles[3][0], param_tiles[0][0], param_tiles[1][0])
    lcmodel_50 = input_model(trange, param_tiles[3][1], param_tiles[0][1], param_tiles[1][1])
    # lcmodel_84 = input_model(trange, param_tiles[3][2], param_tiles[0][2], param_tiles[1][2])

    # convmod_16 = gaussian_filter1d(lcmodel_16, param_tiles[2][0]/dt)
    # convmod_16 = np.interp(fit_ts, trange, convmod_16)
    # A, B = optimal_linear_params(convmod_16, fit_fs, fit_ferrs)
    # convmod_16 = A*convmod_16+B

    convmod_50 = gaussian_filter1d(lcmodel_50, param_tiles[2][1]/dt)
    convmod_50 = np.interp(fit_ts, trange, convmod_50)

    if fit4nuisance is False:
        A, B = optimal_linear_params(convmod_50, fit_fs, fit_ferrs)
    elif fit4nuisance is True:
        A = param_results[5][0]
        B = param_results[6][0]
        
    convmod_50 = A*convmod_50+B

    # convmod_84 = gaussian_filter1d(lcmodel_84, param_tiles[2][2]/dt)
    # convmod_84 = np.interp(fit_ts, trange, convmod_84)
    # A, B = optimal_linear_params(convmod_84, fit_fs, fit_ferrs)
    # convmod_84 = A*convmod_84+B

    print(f"HWHM trise: {param_results[0][0]:.2f} +{param_results[0][1]:.2f} -{param_results[0][2]:.2f}")
    print(f"HWHM tfall: {param_results[0][0]+param_results[1][0]:.2f} +{param_results[1][1]:.2f} -{param_results[1][2]:.2f}")
    print(f"chi2: {chi2red}")
    
    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    
    # Top panel: Data and model
    ax1 = axs[0]
    ax1.errorbar(fit_ts, fit_fs, yerr=fit_ferrs, fmt="o", color="gray", ms=3, alpha=0.5, capsize=2, zorder=0, label=f"Data")
    
    ax1.plot(fit_ts, convmod_50, "r-", linewidth=1.5, label=f"Convolved ($\\chi^2$={chi2red:.1f})", zorder=2)
    # ax1.fill_between(fit_ts, convmod_16, convmod_84, color="r", alpha=0.3, label="16th-84th percentile", zorder=1)

    ax1.plot(trange, lcmodel_50, "b--", linewidth=1.5, label=f"Input ($t_r$={param_results[0][0]:.2f}, $t_f$={param_results[0][0]+param_results[1][0]:.2f})", zorder=2)
    # ax1.fill_between(trange, lcmodel_16, lcmodel_84, color="b", alpha=0.2, label="16th-84th percentile (no conv)", zorder=1)
    
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)

    # radius = int(4.0 * sigma_qs[1]/dt + 0.5)
    # x_rad = np.arange(-radius, radius + 1)
    # Gauss_kern = A_results[0] * np.exp(-0.5 * (x_rad / sigma_qs[1]/dt)**2) + B_results[0]
    # Gauss_kern /= Gauss_kern.sum()
    # ax1.plot(x_rad, Gauss_kern, "c--", linewidth=1.5, label=f"Gauss Kernel ($\sigma$= {sigma_results[0]:.2f} days)")
    
    ax1.set_ylabel("Flux", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    # ax1.set_ylim(-0.05, 1.1)
    ax1.set_xlabel('Days', fontsize=12)

    ## Bottom panel: Residuals
    ax2 = axs[1]
    ax2.errorbar(fit_ts, resids, yerr=None, fmt="o", ms=3, alpha=0.7, capsize=2)
    ax2.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel('Days', fontsize=12)
    ax2.set_ylabel("Residuals", fontsize=12)
    ax2.grid(alpha=0.3)
    print(np.average(resids))
    fig.subplots_adjust(wspace=0, hspace=0)
    
    return(fig, axs)



def simul_emcee(df_ls):
    """
    Simultaneous fit multiple peaks across epochs
    """
    # global params = hwhm_rise, hwhm_delta, 
    # nuiscance params = A_i, B_i, sigma_i, t0_i
    initial = [0.2, 0.5, 8.0, 0.0]

    def log_like(params):
        hwhm_rise, hwhm_delta, sigma, t0 = params
        all_chi2s = []
        for i, df in enumerate(df_ls):            
            fit_inds = np.where(df['gix_mask'].eq(True))[0]
            fit_ts = df.loc[fit_inds,'xax_days'].values
            fit_fs = df.loc[fit_inds,'flux_bkgsub'].values
            fit_ferrs = df.loc[fit_inds,'flux_err'].values
        
            trange = np.linspace(min(fit_ts), max(fit_ts), 10*(len(fit_ts)-1)+1)
            dt = np.average(np.diff(trange))

            model = input_model(trange, t0, hwhm_rise, hwhm_delta)
            conv_model = gaussian_filter1d(model, sigma/dt)
            conv_model = np.interp(fit_ts, trange, conv_model)
            A, B = optimal_linear_params(conv_model, fit_fs, fit_ferrs)
            model_f = A*conv_model+B
            chi2 = np.sum(((fit_fs-model_f)/fit_ferrs)**2)
            all_chi2s.append(chi2+np.sum(np.log(2*np.pi*fit_ferrs**2)))
        return(-0.5*np.sum(all_chi2s))
    
    def log_prior(params):
        hwhm_rise, hwhm_delta, sigma, t0 = params
        
        if (1e-3 < hwhm_rise < 5.0) and (0.0 < hwhm_delta < 10.0) and (1e-3 < sigma < 20.0) and (-10 < t0 < 10):
            return(0.0)
        else:
            return(-np.inf)
    
    def log_probability(params):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return(-np.inf)
        else:
            return(lp + log_like(params))

    ndim = len(initial)
    nwalkers = 32
    
    pos = (initial + 1e-2 * np.random.randn(nwalkers, ndim))
    
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability)
    
    sampler.run_mcmc(pos, 5000, progress=True)

    samples = sampler.get_chain(discard=1000, thin=10, flat=True)

    return(samples)


def simul_results(samples, df_ls, percentile=False):
    
    trise_samps = samples[:,0]
    hdelta_samps = samples[:,1]
    sigma_samps = samples[:,2]
    t0_samps = samples[:,3]

    trise_qs = np.percentile(trise_samps, [16, 50, 84])
    hdelta_qs = np.percentile(hdelta_samps, [16, 50, 84])
    sigma_qs = np.percentile(sigma_samps, [16, 50, 84])
    t0_qs = np.percentile(t0_samps, [16, 50, 84])
    
    trise_results = [trise_qs[1], trise_qs[2]-trise_qs[1], trise_qs[1]-trise_qs[0]]
    hdelta_results = [hdelta_qs[1], hdelta_qs[2]-hdelta_qs[1], hdelta_qs[1]-hdelta_qs[0]]
    sigma_results = [sigma_qs[1], sigma_qs[2]-sigma_qs[1], sigma_qs[1]-sigma_qs[0]]
    t0_results = [t0_qs[1], t0_qs[2]-t0_qs[1], t0_qs[1]-t0_qs[0]]
    
    all_chi2s = []
    all_nps = []
    for i, df in enumerate(df_ls):
        fit_inds = np.where(df['gix_mask'].eq(True))[0]
        fit_ts = df.loc[fit_inds,'xax_days'].values
        fit_fs = df.loc[fit_inds,'flux_bkgsub'].values
        fit_ferrs = df.loc[fit_inds,'flux_err'].values

        trange = np.linspace(min(fit_ts), max(fit_ts), 10*(len(fit_ts)-1)+1)
        dt = np.average(np.diff(trange))
        
        model = input_model(trange, t0_results[0], trise_results[0], hdelta_results[0])
        conv_model = gaussian_filter1d(model, sigma_results[0]/dt)
        conv_model = np.interp(fit_ts, trange, conv_model)
        A, B = optimal_linear_params(conv_model, fit_fs, fit_ferrs)
        model_f = A*conv_model+B
        chi2 = np.sum(((fit_fs-model_f)/fit_ferrs)**2)
        
        dof = len(fit_inds) - len(samples[0]) - 2
        chi2red = chi2 / dof
        
        all_chi2s.append(chi2red)
        all_nps.append([A,B])

    if percentile is True:
        return(trise_qs, hdelta_qs, sigma_qs, t0_qs, trise_results, hdelta_results, sigma_results, t0_results, all_nps, all_chi2s)
    else:
        return(trise_results, hdelta_results, sigma_results, t0_results, all_nps, all_chi2s)    


def plot_simul(samples, df_ls):

    trise_qs, hdelta_qs, sigma_qs, t0_qs, trise_res, hdelta_res, sigma_res, t0_res, nuisc_ps, chi2s = simul_results(samples=samples, df_ls=df_ls, percentile=True)

    for i, df in enumerate(df_ls):
        
        fit_inds = np.where(df['gix_mask'].eq(True))[0]
        fit_ts = df.loc[fit_inds,'xax_days'].values
        fit_fs = df.loc[fit_inds,'flux_bkgsub'].values
        fit_ferrs = df.loc[fit_inds,'flux_err'].values
    
        trange = np.linspace(min(fit_ts), max(fit_ts), 10*(len(fit_ts)-1)+1)
        dt = np.average(np.diff(trange))

        A, B = nuisc_ps[i]
        
        # For plotting
        # lcmodel_16 = input_model(trange, t0_qs[0], trise_qs[0], hdelta_qs[0])
        lcmodel_50 = input_model(trange, t0_qs[1], trise_qs[1], hdelta_qs[1])
        # lcmodel_84 = input_model(trange, t0_qs[2], trise_qs[2], hdelta_qs[2])

        convmod_50 = gaussian_filter1d(lcmodel_50, sigma_res[0]/dt)
        convmod_50 = A*convmod_50+B

        i_image = df['filename'][0].partition('.')[0]
        print(f"Fit of peak in {i_image}")
        print(f"trise: {trise_res[0]:.2f} +{trise_res[1]:.2f} -{trise_res[2]:.2f}")
        print(f"hdelta: {hdelta_res[0]:.2f} +{hdelta_res[1]:.2f} -{hdelta_res[2]:.2f}")
        print(f"A: {A:.2f}")
        print(f"B: {B:.2f}")
        print(f"sigma: {sigma_res[0]:.2f} +{sigma_res[1]:.2f} -{sigma_res[2]:.2f}")
        print(f"t0: {t0_res[0]:.2f} +{t0_res[1]:.2f} -{t0_res[2]:.2f}")
        
        # Plot
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        
        # Top panel: Data and model
        ax1.errorbar(fit_ts, fit_fs, yerr=fit_ferrs, fmt="o", color="gray", ms=3, alpha=0.5, capsize=2, zorder=0, label=f"Data - {i_image}")
        
        ax1.plot(trange, convmod_50, "r-", linewidth=2, label=f"Convolved model ($\\chi^2$={chi2s[i]:.2f})", zorder=2)
        # ax1.fill_between(trange, convmod_16, convmod_84, color="r", alpha=0.3, label="16th-84th percentile", zorder=1)
        
        ax1.plot(trange, lcmodel_50, "b--", linewidth=1.5, label=f"Input model ($hwhm_r$={trise_res[0]:.2f}, $hwhm_f$={trise_res[0]+hdelta_res[0]:.2f})", zorder=2)
        # ax1.fill_between(trange, lcmodel_16, lcmodel_84, color="b", alpha=0.2, label="16th-84th percentile (no conv)", zorder=1)
        
        ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
    
        # radius = int(4.0 * sigma_qs[1]/dt + 0.5)
        # x_rad = np.arange(-radius, radius + 1)
        # Gauss_kern = A_results[0] * np.exp(-0.5 * (x_rad / sigma_qs[1]/dt)**2) + B_results[0]
        # Gauss_kern /= Gauss_kern.sum()
        # ax1.plot(x_rad, Gauss_kern, "c--", linewidth=1.5, label=f"Gauss Kernel ($\sigma$= {sigma_results[0]:.2f} days)")
        
        ax1.set_ylabel("Background-Subtracted Flux", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(-0.05, 1.1)
        ax1.set_xlabel('Days', fontsize=12)
        
        plt.show()
        


### For meauring quality of fits

def calc_pvalchi2(chi2s, dof, t_falls, ptype='upper'):
    """
    Calculate p-value of chi^2
    """
    
    p_vals = []

    if ptype == 'upper':
        for val in chi2s:
            pval = sc_chi2.sf(val*dof, dof)
            p_vals.append(pval)
    elif ptype == 'lower':
        for val in chi2s:
            pval = sc_chi2.cdf(val*dof, dof)
            p_vals.append(pval)
    elif ptype == 'two':
        for val in chi2s:
            pval = 2 * min(p_upper, p_lower)
            p_vals.append(pval)

    return(p_vals)


def find_root(func, val_targ, x_range):
    """
    Find root using bracketing method (guaranteed to converge if root exists in bracket).
    """
    # Function to find root of
    func_r = lambda x: func(x) - val_targ
    
    # brentq requires the function to have opposite signs at bracket endpoints
    try:
        root = opt.brentq(func_r, x_range[0], x_range[1])
        return root
    except ValueError as e:
        print(f"No root in range {x_range}: {e}")
        return None