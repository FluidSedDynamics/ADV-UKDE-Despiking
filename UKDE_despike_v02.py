# Author: Sam Kraus 6/6/2024

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import projections
import matplotlib.patheffects as pe
import matplotlib.patches as patches
import pandas as pd
import os
import glob
from scipy import signal
from sklearn.neighbors import KernelDensity
import plotly.graph_objects as go
import plotly.offline as ofl

def load_prep(fs, raw_folder, datadir_ascii, cnames):
    ## Find the data files and prep for loading the files 
    datadir = datadir_ascii
    ext = '*.dat'     # data file extension

    os.chdir(raw_folder)
    
    # Find all of the data files and store the names 
    files = sorted(glob.glob(datadir+ext)) # sorted by timestamp
    nf=len(files)
    os.chdir(datadir)
    filenames = sorted(glob.glob(ext)) # sorted by timestamp
    for i_KS in range(0,nf):
        filenames[i_KS]=os.path.splitext(os.path.basename(filenames[i_KS]))[0] #strips extension
    os.chdir(raw_folder)

    #strip the endings and find the profiles, save date timestamp
    date_arr=filenames.copy()
    profile=filenames.copy()
    fs_arr=np.zeros(len(filenames), dtype=float)
    for i_KS in range(0,len(filenames)):
        stname = filenames[i_KS] 
        profile[i_KS]=stname[:-14]
        date_arr[i_KS]=stname[-14:-10]+'-'+stname[-10:-8]+'-'+stname[-8:-6]
        
        # find fs if ascii .hdr file
        txtfile = open(datadir+'/'+filenames[i_KS]+'.hdr', 'r')
        lines = txtfile.readlines()
        if any('Sampling rate' in line for line in lines):
            for i_line in lines:
                if 'Sampling rate' in i_line:
                    fs = float(i_line[38:40])
                    print('fs found in .hdr, fs = ', fs, ' Hz')
                else:
                    pass
        else:
            fs = fs
            print('Could not find sampling frequency in .hdr file.')
        fs_arr[i_KS] = fs

    # add in an index for the profiles
    datapt=filenames.copy()
    pp = np.unique(profile) # find the unique profiles
    j_KS=0
    k_KS=1
    for i_KS in range(0,len(filenames)):
        if profile[i_KS]==pp[j_KS]:
            datapt[i_KS]=profile[i_KS]+'_V'+str(k_KS)
            k_KS=k_KS+1
        else:
            k_KS = 1
            datapt[i_KS]=profile[i_KS]+'_V'+str(k_KS)
            k_KS=k_KS+1
            j_KS = j_KS+1

    #Print the file names and index value for each file
    print('This station has ', nf, 'files: \n')

    for i_KS in range(0,nf):
        print('file:',filenames[i_KS],', data point:', datapt[i_KS], '(index: ',i_KS,')')
        
    return nf, files, filenames, datapt, fs_arr, date_arr

# Qn
def calc_Qn(data):
    n = len(data)
    h = (n/2)+1
    k = round((h*(h-1))/2)
    
    interpt_dist_matrix = np.abs(np.subtract.outer(data,data))
    upper_indices = np.triu_indices(interpt_dist_matrix.shape[0], k=1)
    interpt_dist_arr = interpt_dist_matrix[upper_indices]
    dist_sorted = sorted(interpt_dist_arr)
    kth = dist_sorted[k-1]

    if len(data) % 2 == 0:
        cQn = n/(n+3.8)
    else:
        cQn = n/(n+1.4)
    
    return cQn*2.2219*kth

# Silverman (1986) rule-of-thumb (ROT)
def calc_h_ROT_stdev(data):
    n = len(data)
    std = np.std(data, ddof=1)
    return (4/3)**(1/5)*std*n**(-1/5)

# Silverman (1986) rule-of-thumb (ROT) w/ Qn replacing st dev
def calc_h_ROT_Qn(data):
    n = len(data)
    Qn = calc_Qn(data)
    if Qn == 0.0:
        Qn = np.std(data, ddof=1)
    else:
        pass
    return (4/3)**(1/5)*Qn*n**(-1/5)

def thresholds_UKDE(u1,R_cut,S_cut,hu_METHOD):
    # rescale velocities to range from zero to 1
    us = (u1-min(u1))/(max(u1)-min(u1))
    
    # set the bandwidth and bin number
    if hu_METHOD == 'Chen_ures':
        vel_arr = list(set(us)) # convert to set then back again to delete duplicates
        vel_sort = sorted(vel_arr)
        vel_diff = np.diff(vel_sort)
        ures = min(np.abs(vel_diff))
        k = k_chen
        hu = k*ures
    elif hu_METHOD == 'IslamZhu':
        hu = 0.01
    elif hu_METHOD == 'ROT_stdev':
        hu = calc_h_ROT_stdev(us)
    elif hu_METHOD == 'ROT_Qn':
        hu = calc_h_ROT_Qn(us)
    else:
        hu = 0.01 # Islam and Zhu (2013)

    nb = 256 # just set a value; 256 recommended by Islam and Zhu

    n = len(u1)

    # Step 1: compute the kernel density of u utilizing a Gaussian kernel
    u1 = np.asarray(u1, dtype=float)

    # create the array to build the KDE
    X = us[:, np.newaxis] # raw data to be evaluated in kernel
    u = np.linspace(min(us), max(us), nb) # locations in the x to calculate density
    uf = np.linspace(min(u1), max(u1), nb) # to translate back to og vel values

    # calculate the KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=hu).fit(X)
    log_dens = kde.score_samples(u[:, np.newaxis])
    f_hat = np.exp(log_dens)
    
    # Steps 2 and 3: locate foothills of density profile based on slope of density profile
    # use Islam and Zhu's slope cutoff threshold

    imax = np.argmax(f_hat) # find the peak value's index of the kde
    f_hat_max = f_hat[imax]

    # find the upper end of valid points-------------------------------------------------------------
    i = imax+1

    while i < len(f_hat):
        Su = nb*np.abs(f_hat[i+1] - f_hat[i])/f_hat_max # Islam and Zhu (2013) slope definition
        if i < len(f_hat)-2:
            Su_next = nb*(f_hat[i+2] - f_hat[i+1])/f_hat_max # check for gradient reversal
        else:
            Su_next = 0.0
        if f_hat[i]/f_hat_max <= R_cut and Su <= S_cut: 
            iup = i
            break;
        elif f_hat[i]/f_hat_max <= R_cut and Su_next > 0:
            iup = i
            break;
        i = i+1
        if i == len(f_hat)-1:
            iup = i
            break;

    # find the lower end of valid points--------------------------------------------------------------
    i = imax-1

    while i > 0:
        Su = nb*np.abs(f_hat[i] - f_hat[i-1])/f_hat_max # Islam and Zhu (2013) slope definition
        if i > 2:
            Su_next = nb*(f_hat[i-2] - f_hat[i-1])/f_hat_max # check for gradient reversal
        else:
            Su_next = 0.0
        if f_hat[i]/f_hat_max <= R_cut and Su <= S_cut: 
            idown = i
            break;
        elif f_hat[i]/f_hat_max <= R_cut and Su_next > 0:
            idown = i
            break;
        i = i-1
        if i == 0:
            idown = i
            break;

    # set the max and min values for filtering -------------------------------------------------------
    umin = uf[idown]
    umax = uf[iup]
        
    return n, uf, f_hat, umin, umax, hu

def spikeID_UKDE(u1, n, umin, umax):
    # Step 4a: identify spikes
    # for loop from Schnellbach (2022)
    F = np.zeros(n)
    for i in range(n): # sets values of zero-array to 1 if condition for spike is met
        if u1[i] > umax or u1[i] < umin:
            F[i] = 1
        else:
            F[i] = 0.0

    Id_tuple = np.where(F>0) # find flagged values indices
    Id = np.asarray(Id_tuple[0])
    
    return Id

def replace_spikes_side3D(ui, vi, wi, Id_li, ADV_FACE, data):
    # prep u, v, and w unfiltered as list copies to be indexed and written over
    u_out = ui.tolist()
    u_out = pd.DataFrame(u_out, columns=['u_out'])
    v_out = vi.tolist()
    v_out = pd.DataFrame(v_out, columns=['v_out'])
    w_out = wi.tolist()
    w_out = pd.DataFrame(w_out, columns=['w_out'])
    
    # Check if first and/or last data points are spikes, replace by mean, delete Id from list
    n = len(data)
    
    u_nospikes = u_out.drop(Id_li[[0][0]])
    v_nospikes = v_out.drop(Id_li[[1][0]])
    w_nospikes = w_out.drop(Id_li[[2][0]])

    if ADV_FACE == 'side':
        if len(Id_li[[0][0]]) == 0 and len(Id_li[[1][0]]) == 0:
            # check if spikes identified in either u or v, if so proceed - update to Schnellbach
            pass
        else: 
            if len(Id_li[[0][0]]) != 0 and Id_li[0][0] == 0:                              # u first
                u_out.iloc[0] = np.mean(u_nospikes)
                Id_li[0] = np.delete(Id_li[0],0)
            if len(Id_li[[0][0]]) != 0 and Id_li[0][len(Id_li[0])-1] == (n-1):            # u last
                u_out.iloc[n-1] = np.mean(u_nospikes)
                Id_li[0] = np.delete(Id_li[0],-1)
            if len(Id_li[[1][0]]) != 0 and Id_li[1][0] == 0:                              # v first
                v_out.iloc[0] = np.mean(v_nospikes)
                Id_li[1] = np.delete(Id_li[1],0)
            if len(Id_li[[1][0]]) != 0 and Id_li[1][len(Id_li[1])-1] == (n-1):            # v last
                v_out.iloc[n-1] = np.mean(v_nospikes)
                Id_li[1] = np.delete(Id_li[1],-1)

            # Replaces in u and v (Beams 1 and 2)
            for j in [0,1]:
                for i in Id_li[[j][0]]: # goes through each spike index
                    lgv_i = i-1 #last good value
                    ngv_i = i+1 #next good value
                    while lgv_i in Id_li[[j][0]]: #checks if ID of lgv is a spike
                        lgv_i -= 1
                    while ngv_i in Id_li[[j][0]]: #checks if ID of ngv is a spike
                        ngv_i += 1
                    step_u = (u_out.iloc[ngv_i] - u_out.iloc[lgv_i])/(ngv_i-lgv_i)
                    step_v = (v_out.iloc[ngv_i] - v_out.iloc[lgv_i])/(ngv_i-lgv_i)
                    u_out.iloc[i] = u_out.iloc[lgv_i] + (i-lgv_i)*step_u
                    v_out.iloc[i] = v_out.iloc[lgv_i] + (i-lgv_i)*step_v

        if len(Id_li[[2][0]]) == 0:
            # check if spikes identified in w, if so proceed - update to Schnellbach
            pass 
        else:
            if Id_li[2][0] == 0:                              # w first
                w_out.iloc[0] = np.mean(w_nospikes)
                Id_li[2] = np.delete(Id_li[2],0)
            if Id_li[2][len(Id_li[2])-1] == (n-1):            # w last
                w_out.iloc[n-1] = np.mean(w_nospikes)
                Id_li[2] = np.delete(Id_li[2],-1)

            # Replaces in w (Beams 3 and 4)
            for i in Id_li[[2][0]]: # goes through each spike index
                lgv_i = i-1 #last good value
                ngv_i = i+1 #next good value
                while lgv_i in Id_li[[2][0]]: #checks if ID of lgv is a spike
                    lgv_i -= 1
                while ngv_i in Id_li[[2][0]]: #checks if ID of ngv is a spike
                    ngv_i += 1
                step = (w_out.iloc[ngv_i] - w_out.iloc[lgv_i])/(ngv_i-lgv_i)
                w_out.iloc[i] = w_out.iloc[lgv_i] + (i-lgv_i)*step
        
    elif ADV_FACE == 'down':
        if len(Id_li[[0][0]]) == 0 and len(Id_li[[2][0]]) == 0:
            # check if spikes identified in either u or v, if so proceed - update to Schnellbach
            pass
        else: 
            if len(Id_li[[0][0]]) != 0 and Id_li[0][0] == 0:                              # u first
                u_out.iloc[0] = np.mean(u_nospikes)
                Id_li[0] = np.delete(Id_li[0],0)
            if len(Id_li[[0][0]]) != 0 and Id_li[0][len(Id_li[0])-1] == (n-1):            # u last
                u_out.iloc[n-1] = np.mean(u_nospikes)
                Id_li[0] = np.delete(Id_li[0],-1)
            if len(Id_li[[2][0]]) != 0 and Id_li[2][0] == 0:                              # w first
                v_out.iloc[0] = np.mean(v_nospikes)
                Id_li[2] = np.delete(Id_li[2],0)
            if len(Id_li[[2][0]]) != 0 and Id_li[2][len(Id_li[2])-1] == (n-1):            # w last
                v_out.iloc[n-1] = np.mean(v_nospikes)
                Id_li[2] = np.delete(Id_li[2],-1)

            # Replaces in u and w (Beams 1 and 2)
            for j in [0,2]:
                for i in Id_li[[j][0]]: # goes through each spike index
                    lgv_i = i-1 #last good value
                    ngv_i = i+1 #next good value
                    while lgv_i in Id_li[[j][0]]: #checks if ID of lgv is a spike
                        lgv_i -= 1
                    while ngv_i in Id_li[[j][0]]: #checks if ID of ngv is a spike
                        ngv_i += 1
                    step_u = (u_out.iloc[ngv_i] - u_out.iloc[lgv_i])/(ngv_i-lgv_i)
                    step_v = (v_out.iloc[ngv_i] - v_out.iloc[lgv_i])/(ngv_i-lgv_i)
                    u_out.iloc[i] = u_out.iloc[lgv_i] + (i-lgv_i)*step_u
                    v_out.iloc[i] = v_out.iloc[lgv_i] + (i-lgv_i)*step_v

        if len(Id_li[[1][0]]) == 0:
            # check if spikes identified in w, if so proceed - update to Schnellbach
            pass 
        else:
            if Id_li[1][0] == 0:                              # v first
                w_out.iloc[0] = np.mean(w_nospikes)
                Id_li[1] = np.delete(Id_li[1],0)
            if Id_li[1][len(Id_li[1])-1] == (n-1):            # v last
                w_out.iloc[n-1] = np.mean(w_nospikes)
                Id_li[1] = np.delete(Id_li[1],-1)

            # Replaces in w (Beams 3 and 4)
            for i in Id_li[[1][0]]: # goes through each spike index
                lgv_i = i-1 #last good value
                ngv_i = i+1 #next good value
                while lgv_i in Id_li[[1][0]]: #checks if ID of lgv is a spike
                    lgv_i -= 1
                while ngv_i in Id_li[[1][0]]: #checks if ID of ngv is a spike
                    ngv_i += 1
                step = (w_out.iloc[ngv_i] - w_out.iloc[lgv_i])/(ngv_i-lgv_i)
                w_out.iloc[i] = w_out.iloc[lgv_i] + (i-lgv_i)*step

    else:
        print('Need to define orientation of ADV variable, \'ADV_FACE\'. Two options: (1) enter \'side\' for side-facing 4-beam like Nortek Vectrino or (2) enter \'down\' for downward-facing 4-beam like Nortek Vectrino')
    
    return u_out, v_out, w_out

def replace_spikes_1D(ui, Id_li, ADV_FACE, data):
    # prep u as list copies to be indexed and written over
    u_out = ui.tolist()
    u_out = pd.DataFrame(u_out, columns=['u_out'])
    
    # Check if first and/or last data points are spikes, replace by mean, delete Id from list
    n = len(data)
    
    u_nospikes = u_out.drop(Id_li[[0][0]])
    
    if ADV_FACE == 'singular and separate':
        if len(Id_li[[0][0]]) == 0:
            # check if spikes identified in u, if so proceed - update to Schnellbach
            pass
        else: 
            if len(Id_li[[0][0]]) != 0 and Id_li[0][0] == 0:                              # u first
                u_out.iloc[0] = np.mean(u_nospikes)
                Id_li[0] = np.delete(Id_li[0],0)
            if len(Id_li[[0][0]]) != 0 and Id_li[0][len(Id_li[0])-1] == (n-1):            # u last
                u_out.iloc[n-1] = np.mean(u_nospikes)
                Id_li[0] = np.delete(Id_li[0],-1)

            # Replaces in u
            j = 0
            for i in Id_li[[j][0]]: # goes through each spike index
                lgv_i = i-1 #last good value
                ngv_i = i+1 #next good value
                while lgv_i in Id_li[[j][0]]: #checks if ID of lgv is a spike
                    lgv_i -= 1
                while ngv_i in Id_li[[j][0]]: #checks if ID of ngv is a spike
                    ngv_i += 1
                step_u = (u_out.iloc[ngv_i] - u_out.iloc[lgv_i])/(ngv_i-lgv_i)
                u_out.iloc[i] = u_out.iloc[lgv_i] + (i-lgv_i)*step_u

    else:
        print('Need to define orientation of ADV variable, \'ADV_FACE\'.')
        
    return u_out

def plot_KDEfilters(data, u1, uf, f_hat, umin, umax, time_cname,
                    plotpad, plottitle, velname, fig_folder, datapt,
                    SHOW_PLOTS, SAVE_PLOTS, TITLE_PLOTS):
#     # add in an option if top-two-out index files TTO_i are specified - example code:
#     if TITLE_PLOTS == 'yes':
#         if file in TTO_i:
#             plt.title(plottitle+', h = {}'.format(np.round(hu, 5))+'  TOP TWO OUT')
#         else:
#             plt.title(plottitle+', h = {}'.format(np.round(hu, 5)))
#     else:
#         pass
        
    # set color
    if velname == 'u':
        velcolor = 'tab:blue'
    elif velname == 'v':
        velcolor = 'tab:red'
    elif velname == 'w':
        velcolor = 'tab:green'
    else:
        print('Error identifying velcolor for plots.')
        
    # plot KDE filters
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(uf,f_hat, color=velcolor, label='KDE ${}$'.format(velname))
    plt.axvline(x = umin, color = 'r',  alpha=0.5,linestyle = '--')
    plt.axvline(x = umax, color = 'r',  alpha=0.5,linestyle = '--')
    plt.xlabel('${}$ [m/s]'.format(velname))
    plt.ylabel('$\hat{}({})$'.format('{f}',velname))
    plt.legend(loc='best', ncol=1)
    if TITLE_PLOTS == 'yes':
        plt.title(plottitle)
    else:
        pass
    if SAVE_PLOTS == 'yes':
        plt.savefig(fig_folder+'/'+datapt+'_filters_timeseries_{}_comparison.pdf'.format(velname),
                bbox_inches='tight', pad_inches=plotpad)
    else:
        pass
    if SHOW_PLOTS == 'yes':
        plt.show()
    else:
        pass
    plt.close()

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(data[time_cname], u1, color=velcolor, alpha=0.75, label='{}'.format(velname))
    plt.axhline(y = umin, color = 'r',  alpha=0.5, linestyle = '--')
    plt.axhline(y = umax, color = 'r',  alpha=0.5, linestyle = '--')
    ax.set_xlabel('$t$ [s]')
    ax.set_ylabel('${}$ [m/s]'.format(velname))
    plt.legend(loc='best', ncol=1)
    if TITLE_PLOTS == 'yes':
        plt.title(plottitle)
    else:
        pass
    if SAVE_PLOTS == 'yes':
        plt.savefig(fig_folder+'/'+datapt+'_filters_timeseries_{}_comparison.pdf'.format(velname),
                    bbox_inches='tight', pad_inches=plotpad)
    else:
        pass
    if SHOW_PLOTS == 'yes':
        plt.show()
    else:
        pass
    plt.close()

def plot_hist_timeseries(data, u1, u_out_plot, time_cname,
                    plotpad, plottitle, velname, fig_folder, datapt, 
                    SHOW_PLOTS, SAVE_PLOTS, TITLE_PLOTS):
    
    # set color
    if velname == 'u':
        velcolor = 'tab:blue'
        rawcolor = 'tab:cyan'
    elif velname == 'v':
        velcolor = 'tab:red'
        rawcolor = 'tab:orange'
    elif velname == 'w':
        velcolor = 'tab:green'
        rawcolor = 'tab:olive'
    else:
        print('Error identifying velcolor for plots.')
    
    # save histograms and time-series to Figures folder
    ###################################################
    # plot the time series filtered and unfiltered
    fig, ax = plt.subplots(figsize=(6,2))
    ax.plot(data[time_cname], u1, alpha=0.7, label='unfiltered', linewidth=0.7, color=rawcolor)
    ax.plot(data[time_cname], u_out_plot, alpha=0.8, label='filtered', linewidth=0.7, color=velcolor)
    ax.set_xlabel('$t$ [s]')
    ax.set_ylabel('${}$ [m/s]'.format(velname))
    plt.legend(loc=1, ncol=1)
    if 'TITLE_PLOTS' == 'yes':
        plt.title(plottitle)
    else:
        pass
    if SAVE_PLOTS == 'yes':
        plt.savefig(fig_folder+'/'+datapt+'_timeseries_{}_comparison.pdf'.format(velname),
                bbox_inches='tight', pad_inches=plotpad)
    else:
        pass
    if SHOW_PLOTS == 'yes':
        plt.show()
    else:
        pass
    plt.close()

    # plot the histogram filtered and unfiltered
    fig, ax = plt.subplots(figsize=(4,3))
    plt.hist(u1, 30, density=True, alpha=0.7, label='unfiltered', color=rawcolor)
    plt.hist(u_out_plot, 30, density=True, alpha=0.7, label='filtered', color=velcolor)
    plt.xlabel('${}$ [m/s]'.format(velname))
    plt.ylabel('Probability')
    plt.legend()
    if 'TITLE_PLOTS' == 'yes':
        plt.title(plottitle)
    else:
        pass
    if SAVE_PLOTS == 'yes':
        plt.savefig(fig_folder+'/'+datapt+'_hist_{}_comparison.pdf'.format(velname),
                bbox_inches='tight', pad_inches=plotpad)
    else:
        pass
    if SHOW_PLOTS == 'yes':
        plt.show()
    else:
        pass
    plt.close()

def plot_PSD(data, u1, u_out_plot, fs, 
             plotpad, plottitle, velname, fig_folder, datapt, 
             SHOW_PLOTS, SAVE_PLOTS, TITLE_PLOTS):
    
    # set color
    if velname == 'u':
        velcolor = 'tab:blue'
        rawcolor = 'tab:cyan'
    elif velname == 'v':
        velcolor = 'tab:red'
        rawcolor = 'tab:orange'
    elif velname == 'w':
        velcolor = 'tab:green'
        rawcolor = 'tab:olive'
    else:
        print('Error identifying velcolor for plots.')
    
    # PSD using matplotlib's psd - seems to result in spike for y-intercept?
#     psd_u_raw, freq_u_raw = psd(u1, NFFT=min(len(u1), 256), Fs=fs)
#     psd_u_filt, freq_u_filt = psd(u_out_plot, NFFT=min(len(u_out_plot), 256), Fs=fs)
    
    # PSD using scipy's signal.welch - seems to avoid matplotlib's spike for y-intercept
    freq_u_raw, psd_u_raw = signal.welch(u1, fs=fs, nperseg = min(len(u1), 256))
    freq_u_filt, psd_u_filt = signal.welch(u_out_plot, fs=fs, nperseg = min(len(u_out_plot),256))

    # x and y plot limits based on data
    # xlim can't be 0 for log plot, so just manually put in
    xlim_min = 0.09
#     xlim_min = min(min(freq_u_raw), min(freq_u_filt), PSD_xlim_min)
    xlim_max = fs/2*1.1 # max based on Nyquist frequency
    ylim_min = min(min(psd_u_raw), min(psd_u_filt))*0.1
    ylim_max = max(max(psd_u_raw), max(psd_u_filt))*10
    # x and y plot limits manually set
#     xlim_min = 0.09
#     xlim_max = fs/2*1.1
#     ylim_min = 1e-11
#     ylim_max = 1e2
    x_range = np.array([xlim_min, xlim_max])
  
    # plot power spectra density for velocity time series, filtered and unfiltered
    fig, ax = plt.subplots(figsize=(5,5))
    # slope comparison lines
    slope_ys = np.logspace(start=np.log10(ylim_min*0.1), stop=np.log10(ylim_max*100), num=15, base=10)
    for s_y in range(len(slope_ys)):
        if s_y == 0:
            ax.plot(x_range, slope_ys[s_y]*x_range**(-5/3), 'lightgray', label='-5/3 slope', linestyle='dashed')
        else:
            ax.plot(x_range, slope_ys[s_y]*x_range**(-5/3), 'lightgray', linestyle='dashed')
    # plot PSDs
    # plot unfiltered PSD
    ax.loglog(freq_u_raw, psd_u_raw, alpha=0.7, color=rawcolor, label='original ${}$'.format(velname), 
            linewidth=0.7)
    # plot filtered PSD
    ax.loglog(freq_u_filt, psd_u_filt, alpha=0.7, color=velcolor, label='filtered ${}$'.format(velname),
             linewidth=0.7)

    # plot settings
    ax.grid(False)
    plt.xlim([xlim_min, xlim_max])
    plt.ylim([ylim_min, ylim_max])
    plt.xlabel('$Frequency$ $[Hz]$')
    plt.ylabel('$E(f)$ $[m^2/s^2/Hz]$')
    plt.legend(loc='lower left', ncol=1)
    if 'TITLE_PLOTS' == 'yes':
        plt.title(plottitle)
    else:
        pass
    if SAVE_PLOTS == 'yes':
        plt.savefig(fig_folder+'/'+datapt+'_PSD_{}.pdf'.format(velname),
                bbox_inches='tight', pad_inches=plotpad)
    else:
        pass
    if SHOW_PLOTS == 'yes':
        plt.show()
    else:
        pass
    plt.close()

def despike_UKDE_xyz(data, fs, R_cut, S_cut, hu_METHOD, ADV_FACE, time_cname,
                 plotpad, plottitle, fig_folder, datapt, file,
                 SHOW_PLOTS, SAVE_PLOTS, TITLE_PLOTS):
    # takes in and processes 3D data, x-vel=data.v1_m_s, y-vel=data.v2_m_s, z-vel=data.v3_m_s

    # calculate the means
    uavg = data.v1_m_s.mean()
    vavg = data.v2_m_s.mean()
    wavg = data.v3_m_s.mean()

    # calculate the standard deviations
    u_stdev = np.std(data.v1_m_s)
    v_stdev = np.std(data.v2_m_s)
    w_stdev = np.std(data.v3_m_s)

    # Filter with UKDE-Hybrid Method
    ui = np.array(data.v1_m_s)
    vi = np.array(data.v2_m_s)
    wi = np.array(data.v3_m_s)

    vels = [ui,vi,wi]
    velnames = ['u','v','w']
    Id_li = []
    thresholds_li = []

    for vel in range(len(vels)):
        velname = velnames[vel]
        print('velname = ', velname)
        u1 = vels[vel]

        n, uf, f_hat, umin, umax, hu = thresholds_UKDE(u1,R_cut,S_cut,hu_METHOD)

        thresholds_li.append(np.array([umin,umax]))
        
        Id = spikeID_UKDE(u1, n, umin, umax)
        
        Id_li.append(Id) # make list of spike indices
        
        # plots
        plot_KDEfilters(data, u1, uf, f_hat, umin, umax, time_cname,
                    plotpad, plottitle, velname, fig_folder, datapt,
                    SHOW_PLOTS, SAVE_PLOTS, TITLE_PLOTS)

    
    # Step 4b: eliminate outliers detected in horizontal (u or v) or vertical (w)
    # uses Schnellbach (2022) code partially
    u_out, v_out, w_out = replace_spikes_side3D(ui, vi, wi, Id_li, ADV_FACE, data)
    # make output dataframes for velocities into nested array/list
    vels_out = [u_out.values.flatten(),v_out.values.flatten(),w_out.values.flatten()]
    
    dataf = data.copy().drop(['v1_m_s', 'v2_m_s','v3_m_s'],axis=1)
    
    # plots
    for vel in range(len(vels)):
        velname = velnames[vel]
        u1 = vels[vel]
        u_out_plot = vels_out[vel]
        
        plot_hist_timeseries(data, u1, u_out_plot, time_cname,
                    plotpad, plottitle, velname, fig_folder, datapt, 
                    SHOW_PLOTS, SAVE_PLOTS, TITLE_PLOTS)
        
        plot_PSD(data, u1, u_out_plot, fs, 
             plotpad, plottitle, velname, fig_folder, datapt, 
             SHOW_PLOTS, SAVE_PLOTS, TITLE_PLOTS)
    
    # split dataf and insert filtered u, v, and w velocities
    output = pd.concat([u_out, v_out, w_out], axis=1)
    output = output.rename(columns={'u_out':'v1_m_s', 'v_out':'v2_m_s', 'w_out':'v3_m_s'})
    left_part = dataf.iloc[:, :1]
    right_part = dataf.iloc[:, 1:]
    dataf = pd.concat([left_part, output, right_part], axis=1)
    
    # calculate the means
    uavg_f = dataf.v1_m_s.mean()
    vavg_f = dataf.v2_m_s.mean()
    wavg_f = dataf.v3_m_s.mean()
    
    # calculate the standard deviations
    u_stdev_f = np.std(dataf.v1_m_s)
    v_stdev_f = np.std(dataf.v2_m_s)
    w_stdev_f = np.std(dataf.v3_m_s)
        
    return dataf, Id_li, vels

def despike_UKDE_1D(data, fs, R_cut, S_cut, time_cname,
                 plotpad, plottitle, fig_folder, datapt, hu_METHOD, ADV_FACE,
                 SHOW_PLOTS, SAVE_PLOTS, TITLE_PLOTS):
    
    # calculate the means
    uavg = data[vel_cname].mean()
    t_clm = np.array(data[time_cname])

    # Filter with UKDE-Hybrid method
    ui = np.array(data.uvel)
    
    velname = 'u'
    
    Id_li = []
    thresholds_li = []

    u1 = ui

    n, uf, f_hat, umin, umax, hu = thresholds_UKDE(u1,hu_METHOD,R_cut,S_cut)

    thresholds_li.append(np.array([umin,umax]))

    Id = spikeID_UKDE(u1, n, umin, umax)

    Id_li.append(Id) # make list of spike indices

    # plots
    # pause timer while plotting
    timer.pause()
    
    filters(data, u1, uf, f_hat, umin, umax, time_cname,
                    plotpad, plottitle, velname, fig_folder, datapt,
                    SHOW_PLOTS, SAVE_PLOTS, TITLE_PLOTS)
    timer.resume()

    
    # Step 4b: eliminate outliers detected
    # uses Schnellbach (2022) partially
    u_out = replace_spikes_1D(ui,Id_li,ADV_FACE,data)
    u_out = u_out.values.flatten() # turn into array
    
    # plots
    velname = velname
    u1 = u1
    u_out_plot = u_out

    plot_hist_timeseries(data, u1, u_out_plot, time_cname,
                    plotpad, plottitle, velname, fig_folder, datapt, 
                    SHOW_PLOTS, SAVE_PLOTS, TITLE_PLOTS)

    plot_PSD(data, u1, u_out_plot, fs, 
             plotpad, plottitle, velname, fig_folder, datapt, 
             SHOW_PLOTS, SAVE_PLOTS, TITLE_PLOTS)
        
    return u_out, Id_li, ui
