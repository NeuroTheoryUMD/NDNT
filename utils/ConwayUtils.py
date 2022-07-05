import numpy as np
import matplotlib.pyplot as plt
#import scipy.io as sio
from copy import deepcopy
#import NDNT.utils.NDNutils as NDNutils
import NDNT.utils.DanUtils as DU



######### UTILITIES TO DEAL WITH EYE TRACKING SIGNALS #########
def trial_offset_calc( EtraceHR, trN, to_plot=False ):
    """Detect timing (monitor refresh) offset on given trial for EyeScan 1 kHz sampling of 120 Hz signal"""
    dt = 1000/120
    dets = np.sum(abs(np.diff( EtraceHR[trN*4000+np.arange(4000,dtype=np.int64),:], axis=0)),axis=1)
    phase_hist = np.zeros(9)
    for ii in range(479):
        tstart = int(np.round(ii*dt))
        phase_hist += dets[range(tstart,tstart+9)]
    #plt.plot(dets[:200])
    offset = ((np.argmax(phase_hist)+4)%8)-4
    if to_plot:
        plt.plot(phase_hist)
        plt.show()
    return offset


def slow_drift_correct( ETin, tau=120000, to_plot=True):
    """Correct for slow drift in offset of eye traces by subtracting an extremely smoothed mean signal"""
    T, nsig = ETin.shape
    meansig = np.zeros(ETin.shape)
    #for tt in range(tau//2, T-tau//2):
    #    meansig[tt,:] = np.mean(ETin[range(tt-tau//2, tt+tau//2), :],axis=0)
    smkern = np.ones(tau) / tau
    offsets = np.mean(ETin,axis=0)
    for ii in range(nsig):
        print( "Mean-subtraction on dim %d: offset = %0.1f"%(ii, offsets[ii]) )
        meansig[:,ii] = np.convolve(ETin[:,ii], smkern, mode='same')
    
    # Accept average over edges (that 'same' padding does incorrectly): simply average over smaller windows
    cumsumfront, cumsumback = deepcopy(ETin[0, :]),  deepcopy(ETin[-1, :])
    for tt in range(tau//2):  
        meansig[tt] = cumsumfront / (2*tt+1)
        meansig[-tt] = cumsumback / (2*tt+1)
        cumsumfront += ETin[2*tt+1, :] + ETin[2*tt+2, :]
        cumsumback += ETin[-2*tt-2, :] + ETin[-2*tt-3, :]
        # Note that this is incredibly slow:  did cumulative sum instead, but this was the same
        #meansig[tt, :] = np.mean(ETin[:2*tt+1, :], axis=0)
        #meansig[-tt, :] = np.mean(ETin[-2*tt-1:, :], axis=0)

    if to_plot:
        DU.ss(nsig,1)
        for ii in range(nsig):
            plt.subplot(nsig,1,ii+1)
            plt.plot(np.arange(T)/1000, ETin[:,ii],'c')
            plt.plot(np.arange(T)/1000, meansig[:,ii],'k')
            plt.ylim([-60+offsets[ii],60+offsets[ii]])
            plt.xlim([0,T/1000])

    return deepcopy(ETin)-meansig
# END slow_drift_correct()


def blink_process( ETin, blink_pad=50, verbose=True, blink_thresh=40):
    """Zero out eye traces where blink is detected, and refurn new eye trace and blink locations
    Works for EyeScan and Eyelink, although default set for EyeLink"""
    ETout = deepcopy(ETin)
    
    a = np.where((abs(ETin[:,0]) > blink_thresh) | (abs(ETin[:,1]) > blink_thresh))[0]
    b = np.where(np.diff(a) > 10)[0]
    b = np.append(b, len(a)-1)
    blinks = []
    tstart = 0
    for ii in range(len(b)):
        blink_start = np.maximum(a[tstart]-blink_pad, 0)
        blink_end = np.minimum(a[b[ii]]+blink_pad, ETin.shape[0])
        arng = np.arange( blink_start, blink_end )
        blinks.append([blink_start, blink_end])
        #avs = np.mean(ETraw[arng[0]-np.arange(5),:],axis=0)
        ETout[arng,0] = 0
        ETout[arng,1] = 0
        tstart = b[ii]+1
    if verbose:
        print( "  %d blinks detected and zeroed"%len(blinks) )
    return ETout, np.array(blinks, dtype=np.int64)

## EyeLink processing-specific
def ETtrial_smoother( ets0, box_size=300, end_smoothing=0 ):
    etsm = deepcopy(ets0)    
    for tt in range(9, box_size):
        etsm[tt//2, :] = np.mean(ets0[:tt,:], axis=0)
        etsm[-(tt//2), :] = np.mean(ets0[-tt:,:], axis=0)
    for tt in range(box_size//2, ets0.shape[0]-box_size//2+1):
        etsm[tt, :] = np.mean(ets0[range(tt-box_size//2, tt+box_size//2),:], axis=0)
    if end_smoothing > 0:
        etsm[:end_smoothing,:] = etsm[end_smoothing,:]
        etsm[-end_smoothing:,:] = etsm[-send_smoothing,:]
    return etsm


def ETtrial_filter( ets0, box_size=300, end_smoothing=0, med_filt = 0 ):
    #return deepcopy(ets0) - ETtrial_smoother( ets0, box_size=box_size, end_smoothing=end_smoothing)
    # median_filter
    ets1 = deepcopy(ets0)
    if med_filt > 0:
        for tt in range(med_filt,ets0.shape[0]-med_filt-1):
            ets1[tt,:] = np.median(ets0[range(tt-med_filt,tt+med_filt+1),:], axis=0)
    return ets1 - ETtrial_smoother( ets0, box_size=box_size, end_smoothing=end_smoothing)


def dsacc_compute( ets, sm=24, sac_gap=64, to_plot=False, drift_filter=500 ):
    # Set drift-filter = None to turn off: 500 seems pretty optimal
    assert sm%2==0, "smoothing (sm) must be multiple of 2"
    assert sac_gap%4==0, "sac_gap (sm) must be multiple of 4"
    # Subtract running average
    T = ets.shape[0]
    if drift_filter is not None:
        g = ETtrial_filter(ets, box_size=drift_filter)
    else:
        g = deepcopy(ets)
    x0 = g[:,0]
    y0 = g[:,1]
    x1 = deepcopy(x0)
    y1 = deepcopy(y0)
    for tt in range(T):
        x1[tt] = np.mean(x0[range(np.maximum(0,tt-sm//2), np.minimum(T-1,tt+sm//2))])
        y1[tt] = np.mean(y0[range(np.maximum(0,tt-sm//2), np.minimum(T-1,tt+sm//2))])

    dx = x1[sac_gap//2:T]-x1[:T-sac_gap//2]
    dy = y1[sac_gap//2:T]-y1[:T-sac_gap//2]

    dsacc = np.zeros(T)
    dsacc[range(sac_gap//4, T-sac_gap//4)] = dx**2+dy**2
    
    if to_plot:
        DU.ss(5,1)
        plt.subplot(511)
        plt.plot(x0,'g')
        plt.plot(x1,'k')
        plt.subplot(512)
        plt.plot(x1,'g')
        plt.plot(np.arange(sac_gap,T),x1[:-sac_gap],'r')
        plt.plot(np.arange(sac_gap//4, T-sac_gap//4), dx**2,'k')
        plt.subplot(513)
        plt.plot(y0,'c')
        plt.plot(y1,'k')
        plt.subplot(514)
        plt.plot(y1,'g')
        plt.plot(np.arange(sac_gap, T),y1[:-sac_gap],'r')
        plt.plot(np.arange(sac_gap//4, T-sac_gap//4), dy**2,'k')
        plt.subplot(515)
        plt.plot(dsacc,'k')
        plt.plot(np.arange(sac_gap//4, T-sac_gap//4), dx**2,'g')
        plt.plot(np.arange(sac_gap//4, T-sac_gap//4), dy**2,'c')
        plt.plot(dsacc,'k')
        plt.show()

    return dsacc


def saccade_detect( ets, blinks=None, min_sac_interval=100, T=4000):
    ts = np.arange(T)
    Camps, Csacs = [], []
    if blinks is not None:
        tr_blinks = blinks[:,0]//4000
    else:
        tr_blinks = []
    for trT in np.arange(ets.shape[0]//T):
        dsacc = utils.dsacc_compute( ets[ts+T*trT,:], sm=24, sac_gap=64, drift_filter=500 ) 
        pks, amps = utils.find_peaks(dsacc, thresh=1.0, clearance=min_sac_interval) 
        # Note low threshold, so can winnow after see distributions
        a = np.argsort(pks)
        pk_ts = pks[a]+ trT*T
        if trT in tr_blinks:
            # Find and subtract blinks
            btrial = np.where(tr_blinks == trT)[0][0]
            b = np.where((pk_ts < blinks[btrial,0]) | (pk_ts > blinks[btrial,1]))[0]
        else:
            b = np.arange(len(a))
        if np.sum(amps > 12000) > 0:
            print( "Potential blink trial %d"%trT )
        Csacs += list(pk_ts[b])
        Camps += list(amps[a[b]])
    Csacs = np.array(Csacs, dtype=np.int64)
    Camps = np.array(Camps, dtype=np.float32)
    
    return Csacs, Camps


def trial_saccade_display( tr, ets, sacc_ts, sacc_amps=None ):
    """# Trial display with saccades for high-res eye traces"""
    a = np.where((sacc_ts >= tr*4000) & (sacc_ts < (tr+1)*4000))[0]
    ts = sacc_ts[a]-tr*4000
    if sacc_amps is not None:
        print('Amplitudes:', np.sqrt(sacc_amps[a]))    
    DU.ss()
    plt.plot(np.arange(4000)*0.001, ets[tr*4000+np.arange(4000),0], 'g' )
    plt.plot(np.arange(4000)*0.001, ets[tr*4000+np.arange(4000),1], 'c' )
    ys = plt.ylim()
    for ii in range(len(a)):
        plt.plot(np.ones(2)*ts[ii]*0.001, ys,'r--')
    plt.xlim([0, 4])
    plt.ylim(ys)
    plt.show()

    a = np.where((sacc_ts >= tr*4000) & (sacc_ts < (tr+1)*4000))[0]
    ts = sacc_ts[a]-tr*4000
    if sacc_amps is not None:
        print('Amplitudes:', np.sqrt(sacc_amps[a]))    
    DU.subplot_setup( 1, 1, row_height=3.0 )
    plt.plot(ets[tr*4000+np.arange(4000),0], 'g' )
    plt.plot(ets[tr*4000+np.arange(4000),1], 'c' )
    ys = plt.ylim()
    for ii in range(len(a)):
        plt.plot(np.ones(2)*ts[ii], ys,'k')
    plt.ylim(ys)
    plt.show()
