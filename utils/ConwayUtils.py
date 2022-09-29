import numpy as np
import matplotlib.pyplot as plt
#import scipy.io as sio
from copy import deepcopy
#import NDNT.utils.NDNutils as NDNutils
import NDNT.utils.DanUtils as DU


######### QUALITY CONTROL ##############
def median_smoothing( f, L=5):
    mout = deepcopy(f)
    for tt in range(L, len(f)-L):
        mout[tt] = np.median(f[np.arange(tt-L,tt+L)])
    return mout


def firingrate_datafilter( fr, Lmedian=10, Lhole=30, FRcut=1.0, frac_reject=0.1, to_plot=False, verbose=False ):
    """Generate data filter for neuron given firing rate over time"""
    if to_plot:
        verbose = True
    mx = median_smoothing(fr, L=Lmedian)
    df = np.zeros(len(mx))
    # pre-filter ends
    m = np.median(fr)
    v = np.where((mx >= m-np.sqrt(m)) & (mx <= m+np.sqrt(m)))[0]
    df[range(v[0], v[-1])] = 1
    m = np.median(fr[df > 0])
    if m < FRcut:
        # See if can salvage: see if median of 1/4 of data is above FRcut
        msplurge = np.zeros(4)
        L = len(mx)//4
        for ii in range(4):
            msplurge[ii] = np.median(mx[range(L*ii, L*(ii+1))])
        m = np.max(msplurge)
        if m < FRcut:
            if verbose:
                print('  Median criterium fail: %f'%m)
            return np.zeros(len(mx))
        # Otherwise back in game: looking for higher median
    v = np.where((mx >= m-np.sqrt(m)) & (mx <= m+np.sqrt(m)))[0]
    df = np.zeros(len(mx))
    df[range(v[0], v[-1]+1)] = 1
    # Last
    m = np.median(fr[df > 0])
    v = np.where((mx >= m-np.sqrt(m)) & (mx <= m+np.sqrt(m)))[0]
    # Look for largest holes
    ind = np.argmax(np.diff(v))
    largest_hole = np.arange(v[ind], v[ind+1])
    if len(largest_hole) > Lhole:
        if verbose:
            print('  Removing hole size=%d'%len(largest_hole))
        df[largest_hole] = 0
        # Decide whether to remove one side of hole or other based on consistent firing rate change after the hole
        chunks = [np.arange(v[0], largest_hole[0]), np.arange(largest_hole[-1], v[-1])]
        mfrs = [np.median(fr[chunks[0]]), np.median(fr[chunks[1]])]

        if (len(chunks[0]) > len(chunks[1])) & (mfrs[0] > FRcut):
            dom = 0
        else: 
            dom = 1
        if ((mfrs[dom] > mfrs[1-dom]) & (mfrs[1-dom] < mfrs[dom]-np.sqrt(mfrs[dom]))) | \
                ((mfrs[dom] < mfrs[1-dom]) & (mfrs[1-dom] > mfrs[dom]+np.sqrt(mfrs[dom]))):
            #print('eliminating', 1-dom)
            df[chunks[1-dom]] = 0
    
    # Eliminate any small islands of validity (less than Lhole)
    a = np.where(df == 0)[0]  # where there are zeros
    if len(a) > 0:
        b = np.diff(a) # the length of islands (subtracting 1: 0 is no island)
        c = np.where((b > 1) & (b < Lhole))[0]  # the index of the islands that are the wrong size
        # a[c] is the location of the invalid islands, b[c] is the size of these islands (but first point is zero before) 
        for ii in range(len(c)):
            df[a[c[ii]]+np.arange(b[c[ii]])] = 0

    # Final reject criteria: large fraction of mean firing rates in trial well above median poisson limit
    m = np.median(fr[df > 0])
    #stability_ratio = len(np.where(Ntrack[df > 0,cc] > m+np.sqrt(m))[0])/np.sum(df > 0)
    stability_ratio = len(np.where(abs(mx[df > 0]-m) > np.sqrt(m))[0])/np.sum(df > 0)
    if stability_ratio > frac_reject:
        if verbose:
            print('  Stability criteria not met:', stability_ratio)
        df[:] = 0
    if to_plot:
        DU.ss(2,1)
        plt.subplot(211)
        plt.plot(fr,'b')
        plt.plot(mx,'g')
        plt.plot([0,len(fr)], [m, m],'k')
        plt.plot([0,len(fr)], [m-np.sqrt(m), m-np.sqrt(m)],'r')
        plt.plot([0,len(fr)], [m+np.sqrt(m), m+np.sqrt(m)],'r')
        plt.subplot(212)
        plt.plot(df)
        plt.show()
   
    return df


def DFexpand( dfs, NT=None, BLsize=240 ):
    NBL, NC = dfs.shape
    if NT is None:
        NT = BLsize*NBL
    Xdfs = np.zeros([NT, NC])
    Xdfs[:BLsize*NBL, :] = np.repeat(dfs, BLsize, axis=0)
    if NT > BLsize*NBL:
        Xdfs[BLsize*NBL:, :] = Xdfs[BLsize*NBL-1, :]
    return Xdfs

########## RF MANIPULATION #################
def RFstd_evaluate( kstd, sm=3 ):
    from NDNT.modules.layers.convlayers import ConvLayer
    from NDNT.utils.DanUtils import max_multiD
    import torch

    NX = kstd.shape[0]
    L1 = sm*6+1
    g = np.zeros([L1,L1])
    for xx in range(L1):
        for yy in range(L1):
            g[xx,yy] = np.exp(-((xx-L1/2)**2 + (yy-L1/2)**2)/(2*sm**2))
    smooth_par = ConvLayer.layer_dict( 
        input_dims=[1,NX,NX,1], num_filters=1, bias=False, filter_dims=L1, padding='same', NLtype='lin')
    smoother = ConvLayer(**smooth_par)
    smoother.weight.data = torch.tensor(g, dtype=torch.float32).reshape([-1,1])
    smoothed_k = smoother(torch.tensor(np.reshape(kstd,[-1,1])))[0,:].reshape([NX,NX]).detach().numpy()
    x, y = max_multiD( smoothed_k )
    SNR = smoothed_k[x,y] / np.mean(smoothed_k)
    return x,y, SNR


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


def slow_drift_correct( ETin, tau=120000, to_plot=True, average_ends=True):
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
    if average_ends:
        for ii in range(nsig):
            meansig[:tau//2, ii] = np.mean(meansig[:tau//2, ii])
            meansig[-tau//2-2:, ii] = np.mean(meansig[-tau//2:, ii])
    else:
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
    
    if ETin.shape[1] > 2:
        # then binocular
        a = np.where(
            (abs(ETin[:,0]) > blink_thresh) | (abs(ETin[:,1]) > blink_thresh) | \
            (abs(ETin[:,2]) > blink_thresh) | (abs(ETin[:,3]) > blink_thresh))[0]
    else:
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
        ETout[arng,:] = 0
        #ETout[arng,1] = 0
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
        etsm[-end_smoothing:,:] = etsm[-end_smoothing,:]
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
    """ 
    drift-filter = None to turn off: 500 seems pretty optimal
    trial_edge: ignore saccades on edges of trial, effectively in units of ms
    """

    assert sm%2==0, "smoothing (sm) must be multiple of 2"
    assert sac_gap%4==0, "sac_gap (sm) must be multiple of 4"
    # Subtract running average
    T = ets.shape[0]
    if drift_filter is not None:
        g = ETtrial_filter(ets, box_size=drift_filter)
    else:
        g = deepcopy(ets)

    #nsig = g.shape[1]
    dsacc = np.zeros(T)

    xs = deepcopy(g)
    for tt in range(T):
        xs[tt, :] = np.mean(g[range(np.maximum(0,tt-sm//2), np.minimum(T-1,tt+sm//2)), :], axis=0)
    
    #dxs = xs[sac_gap//2:T, :]-xs[:T-sac_gap//2, :]

    for ii in range(g.shape[1]):
        x0 = g[:, ii]
        #y0 = g[:,1]
        x1 = deepcopy(x0)
        #y1 = deepcopy(y0)
        for tt in range(T):
            x1[tt] = np.mean(x0[range(np.maximum(0,tt-sm//2), np.minimum(T-1,tt+sm//2))])
            #y1[tt] = np.mean(y0[range(np.maximum(0,tt-sm//2), np.minimum(T-1,tt+sm//2))])

        dx = x1[sac_gap//2:T]-x1[:T-sac_gap//2]
        #dy = y1[sac_gap//2:T]-y1[:T-sac_gap//2]
        #dsacc[range(sac_gap//4, T-sac_gap//4)] = dx**2+dy**2
        dsacc[range(sac_gap//4, T-sac_gap//4)] += dx**2
    
    if to_plot:
        if g.shape[1] > 2:
            print("This needs to be updated to deal with binocular eye tracking")
        DU.ss(5,1)
        plt.subplot(511)
        plt.plot(g[:, 0],'g')
        plt.plot(x1,'k')
        plt.subplot(512)
        plt.plot(x1,'g')
        plt.plot(np.arange(sac_gap,T),x1[:-sac_gap],'r')
        plt.plot(np.arange(sac_gap//4, T-sac_gap//4), dx**2,'k')
        plt.subplot(513)
        plt.plot(g[:, 1],'c')
        #plt.plot(y1,'k')
        plt.subplot(514)
        #plt.plot(y1,'g')
        plt.plot(g[:, 1],'g')
        plt.plot(np.arange(sac_gap, T), g[:-sac_gap, 1],'r')
        plt.plot(np.arange(sac_gap//4, T-sac_gap//4), dy**2,'k')
        plt.subplot(515)
        plt.plot(dsacc,'k')
        plt.plot(np.arange(sac_gap//4, T-sac_gap//4), dx**2,'g')
        plt.plot(np.arange(sac_gap//4, T-sac_gap//4), dy**2,'c')
        plt.plot(dsacc,'k')
        plt.show()

    return dsacc


def saccade_detect( ets, blinks=None, min_sac_interval=100, T=4000, trial_edge=80):
    ts = np.arange(T)
    Camps, Csacs = [], []
    if blinks is not None:
        tr_blinks = blinks[:,0]//4000
    else:
        tr_blinks = []
    for trT in np.arange(ets.shape[0]//T):
        dsacc = dsacc_compute( ets[ts+T*trT,:], sm=24, sac_gap=64, drift_filter=500 ) 
        pks, amps = DU.find_peaks(dsacc, thresh=1.0, clearance=min_sac_interval) 

        # Eliminate saccades on trial edges
        a = np.where((pks >= trial_edge) & (pks < T-trial_edge))[0]
        pks = pks[a]
        amps = amps[a]
        
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
    for ii in range(ets.shape[1]//2):
        DU.ss()
        plt.plot(np.arange(4000)*0.001, ets[tr*4000+np.arange(4000),2*ii], 'g' )
        plt.plot(np.arange(4000)*0.001, ets[tr*4000+np.arange(4000),2*ii+1], 'c' )
        ys = plt.ylim()
        for ii in range(len(a)):
            plt.plot(np.ones(2)*ts[ii]*0.001, ys,'r--')
        plt.xlim([0, 4])
        plt.ylim(ys)
        plt.show()
