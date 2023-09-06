### FILTER PLOT UTILITIES ###
import numpy as np
import matplotlib.pyplot as plt

def plot_filters_1D(ws, cmap='gray', num_cols=8, row_height=2, fix_scale=True, **kwargs):
    """function to plot 1-D spatiotemporal filters (so, 2-d images) by passing in weights of multiple filters"""
    num_filters = ws.shape[-1]
    if num_filters < 8:
        num_cols = 8
    num_rows = np.ceil(num_filters/num_cols).astype(int)
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
    fig.set_size_inches(16, row_height*num_rows)
    plt.tight_layout()

    m = np.max(abs(ws))
    for cc in range(num_filters):
        ax = plt.subplot(num_rows, num_cols, cc+1)
        plt.plot(ws[...,cc], color='k')
        plt.axhline(0, color='k', ls='--')
        if fix_scale:
            plt.ylim([-m, m])
        # ax.set_yticklabels([])
        # ax.set_xticklabels([])
    plt.show()


def plot_filters_ST1D(ws, cmap='gray', num_cols=None, row_height=2, fix_scale=True, **kwargs):
    """function to plot 1-D spatiotemporal filters (so, 2-d images) by passing in weights of multiple filters"""
    num_filters = ws.shape[-1]
    if num_cols is None:
        num_cols = 8
    num_rows = np.ceil(num_filters/num_cols).astype(int)
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
    fig.set_size_inches(16, row_height*num_rows)
    plt.tight_layout()

    m = np.max(abs(ws))
    for cc in range(num_filters):
        ax = plt.subplot(num_rows, num_cols, cc+1)
        if fix_scale:
            plt.imshow(ws[:,:,cc].T, cmap=cmap, aspect='auto', interpolation='none', vmin=-m, vmax=m)
        else:
            plt.imshow(ws[:,:,cc].T, cmap=cmap, aspect='auto', interpolation='none')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.show()
### END plot_filters_ST1D

def plot_filters_ST2D(ws, sort=False, **kwargs):
    """Stolendirectly from neureye -> core.plot_filters, and modified"""
    from NDNT.utils import subplot_setup

    if len(ws.shape) < 4:
        nfilt = 1
    else:
        nfilt = ws.shape[-1]
    ei_mask = np.ones(nfilt)
    sz = ws.shape
    # Collapse spatial dims together
    w = ws.reshape(sz[0]*sz[1], sz[2], nfilt)
    
    if type(sort) is np.ndarray:
        cinds = sort
    elif sort:
        n = np.asarray([np.max(abs(w[:,:,i])) for i in range(nfilt)])
        cinds = np.argsort(n)[::-1][-len(n):]
    else:
        cinds = np.arange(0, nfilt)

    sx = np.ceil(np.sqrt(nfilt*2)).astype(int)
    sy = np.round(np.sqrt(nfilt*2)).astype(int)
    # sx,sy = U.get_subplot_dims(nfilt*2)
    mod2 = sy % 2
    sy += mod2
    sx -= mod2

    subplot_setup(sx, sy)
    for cc,jj in zip(cinds, range(nfilt)):
        plt.subplot(sx,sy,jj*2+1)
        wtmp = np.squeeze(w[:, :, cc])
        m = np.max(abs(wtmp))
        bestlag = np.argmax(np.std(wtmp, axis=0))
        plt.imshow(np.reshape(wtmp[:,bestlag], (sz[0], sz[1])), interpolation=None, vmin=-m, vmax=m )
        wmax = np.argmax(wtmp[:, bestlag])
        wmin = np.argmin(wtmp[:, bestlag])
        plt.axis("off")
        plt.title(str(cc))
        
        plt.subplot(sx,sy,jj*2+2)
        if ei_mask[cc]>0:
            plt.plot(wtmp[wmax,:], 'b-')
            plt.plot(wtmp[wmin,:], 'b--')
        else:
            plt.plot(wtmp[wmax, :], 'r-')
            plt.plot(wtmp[wmin, :], 'r--')

        plt.axhline(0, color='k')
        plt.axvline(bestlag, color=(.5, .5, .5))
        plt.axis("off")


def plot_filters_ST3D(ws, sort=False, **kwargs):
    from NDNT.utils import subplot_setup

    MAX_PER_LINE = 2
    clrs='kgbrcm'

    nfilt = ws.shape[-1]
    ei_mask = np.ones(nfilt)
    sz = ws.shape
    nchan = sz[0]

    # Max 
    if nfilt < MAX_PER_LINE:
        sy = nfilt*(nchan+1)
        sx = 1
    else:
        sy = ((nchan+1)*MAX_PER_LINE)
        sx = np.ceil(nfilt/MAX_PER_LINE).astype(int)

    if type(sort) is np.ndarray:
        cinds = sort
    elif sort:
        n = np.asarray([np.max(abs(w[:,:,i])) for i in range(nfilt)])
        cinds = np.argsort(n)[::-1][-len(n):]
    else:
        cinds = np.arange(0, nfilt)

    # Collapse spatial dims together
    wt = ws.reshape(nchan, sz[1]*sz[2], sz[3], nfilt)
    
    f = subplot_setup(sx, sy, fighandle=True)
    f.tight_layout()
    for cc,jj in zip(cinds, range(nfilt)):
        bestlags = np.argmax(np.max(abs(wt[:,:,:, cc]), axis=1), axis=1)
        bestxy = np.argmax(np.max(abs(wt[:,:,:, cc]), axis=2), axis=1)

        plt.subplot(sx,sy,jj*(nchan+1)+1)
        for nn in range(nchan):
            plt.plot(wt[nn, bestxy[nn], :, cc ], clrs[nn])
            #plt.axvline(bestlags[nn], clrs[nn]+'--')
        plt.plot([0, sz[3]],[0,0],'r--')
        #plt.axhline(0, color='k')
        plt.title(str(cc))

        m = np.max(abs(wt[:,:,:,cc]))
        for nn in range(nchan):
            plt.subplot(sx,sy,jj*(nchan+1)+2+nn)
            plt.imshow( ws[nn, :, :, bestlags[nn], cc], interpolation=None, vmin=-m, vmax=m )
            plt.axis("off")


## Additional display-type plots
def plot_scatter( xs, ys, clr='g' ):
    assert len(xs) == len(ys), 'data dont match'
    for nn in range(len(xs)):
        plt.plot([xs[nn]], [ys[nn]], clr+'o', fillstyle='full')
        if clr != 'k':
            plt.plot([xs[nn]], [ys[nn]], 'ko', fillstyle='none')
    #plt.show()
    

def plot_internal_weights(ws, num_inh=None):
    from copy import deepcopy
    ws_play = deepcopy(ws)
    num_dim = ws.shape[0]
    if num_inh is not None:
        ws_play[range(num_dim-num_inh, num_dim), :] *= -1
    m = np.max(abs(ws_play))
    plt.imshow(ws_play, cmap='bwr', vmin=-m, vmax=m)
