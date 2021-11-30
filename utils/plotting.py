### FILTER PLOT UTILITIES ###
import numpy as np
import matplotlib.pyplot as plt


def plot_filters_ST1D(ws, cmaps='gray', num_cols=8, row_height=2, time_reverse=True):
    """function to plot 1-D spatiotemporal filters (so, 2-d images) by passing in weights of multiple filters"""
    num_filters = ws.shape[-1]
    if num_filters < 8:
        num_cols = num_filters
    num_rows = np.ceil(num_filters/num_cols).astype(int)
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols)
    fig.set_size_inches(16, row_height*num_rows)
    plt.tight_layout()
    for cc in range(num_filters):
        ax = plt.subplot(num_rows, num_cols, cc+1)
        plt.imshow(ws[:,:,cc].T, cmap=cmaps)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.show()
### END plot_filters_ST1D

def plot_filters_ST2D(ws, sort=False):
    """Stolendirectly from neureye -> core.plot_filters, and modified"""
    from NDNT.utils import subplot_setup

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
        bestlag = np.argmax(np.std(wtmp, axis=0))
        plt.imshow(np.reshape(wtmp[:,bestlag], (sz[0], sz[1])), interpolation=None )
        wmax = np.argmax(wtmp[:, bestlag])
        wmin = np.argmin(wtmp[:, bestlag])
        plt.axis("off")

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
