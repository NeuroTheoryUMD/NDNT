### FILTER PLOT UTILITIES ###
import numpy as np
import matplotlib.pyplot as plt


def plot_filters_ST1D(  ws, cmaps='gray', num_cols=8, row_height=2, time_reverse=True):
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

