import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_losses(ckpts_directory):
    # look for a file prefixed with "events.out" to get the filename
    # e.g. 'experiments/exp_NIM_test/NIM_expt04/checkpoints/
    #       M011_NN/version1/events.out.tfevents.1674778321.beast.155528.0'
    event_filenames = glob.glob(os.path.join(ckpts_directory, 'events.out.*'))
    assert not len(event_filenames) < 1, 'no event file found in the checkpoints directory'
    assert not len(event_filenames) > 1, 'multiple event files found in the checkpoints directory'
    event_filename = event_filenames[0]
    event_acc = EventAccumulator(event_filename)
    event_acc.Reload()
    # Show all tags in the log file -- print(event_acc.Tags())
    # get wall clock, number of steps and value for a scalar 'Accuracy'
    loss_losses = [e.value for e in event_acc.Scalars('Loss/Loss')]
    train_losses = [e.value for e in event_acc.Scalars('Loss/Train')]
    reg_losses = [e.value for e in event_acc.Scalars('Loss/Reg')]
    train_epoch_losses = [e.value for e in event_acc.Scalars('Loss/Train (Epoch)')]
    val_epoch_losses = [e.value for e in event_acc.Scalars('Loss/Validation (Epoch)')]
    return {
        'all_loss': loss_losses,
        'train_loss': train_losses,
        'reg_loss': reg_losses,
        'train_epoch_loss': train_epoch_losses,
        'val_epoch_loss': val_epoch_losses
    }


# Tensorboard exponential moving average smoothing function
# from: https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
def smooth_ema(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - weight**num_acc
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)
    return smoothed


def smooth_conv(scalars, smoothing):
    window_length = int(smoothing * len(scalars))
    window_length = window_length if window_length > 0 else 1
    # smooth the losses using a convolution
    smoothed = np.convolve(scalars, np.ones(window_length)/window_length, mode='valid')
    return smoothed


def plot_losses(ckpts_directory, smoothing=None, figsize=(20, 8)):
    # load the losses
    losses = load_losses(ckpts_directory)
    
    # smooth the losses if requested
    if smoothing is not None:
        for key in losses.keys():
            losses[key] = smooth_ema(losses[key], smoothing)

    # plot the losses in a 2x3 grid
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(231)
    ax1.plot(losses['all_loss'])
    ax1.set_title('Loss/Loss')
    ax2 = fig.add_subplot(232)
    ax2.plot(losses['train_loss'])
    ax2.set_title('Loss/Train')
    ax3 = fig.add_subplot(233)
    ax3.plot(losses['reg_loss'])
    ax3.set_title('Loss/Reg')
    ax4 = fig.add_subplot(234)
    ax4.plot(losses['train_epoch_loss'])
    ax4.set_title('Loss/Train (Epoch)')
    ax5 = fig.add_subplot(235)
    ax5.plot(losses['val_epoch_loss'])
    ax5.set_title('Loss/Validation (Epoch)')
    plt.show()
