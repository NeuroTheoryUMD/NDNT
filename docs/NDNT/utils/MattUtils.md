Module NDNT.utils.MattUtils
===========================

Functions
---------

`load_losses(ckpts_directory)`
:   

`plot_losses(ckpts_directory, smoothing=None, figsize=(20, 8))`
:   

`smooth_conv(scalars, smoothing)`
:   

`smooth_ema(scalars: list[float], weight: float) ‑> list[float]`
:   EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699