print(f'Invoking __init__.py for {__name__}')


from . import ffnet_dicts as dicts
from . import plotting as plot


from .create_reg_matrices import *

# General NDN utility functions
from .NDNutils import *

# Plot functions used by the NDN
from .plotting import plot_filters_1D
from .plotting import plot_filters_2D
from .plotting import plot_filters_ST1D
from .plotting import plot_filters_ST2D
from .plotting import plot_filters_ST3D

# Dan-specific utilities
from .DanUtils import *
