
import os, sys
fpath = os.path.dirname(os.path.realpath(__file__))
print(fpath)
sys.path.insert(0, fpath) # why is this necessary?

# from . import regularization
# from . import LBFGS
# from . import NDNutils
# from . import trainers
# from . import NDNlayer
# from . import FFnetworks
# from . import NDNtorch
# from . import NDNLosses
# from . import datasets_dab
