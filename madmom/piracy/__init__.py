import os

# Resolve checkpoint paths absolutely against this package's directory, so they
# load regardless of the process working directory (the Spectrum app launches
# the beat tracker with CWD set to Madmom/env/Scripts).
_TCN_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'tcn_models')

TORCH_TCN_REG = {'input_size': 162,
 				 'hidden_size': 150,
 				 'num_layers': 5,
 				 'hist_len': 10*100,
 				 'kernel_size': 5,
 				 'buffer_size': 10,
 				 'torch_loc': os.path.join(_TCN_MODELS_DIR, 'reg_latest.ckpt')}

TORCH_TCN_SMALL = {'input_size': 162,
				   'hidden_size': 50,
				   'num_layers': 5,
				   'kernel_size': 5,
				   'hist_len': 10*100,
				   'buffer_size': 10,
				   'torch_loc': os.path.join(_TCN_MODELS_DIR, 'small', 'step_050000.ckpt')}

TORCH_TCN_TINY = {'input_size': 162,
 				  'hidden_size': 25,
 				  'num_layers': 3,
 				  'kernel_size': 5,
 				  'hist_len': 10*1000,
 				  'buffer_size': 5,
 				  'torch_loc': os.path.join(_TCN_MODELS_DIR, 'tiny', 'step_050000.ckpt')}