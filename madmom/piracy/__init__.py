TORCH_TCN_REG = {'input_size': 162,
 				 'hidden_size': 150,
 				 'num_layers': 5,
 				 'hist_len': 10*100,
 				 'kernel_size': 5,
 				 'buffer_size': 10,
 				 'torch_loc': 'madmom/piracy/tcn_models/reg/step_100000.ckpt'}

TORCH_TCN_SMALL = {'input_size': 162,
				   'hidden_size': 50,
				   'num_layers': 5,
				   'kernel_size': 5, 
				   'hist_len': 10*100,
				   'buffer_size': 10,
				   'torch_loc': 'madmom/piracy/tcn_models/small/step_050000.ckpt'}

TORCH_TCN_TINY = {'input_size': 162,
 				  'hidden_size': 25,
 				  'num_layers': 3,
 				  'kernel_size': 5,
 				  'hist_len': 10*1000,
 				  'buffer_size': 5,
 				  'torch_loc': 'madmom/piracy/tcn_models/tiny/step_050000.ckpt'}