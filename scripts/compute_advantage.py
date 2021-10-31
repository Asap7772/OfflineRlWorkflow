import argparse
import torch
import numpy as np
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy_chaining,load_data_from_npy_chaining_mult
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
import tqdm
import roboverse
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN, VQVAEEncoderConcatCNN, \
	ConcatBottleneckVQVAECNN, VQVAEEncoderCNN

cnn_params=dict(
			kernel_sizes=[3, 3, 3],
			n_channels=[16, 16, 16],
			strides=[1, 1, 1],
			hidden_sizes=[1024, 512, 256],
			paddings=[1, 1, 1],
			pool_type='max2d',
			pool_sizes=[2, 2, 1],  # the one at the end means no pool
			pool_strides=[2, 2, 1],
			pool_paddings=[0, 0, 0],
			image_augmentation=True,
			image_augmentation_padding=4,
			spectral_norm_conv=False,
			spectral_norm_fc=False,
		)

if __name__ == '__main__':
	ptu.set_gpu_mode(False)

	parser = argparse.ArgumentParser()
	parser.add_argument("--params1", type=str, required=True)
	parser.add_argument("--params2", type=str, required=True)

	args = parser.parse_args()
	data1, data2 = torch.load(args.params1), torch.load(args.params2)

	action_dim=8

	cnn_params.update(
		input_width=48,
		input_height=48,
		input_channels=3,
		output_size=1,
		added_fc_input_size=action_dim,
	)
	#qf1 = data1['trainer/trainer'].qf1
	#qf1 = VQVAEEncoderConcatCNN(**cnn_params)
	qf1 = ConcatCNN(**cnn_params)
	qf1.load_state_dict(data1['qf1_state_dict'])
	qf1.image_augmentation=False

	cnn_params.update(
		output_size=256,
		added_fc_input_size=0,
		hidden_sizes=[1024, 512],
		spectral_norm_fc=False,
		spectral_norm_conv=False,
	)

	policy_obs_processor1 = VQVAEEncoderCNN(**cnn_params)
	policy_obs_processor1.image_augmentation=False
	#policy_obs_processor2 = VQVAEEncoderCNN(**cnn_params)
	cnn_params.update(
		output_size=256,
		added_fc_input_size=0,
		hidden_sizes=[1024, 512],
		spectral_norm_fc=False,
		spectral_norm_conv=False,
	)
	policy_obs_processor2 = CNN(**cnn_params)
	policy_obs_processor2.image_augmentation=False


	pol1= TanhGaussianPolicy(
		obs_dim=cnn_params['output_size'],
		action_dim=action_dim,
		hidden_sizes=[256, 256, 256],
		obs_processor=policy_obs_processor1,
		shared_encoder=False,
	)

	pol2= TanhGaussianPolicy(
		obs_dim=cnn_params['output_size'],
		action_dim=action_dim,
		hidden_sizes=[256, 256, 256],
		obs_processor=policy_obs_processor2,
		shared_encoder=False,
	)


	pol1.load_state_dict(data1['policy_state_dict'])
	pol2.load_state_dict(data2['policy_state_dict'])
	#pol2 = data2['evaluation/policy'].stochastic_policy

	#pol1, pol2 = pol1.cuda(), pol2.cuda()
	#qf1 = qf1.cuda()
	
	buffers = []	
	ba = lambda x, p=1, y=None: buffers.append((path+x,dict(p=p,alter_type=y,)))
	path = '/home/stephentian/prior_data/'
	ba('pick_35obj_Widow250PickTrayMult-v0_5K_save_all_noise_0.1_2021-05-07T01-17-10_4375.npy', p=1, y='zero')
	ba('place_35obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-17-42_4875.npy', p=1)
	variant = {}
	variant['prior_buffer'] = buffers[0]
	variant['task_buffer'] = buffers[1]
	eval_env = roboverse.make('Widow250PickTrayMult-v0', transpose_image=True)
	data_buffer = load_data_from_npy_chaining(variant, eval_env, 'image') 

	advs = []
	for i in tqdm.tqdm(range(1000)):
		batch = data_buffer.random_batch(32)
		obs = ptu.from_numpy(batch['observations'])
		obs = obs.cpu() 
		# Using QVALUES from SETUP 1, compute the Qvalues of actions from setup 1 vs setup 2		

		pol1_acts, _, _, new_log_pi, *_ = pol1(
			obs, reparameterize=True, return_log_prob=True,
		)
		pol2_acts, _, _, new_log_pi, *_ = pol2(
			obs, reparameterize=True, return_log_prob=True,
		)
		qvals_pol1acts = qf1(obs, pol1_acts) 
		qvals_pol2acts = qf1(obs, pol2_acts) 
		adv = qvals_pol1acts - qvals_pol2acts
		advs.append(adv)
	advs = torch.cat(advs)
	print(advs.mean())
		


		
