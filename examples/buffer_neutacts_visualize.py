import numpy as np
import matplotlib.pyplot as plt
import gc

save_path='/home/asap7772/cog/images/'
paths = [
    # ('/nfs/kun1/users/asap7772/prior_data/task_singleneut_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-03-25T22-52-59_9750.npy', 'coll_task'),
    # ('/nfs/kun1/users/asap7772/cog_data/drawer_task.npy', 'public_task'),
    # ('/nfs/kun1/users/asap7772/cog_data/closed_drawer_prior.npy', 'public_prior'),
    # ('/nfs/kun1/users/asap7772/prior_data/coglike_prior_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-03T17-32-00_10000.npy', 'cog_like_prior'),
    ('/nfs/kun1/users/asap7772/prior_data/coglike_task_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-03T17-32-05_10000.npy', 'cog_like_task'),
    ]

for p, fname in paths:
    print(fname)
    with open(p, 'rb') as f:
        gc.collect()
        data = np.load(f, allow_pickle=True)
        import ipdb; ipdb.set_trace()
        bins = [i * .1 for i in range(11)]

        actions = np.array([np.array(data[i]['actions']) for i in range(len(data))])
        neut_acts = actions.reshape(-1,actions.shape[-1])[:,-1]

        plt.figure()
        plt.title('hist for ' + fname)
        plt.hist(neut_acts, bins=bins)
        plt.savefig(save_path + 'hist' + fname + '.png')
        plt.close()