import numpy as np
import matplotlib.pyplot as plt

def get_buffer_size(data):
    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    return num_transitions

def process_images(observations):
    output = []
    for i in range(len(observations)):
        image = observations[i]['image']
        if len(image.shape) == 3:
            image = np.transpose(image, [2, 0, 1])
            image = (image.flatten())/255.0
        else:
            print('image shape: {}'.format(image.shape))
            raise ValueError
        output.append(dict(image=image))
    return output


def plot(data, num_traj, fig, ax1,ax2, ax3, color, legend_name):
    # color = plt.cm.get_cmap('gist_rainbow', num_traj)

    # ax1.set_xlim([-1,1])
    # ax1.set_ylim([-1,1])
    # ax2.set_xlim([-1,1])
    # ax2.set_ylim([-1,1])
    # ax3.set_xlim([-1,1])
    # ax3.set_ylim([-1,1])

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')

    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')

    fig.suptitle(save_path)

    for j in range(num_traj):
        states = np.array([data[j]['observations'][i]['state'] for i in range(len(data[j]['observations']))])[:,:3]
        x,y,z = states[:,0],states[:,1],states[:,2]

        print(x[0],y[0],z[0])
        # c = int(np.random.rand()*num_traj)
        if j == 0:
            params = dict(c=color, marker='.', linewidth=0, markersize=1, label=legend_name) #linewidth=0.05, c=color(c)
        else:
            params = dict(c=color, marker='.', linewidth=0, markersize=1)
        l1 = ax1.plot(x,y,**params)
        l2 = ax2.plot(x,z,**params)
        l3 = ax3.plot(y,z,**params)

        if j+1 % 100 == 0: 
            print(j+1)
            plt.savefig(save_path+'.png',dpi=1200)
    plt.savefig(save_path+'.png',dpi=1200)
    return l1,l2,l3

if __name__ == '__main__':
    save_path='/home/asap7772/cog/images/newoldtask_scatter'
    num_traj = 100

    fig, (ax1,ax2,ax3) = plt.subplots(3,1)
    fig.set_figheight(10)
    fig.set_figwidth(7)
    
    p = '/nfs/kun1/users/asap7772/prior_data/task_singleneut_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-03-25T22-52-59_9750.npy'
    with open(p, 'rb') as f:
        data = np.load(f, allow_pickle=True)
    
    plots1 = plot(data, num_traj,fig,ax1,ax2,ax3,'r', 'new_task')
    print('finished',p)

    p = '/nfs/kun1/users/asap7772/cog_data/drawer_task.npy'
    with open(p, 'rb') as f:
        data = np.load(f, allow_pickle=True)
    
    plots2 = plot(data, num_traj,fig,ax1,ax2,ax3,'g', 'old_task')
    print('finished',p)

    # p = '/nfs/kun1/users/asap7772/prior_data/pickplace_newenv_Widow250PickPlaceMultiObjectMultiContainerTrain-v0_20K_save_all_noise_0.1_2021-03-18T01-38-58_19500.npy'
    # with open(p, 'rb') as f:
    #     data = np.load(f, allow_pickle=True)
    
    # plots3 = plot(data, num_traj,fig,ax1,ax2,ax3,'b', 'prior_pickplace')
    # print('finished',p)

    # p = '/nfs/kun1/users/asap7772/cog_data/drawer_task.npy'
    # with open(p, 'rb') as f:
    #     data = np.load(f, allow_pickle=True)
    
    # plots4 = plot(data, num_traj,fig,ax1,ax2,ax3,'r', 'task_drawer')
    # print('finished',p)

    # p = '/nfs/kun1/users/asap7772/cog_data/closed_drawer_prior.npy'
    # with open(p, 'rb') as f:
    #     data = np.load(f, allow_pickle=True)
    
    # plots5 = plot(data, num_traj,fig,ax1,ax2,ax3,'y', 'prior_drawer')
    # print('finished',p)
    
    plt.legend()
    plt.savefig(save_path+'.png',dpi=1200)
    print('legend added')