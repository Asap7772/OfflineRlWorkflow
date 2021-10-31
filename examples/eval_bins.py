import roboverse
import torch

env_name = 'Widow250MultiObjectGraspTrain-v0'
env = roboverse.make(env_name, transpose_image=True)
env.multi_tray_num = 0

chkpt = ''
module = torch.load(chkpt)

