# Offline RL Workflow 

This repository accompanies the following paper:

**A Workflow for Offline Model-Free Robotic RL** <br/>
Aviral Kumar*, Anikait Singh*, Stephen Tian, Chelsea Finn, Sergey Levine <br/>
[Conference on Robot Learning](https://www.robot-learning.org/), 2021 <br/>
[Website](https://sites.google.com/view/offline-rl-workflow) | [Arxiv](https://arxiv.org/abs/2109.10813) | [Video](https://www.youtube.com/watch?v=tkiQZSy0waI)

Offline reinforcement learning (RL) enables learning control policies
by utilizing only prior experience, without any online interaction. This can allow
robots to acquire generalizable skills from large and diverse datasets, without any
costly or unsafe online data collection. Despite recent algorithmic advances in
offline RL, applying these methods to real-world problems has proven challenging. Although offline RL methods can learn from prior data, there is no clear
and well-understood process for making various design choices, from model architecture to algorithm hyperparameters, without actually evaluating the learned
policies online. In this paper, our aim is to develop a practical workflow for using
offline RL analogous to the relatively well-understood workflows for supervised
learning problems. To this end, we devise a set of metrics and conditions that
can be tracked over the course of offline training, and can inform the practitioner
about how the algorithm and model architecture should be adjusted to improve final performance. Our workflow is derived from a conceptual understanding of the
behavior of conservative offline RL algorithms and cross-validation in supervised
learning. We demonstrate the efficacy of this workflow in producing effective policies without any online tuning, both in several simulated robotic learning scenarios
and for three tasks on two distinct real robots, focusing on learning manipulation
skills with raw image observations with sparse binary rewards. 

This code is based on the [COG](https://github.com/avisingh599/cog)
implementation. 

## Usage

In this paper, two offline RL algorithms were tested: [CQL](https://github.com/Asap7772/CQL) and [BRAC](https://github.com/google-research/google-research/tree/master/behavior_regularized_offline_rl). We utilized these algorithms as baseline on which we applied our workflow. Below are some sample commands to replicate baseline perfromance.

```bash
# CQL baseline
python examples/cql_workflow.py --env=Widow250DoubleDrawerOpenGraspNeutral-v0 --max-path-length=50 --buffer 0 --num_traj 100
python examples/cql_workflow.py --env=Widow250DoubleDrawerCloseOpenGraspNeutral-v0 --max-path-length=80 --buffer 1 --num_traj 100
python examples/cql_workflow.py --env=Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0 --max-path-length=80 --buffer 2 --num_traj 100

# BRAC baseline
python examples/brac_workflow.py --env=Widow250DoubleDrawerOpenGraspNeutral-v0 --max-path-length=50 --buffer 0 --num_traj 100
python examples/brac_workflow.py --env=Widow250DoubleDrawerCloseOpenGraspNeutral-v0 --max-path-length=80 --buffer 1 --num_traj 100
python examples/brac_workflow.py --env=Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0 --max-path-length=80 --buffer 2 --num_traj 100
```

Parts of the workflow can be applied using their respective flags. For example, to run the CQL algorithm with the VIB architecture utilized in the overfitting regime, you can run the following command:

```bash
python examples/cql_workflow.py --env=Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0 --max-path-length=80 --buffer 2 --num_traj 100 --bottleneck --bottlneck_const=1e-3
```

Here is a list of other flags that are available for the CQL and BRAC algorithms. I have listed some relevant flags and their usage.

- General Hyperparameters
  - '--env': Environment to train on.
  - '--buffer': Buffer to use.
  - '--max-path-length': Maximum length of a single rollout.
- CQL algorithm
  - '--min_q_weight': CQL alpha parameter. Controls conservativeness of the algorithm.
- Brac algorithm
  - '--beta': Value of beta in BRAC
  - '--behavior_path': Path to behavior policy
- Workflow Corrections
  - '--bottleneck': Use the VIB architecture. (Overfitting Correction)
  - '--bottleneck_const': Regularization constant for the VIB.
  - '--num_traj': Number of trajectories to collect.
  - '--dr3_feat': Use DR3 regularization. (Underfitting Correction)
  - '--dr3_weight': Weight for DR3 regularization.
  - '--regularization': Enable L1/L2 regularization. (Overfitting Correction)
  - '--regularization_type': L1/L2 regularization type.
  - '--dropout': Enable dropout regularization. (Overfitting Correction)
  - '--resnet_enc': Use encoder with residual layers (Underfitting Correction)

## Setup
Our code is based on CQL, which is in turn based on [rlkit](https://github.com/vitchyr/rlkit). 
The setup instructions are similar to rlkit, but we repeat them here for convenience:

```shell script
conda create --name workflow python=3.8
conda activate workflow
pip install -r requirements.txt
```

After the above, please install the following fork of [roboverse](https://github.com/Asap7772/roboverse) by running:
```bash
python setup.py develop
``` 

## Datasets
The datasets used in this project can be downloaded using this 
[Google drive link](https://drive.google.com/drive/folders/1jxBQE1adsFT1sWsfatbhiZG6Zkf3EW0Q?usp=sharing). (to be updated)

If you would like to download the dataset on a remote machine via the command
line, consider using [gdown](https://pypi.org/project/gdown/). 

