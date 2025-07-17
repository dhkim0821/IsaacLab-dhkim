./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Prestoe2d-Direct-v0 --headless

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Prestoe2d-Direct-v0 --checkpoint logs/.../model_100.pt

./isaaclab.sh -p -m tensorboard.main -logdir logs/rsl_rl/prestoe2d_direct

Change constraints
1. change action space dimension
2. change loss computation in ppo.py (home/daros/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/rsl_rl/algorithms/ppo.py)