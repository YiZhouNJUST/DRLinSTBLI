DRLinSTBLI
=
# Introduction
**Case MBVIM in the paper:**

_Control of Hypersonic Shock-Wave/Laminar Boundary-Layer Interaction using Deep Reinforcement Learning_

**Abstract:**
Deep reinforcement learning (DRL) is applied to control the flow separation induced by the shock-wave/boundary-layer interaction (SBLI)  for the first time in a laminar compression ramp configuration simulated via OpenFOAM.
The wall pressure coefficient serves as the observed state, while the agent outputs microjet actuation as the corresponding action. 
The wall skin friction coefficient, indicating the extent of SBLI-induced flow separation, is used as the reward to guide the learning process.
Under both constant and varying incoming Mach number conditions, a converged controlled model can be obtained through the repeated DRL training, resulting in a 40% reduction in the separation area.
The suppression of the separated shear layer is owing to the weakening of the separation shock and the energizing of the local boundary layer under the DRL control.
To reduce the computational costs of CFD simulations, a long short-term memory network is employed as a surrogate model for the environment in DRL. 
This model-based approach reduces overall training time by nearly 60% and at the same time achieves accuracy comparable to the model-free method that only interacts with the CFD environment.
The present study opens a gateway to the closed-loop control of SBLI using state-of-the-art DRL. 

![image](https://github.com/YiZhouNJUST/DRLinSTBLI/blob/master/framework.jpg)

# Requirement
OpenFOAM 7, Anaconda, Pytorch, Stable-Baselines3, gym

# Code Guide
**0:** Initial flow

**constant:** CFD parameters

**system:** control file of CFD

**DRLinSTBLI-58w_oneJet_3D/main.py:** control file of DRL

**DRLinSTBLI-58w_oneJet_3D/Env.py:** environment model

**DRLinSTBLI-58w_oneJet_3D/ExchangeFoam.py:** data exchange between CFD and DRL

**DRLinSTBLI-58w_oneJet_3D/LSTMenv.py**: surrogate model for environment

# Run
python DRLinSTBLI-58w_oneJet_3D/main.py

successful runnning
'''
**Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
numEpisode: 1
numEpisode: 2
numEpisode: 3
numEpisode: 4
numEpisode: 5
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 50       |
|    ep_rew_mean     | -31.7    |
| time/              |          |
|    episodes        | 4        |
|    fps             | 0        |
|    time_elapsed    | 16046    |
|    total_timesteps | 200      |
| train/             |          |
|    actor_loss      | 0.726    |
|    critic_loss     | 0.0877   |
|    learning_rate   | 0.0001   |
|    n_updates       | 99       |
---------------------------------
numEpisode: 6
numEpisode: 7
numEpisode: 8
numEpisode: 9
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 50       |
|    ep_rew_mean     | -33.6    |
| time/              |          |
|    episodes        | 8        |
|    fps             | 0        |
|    time_elapsed    | 31518    |
|    total_timesteps | 400      |
| train/             |          |
|    actor_loss      | 0.977    |
|    critic_loss     | 0.0886   |
|    learning_rate   | 0.0001   |
|    n_updates       | 299      |
---------------------------------
numEpisode: 10**
'''
