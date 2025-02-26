DRLinSTBLI
=
# Introduction
Case MBVIM in the paper:

_Control of Hypersonic Shock-Wave/Laminar Boundary-Layer Interaction using Deep Reinforcement Learning_

![image](https://github.com/YiZhouNJUST/DRLinSTBLI/blob/master/framework.jpg)

# Requirement
OpenFOAM 7, Anaconda, Pytorch, Stable-Baselines3, gym

# Code Guide
0: Initial flow

constant: CFD parameters

system: control file of CFD

DRLinSTBLI-58w_oneJet_3D/main.py: control file of DRL

DRLinSTBLI-58w_oneJet_3D/Env.py: environment model

DRLinSTBLI-58w_oneJet_3D/ExchangeFoam.py: data exchange between CFD and DRL

DRLinSTBLI-58w_oneJet_3D/LSTMenv.py: surrogate model for environment

# Run
python DRLinSTBLI-58w_oneJet_3D/main.py
