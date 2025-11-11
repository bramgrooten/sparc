# SPARC

This repository contains our open-source code for the SPARC paper on generalization in RL. 
SPARC is a **s**ingle-**p**hase **a**daptation method for **r**obust **c**ontrol.
It works especially well in out-of-distribution (OOD) environments, 
where the agent adapts to new contextual conditions 
_without_ access to privileged context information at test time. 
SPARC is able to infer the context through a history of the agent's own actions and observations. 

The paper presents experiments on Gran Turismo 7 and MuJoCo environments. 
The code for Gran Turismo 7 (which runs exclusively on PlayStation) is proprietary and not included in this repository. 
The MuJoCo code is open source. 
We include the newly created benchmarks for MuJoCo environments with wind:

- `WindHalfCheetah-v5`
- `WindHopper-v5`
- `WindWalker2d-v5`

These are all in the `environments` folder. See all methods in the `algorithms` directory.


## Installation

Run the following commands to install:

```shell
conda create -n sparc python=3.10
conda activate sparc
pip install -r requirements.txt
```

## Training

To train SPARC on a wind-perturbed MuJoCo environment, run:

```shell
python -m train --alg sparc --env WindHalfCheetah-v5 --wandb_mode disabled
```

Implemented algorithms are:
- SPARC
- RMA (note: this baseline must be trained in 2 separate phases, see both files)
- SAC (only observations as input)
- History Input (no context inference, history as input)
- Oracle (privileged context information, even at test time)


## Visualize 

To view the wind-perturbed environments, run:
```shell
python -m utils.visualize_envs
```
Inside that script, you can adjust the environment and the wind strength.

The following table shows the wind ranges that we used in our experiments:

| Environment  | Train _x_   | Train _z_    | Test _x_   | Test _z_   |
|:-------------|:------------|:-------------|:-----------|:-----------|
| HalfCheetah  | [-2.5, 2.5] | [-5, 5]      | [-5, 5]    | [-10, 10]  |
| Hopper       | [-10, 10]   | [-2.5, 2.5]  | [-20, 20]  | [-5, 5]    |
| Walker2d     | [-10, 10]   | [-2.5, 2.5]  | [-20, 20]  | [-5, 5]    |

