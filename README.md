# Explainable Federated Reinforcement Learning in the Smart Grid

This repository contains the implementation of **Federated Reinforcement Learning (FRL)** and **Explainable Federated Reinforcement Learning (FRL-ESP)** for smart-grid energy management.

The project investigates how multiple reinforcement-learning agents can coordinate their decisions in a federated setting while incorporating explainability into the decision-making process. The experiments focus on distributed energy-resource management, including battery energy storage systems (BESS), photovoltaic (PV) generation, load profiles, and grid-level control.

The repository contains two main implementations:

* **`BESS_FRL/`** – baseline Federated Reinforcement Learning implementation.
* **`ESP_FRL/`** – FRL implementation incorporating the ESP-based explainability framework.

---

## 1. Repository Structure

The main structure of the repository is:

```text
Explainable-federated-reinforcement-learning-in-the-Smart-Grid/
│
├── README.md
├── requirements.txt
│
├── BESS_FRL/
│   ├── main.py
│   ├── agent.py
│   ├── agent_branch.py
│   ├── agent_muti.py
│   ├── environment.py
│   ├── environment_branch.py
│   ├── environment_branch_1.py
│   ├── environment_muti.py
│   ├── deep_q_network.py
│   ├── pvloaddata.mat
│   ├── pvloaddata.xls
│   └── ...
│
└── ESP_FRL/
    ├── main.py
    ├── agent_branch.py
    ├── agent_muti.py
    ├── environment_muti.py
    ├── deep_q_network_ESP.py
    └── ...
```

Several additional scripts are included for alternative network architectures, experiments, plotting, and debugging. The primary entry point for the FRL experiments is `main.py`.

---

## 2. Method Overview

### Federated Reinforcement Learning

The baseline implementation uses multiple reinforcement-learning agents to make distributed control decisions for the smart-grid environment.

Each agent observes its local state and selects an action using a Deep Q-Network (DQN). Information from multiple agents can then be combined through the federated reinforcement-learning architecture.

The default implementation currently defines:

```python
numofagents = 55
```

and each agent has five possible actions:

```python
num_action = 5
```

The action space is related to battery charging/discharging decisions.

### FRL with ESP

The `ESP_FRL/` implementation extends the federated reinforcement-learning framework with the **ESP explainability mechanism**.

The objective is not only to obtain effective control decisions but also to provide additional information for understanding the learned decision-making process.

---

## 3. Computing Environment

The original experiments were developed and tested using the following environment:

| Component        | Configuration                 |
| ---------------- | ----------------------------- |
| CPU              | Intel Core i7                 |
| GPU              | NVIDIA GTX TITAN X            |
| Memory           | 64 GB RAM                     |
| Operating System | Ubuntu 20.04 LTS / Windows 10 |
| Python           | 3.7.7                         |
| TensorFlow       | 1.12.0                        |
| Keras            | 2.2.4                         |
| CUDA             | 10.0                          |
| cuDNN            | 7.5                           |

> **Important:** This repository uses **TensorFlow 1.x** and other legacy dependencies. Running the code directly with recent TensorFlow or Python releases may produce compatibility errors.

For reproducibility, using a dedicated Conda environment with Python 3.7 is recommended.

---

## 4. Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/Notmentionedhere/Explainable-federated-reinforcement-learning-in-the-Smart-Grid.git
cd Explainable-federated-reinforcement-learning-in-the-Smart-Grid
```

Alternatively, download the repository as a ZIP file from GitHub and extract it.

### Step 2: Create a Python environment

Anaconda or Miniconda is recommended because the project depends on older versions of Python and TensorFlow.

For example:

```bash
conda create -n frl-esp python=3.7
conda activate frl-esp
```

### Step 3: Install the dependencies

The original Python environment is recorded in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Because the requirements file was generated from the original development environment, it contains some platform-specific package references. Therefore, installation of the complete file may fail on a different operating system or a modern Python installation.

The most important versions used by the original implementation include:

```text
tensorflow==1.12.0
Keras==2.2.4
numpy==1.15.4
scipy==1.2.0
matplotlib==3.0.2
scikit-learn==0.20.2
```

If `pip install -r requirements.txt` fails, create a legacy-compatible environment and install the required packages individually.

---

## 5. Input Data

The repository contains MATLAB (`.mat`), Excel (`.xls`), and NumPy data files used by different experiments.

For example:

```text
pvloaddata.mat
pvloaddata.xls
k_v.mat
lambda.mat
```

The environment code loads data using relative paths, for example:

```python
sio.loadmat('pvloaddata.mat')
```

Therefore, the working directory is important when running the experiments.

For the safest execution, enter the corresponding experiment directory before running `main.py`.

---

# 6. Running the Baseline FRL Experiment

The baseline FRL implementation is located in:

```text
BESS_FRL/
```

First enter the directory:

```bash
cd BESS_FRL
```

Then run:

```bash
python main.py
```

On systems where Python 3 is invoked explicitly:

```bash
python3 main.py
```

The default configuration is defined in `args_init()` in `main.py`.

For example, some of the default parameters are:

```text
Number of agents:        55
State dimension:         3
Actions per agent:       5
Batch size:              240
Learning rate:           0.001
Discount factor (gamma): 0.995
Replay memory size:      100000
Epochs:                  100
Training episodes:       10
Validation episodes:     10
Testing episodes:        10
```

Most parameters can be changed from the command line.

For example:

```bash
python main.py \
    --epochs 50 \
    --train_episodes 20 \
    --test_episodes 20 \
    --learning_rate 0.001 \
    --gamma 0.995
```

To see the available command-line arguments:

```bash
python main.py --help
```

### Important note for `BESS_FRL`

The current `BESS_FRL/main.py` contains:

```python
from utils import get_time, str2bool
```

but `utils.py` is not present in the `BESS_FRL/` directory in the current repository snapshot.

If you receive:

```text
ModuleNotFoundError: No module named 'utils'
```

this missing dependency needs to be restored before the baseline experiment can run. A corresponding `utils.py` exists under `ESP_FRL/`, but users should verify that it is intended to be shared with the baseline implementation before copying or reusing it.

---

# 7. Running the Explainable FRL-ESP Experiment

The explainable implementation is located in:

```text
ESP_FRL/
```

From the repository root:

```bash
cd ESP_FRL
```

Run:

```bash
python main.py
```

or:

```bash
python3 main.py
```

The same general training parameters can be configured through command-line arguments.

For example:

```bash
python main.py \
    --epochs 100 \
    --train_episodes 10 \
    --valid_episodes 10 \
    --test_episodes 10
```

You can inspect all available parameters using:

```bash
python main.py --help
```

---

## 8. Important Training Parameters

The following arguments are defined in `main.py` and can be useful when reproducing or modifying the experiments.

| Argument                    |        Default | Description                                    |
| --------------------------- | -------------: | ---------------------------------------------- |
| `--num_agents`              |             55 | Number of participating agents                 |
| `--state_dim`               |              3 | Dimension of each agent's state                |
| `--hist_len`                |              1 | State-history length                           |
| `--num_action`              |              5 | Number of actions available to each agent      |
| `--batch_size`              |            240 | Training batch size                            |
| `--learning_rate`           |          0.001 | Learning rate                                  |
| `--gamma`                   |          0.995 | Reinforcement-learning discount factor         |
| `--lambda_`                 |            0.5 | Lambda parameter used by the FRL architecture  |
| `--replay_size`             |         100000 | Maximum replay-memory size                     |
| `--epochs`                  |            100 | Maximum number of training epochs              |
| `--train_episodes`          |             10 | Training episodes per epoch                    |
| `--valid_episodes`          |             10 | Validation episodes                            |
| `--test_episodes`           |             10 | Testing episodes                               |
| `--target_steps`            |              5 | Target-network update interval                 |
| `--exploration_rate_start`  |            1.0 | Initial exploration rate                       |
| `--exploration_rate_end`    |            0.1 | Final exploration rate                         |
| `--exploration_decay_steps` |           1000 | Exploration decay period                       |
| `--gpu_fraction`            |            0.8 | Fraction of GPU memory available to TensorFlow |
| `--load_weights`            |          False | Load previously trained model weights          |
| `--save_weights`            |          False | Save trained model weights                     |
| `--train_mode`              | `frl_separate` | FRL training mode                              |
| `--predict_net`             |         `both` | Network used for prediction                    |

The exact behavior of some experimental parameters depends on the selected network and agent implementation.

---

## 9. Saving Model Weights

Model saving is disabled by default:

```text
--save_weights False
```

To enable it:

```bash
python main.py --save_weights True
```

The code saves model weights under the `weights/` directory using names based on the selected training mode and network.

For example, the code may generate a path similar to:

```text
weights/frl_separate_both.h5
```

Make sure that the required output directory exists before training:

```bash
mkdir -p weights
```

---

## 10. Loading Previously Trained Weights

Loading is disabled by default:

```text
--load_weights False
```

It can be enabled with:

```bash
python main.py --load_weights True
```

The expected weight filename depends on the values of:

```text
--train_mode
--predict_net
```

For example:

```bash
python main.py \
    --load_weights True \
    --train_mode frl_separate \
    --predict_net both
```

Check the corresponding `main.py` and network implementation if using custom model names.

---

## 11. Output and Results

During training, the program reports information such as:

* training progress;
* episode rewards;
* validation/test performance;
* state-of-charge information;
* processing time; and
* best observed results.

The training procedure also saves a MATLAB result file:

```text
result_record.mat
```

using:

```python
scipy.io.savemat(
    'result_record.mat',
    {'result_record': result_record1}
)
```

Several example result files are already included in the repository, such as:

```text
result_record0.mat
result_record1.mat
result_record2.mat
result_record3.mat
```

These files can be loaded in MATLAB or Python for further analysis.

For example, in Python:

```python
import scipy.io as sio

results = sio.loadmat("result_record.mat")
result_record = results["result_record"]

print(result_record.shape)
```

---

## 12. Plotting Results

MATLAB scripts are provided for plotting and post-processing some experimental results.

Examples include:

```text
plotreward.m
rewardcal.m
```

To use these files, open MATLAB, navigate to the appropriate experiment directory, and run the desired script.

Some `.mat` result files are also included to support result analysis and visualization.

---

## 13. Main Source Files

### `main.py`

Main experiment entry point. It:

1. defines the experimental parameters;
2. creates the environment;
3. initializes replay memory;
4. creates the FRL/DQN model;
5. creates the reinforcement-learning agents;
6. performs training;
7. performs validation/testing;
8. optionally saves model weights; and
9. records experimental results.

### `agent_branch.py`

Implements the agent-level training and evaluation procedures.

Its responsibilities include interaction with the environment, action selection, experience collection, network training, and result recording.

### `environment_branch.py`

Defines the smart-grid reinforcement-learning environment.

It contains the state representation, battery state-of-charge information, available actions, and environment transitions used during training.

### `deep_q_network_branch55.py`

Contains the DQN/FRL network used by the current `main.py` configuration.

The number `55` corresponds to the configuration designed for multiple agents in the experiment.

### `replay_memory_muti.py`

Implements replay memory for multi-agent reinforcement learning.

Experiences collected during environment interaction are stored and sampled during DQN training.

### `deep_q_network_ESP.py`

Located in `ESP_FRL/`, this file contains code associated with the ESP-based explainable reinforcement-learning implementation.

---

## 14. Typical Workflow

A typical experiment can be performed as follows.

### Baseline FRL

```bash
# Clone repository
git clone https://github.com/Notmentionedhere/Explainable-federated-reinforcement-learning-in-the-Smart-Grid.git

# Enter repository
cd Explainable-federated-reinforcement-learning-in-the-Smart-Grid

# Activate environment
conda activate frl-esp

# Enter baseline experiment
cd BESS_FRL

# Check available parameters
python main.py --help

# Run experiment
python main.py
```

### Explainable FRL-ESP

```bash
# Return to repository root
cd ..

# Enter ESP experiment
cd ESP_FRL

# Check available parameters
python main.py --help

# Run experiment
python main.py
```

For a shorter test run, the number of epochs and episodes can be reduced:

```bash
python main.py \
    --epochs 5 \
    --train_episodes 2 \
    --valid_episodes 2 \
    --test_episodes 2
```

This is useful for checking whether the environment and dependencies have been configured correctly before starting a full experiment.

---

## 15. Troubleshooting

### TensorFlow installation problems

The project was developed using:

```text
TensorFlow 1.12.0
Python 3.7.7
```

These are legacy versions and may not install directly in a modern Python environment.

If you encounter TensorFlow installation errors, use a dedicated legacy Conda environment rather than the system Python installation.

### `ModuleNotFoundError`

Make sure you run the program from the corresponding experiment directory:

```bash
cd ESP_FRL
python main.py
```

rather than running it from an unrelated working directory.

Also note the currently missing `BESS_FRL/utils.py` dependency described above.

### Data file not found

An error such as:

```text
FileNotFoundError: pvloaddata.mat
```

usually indicates that the program was started from the wrong working directory or that the required data file is not present.

Run the experiment from the directory containing the required data:

```bash
cd BESS_FRL
```

or:

```bash
cd ESP_FRL
```

and verify the required `.mat` files.

### CUDA/cuDNN errors

The original GPU environment used:

```text
CUDA 10.0
cuDNN 7.5
```

Newer CUDA versions are not necessarily compatible with TensorFlow 1.12.

If exact GPU reproduction is not required, a CPU-compatible TensorFlow environment may be easier to configure.

### Out-of-memory errors

Reduce the batch size:

```bash
python main.py --batch_size 64
```

instead of the default value of `240`.

The appropriate value depends on the available GPU/CPU memory and experiment configuration.

---


## 16. Notes

This repository contains research code developed for experimental evaluation rather than a production software package.

Several files represent alternative architectures, intermediate experiments, testing utilities, or legacy implementations. Therefore, not every Python or MATLAB file is required for the primary FRL and FRL-ESP experiments.

For most users, the recommended starting points are:

```text
BESS_FRL/main.py
ESP_FRL/main.py
```

Users interested specifically in the explainability component should additionally inspect:

```text
ESP_FRL/deep_q_network_ESP.py
```

