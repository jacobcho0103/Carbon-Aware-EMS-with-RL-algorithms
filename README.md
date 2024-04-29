# Carbon-Aware Energy Management System with Reinforcement Learning

This repository contains a collection of components designed for reinforcement learning (RL) experiments in an energy management domain. The project is structured to facilitate the development, training, and evaluation of RL agents within a custom energy management environment. Below is an overview of the primary components and their functionalities.

## Data Inputs

- `Data_input_PV.csv`: Core datasets designed for reinforcement learning experiments within the energy management domain. These files contain time-series data essential for simulating and navigating energy system complexities. The dataset includes the following columns:

  - **Demand**
  - **Price**
  - **PV Generation**
  - **Day**
  - **Month**
  - **Hour**

## RL Monitoring Utilities

- `plotting.py`: A utility script for visualizing training metrics, including rewards, losses, and other relevant performance indicators over time.
- `reward_callback.py`: Contains callback functions designed to compute or adjust rewards based on specific conditions during the training process.
- `vec_monitor.py`: Implements monitoring and logging for vectorized environments, facilitating detailed analysis of agent performance and environment states.

## Agents

- `agent.py`: Defines the RL agent's architecture, including the policy network, learning algorithms, and any additional logic for decision-making.
- `agents_main.py`: The main script used to kickstart the training or evaluation process. It orchestrates the interaction between agents and the environment, handling the training loop, logging, and any necessary pre/post-processing.

## Environment

- `Basic_energy_management_env.py`: Implements the basic energy management environment. It simulates the dynamics of energy systems, handling state transitions, reward calculations, and any interactions an agent can have within this domain.
- `C_energy_management_env.py`: Implements the carbon-aware energy management environment. It is the extension of the basic EMS, which can consider the carbon emission costs as part of reward functions.
- `env_registration.py`: Facilitates the registration of the custom environment with the RL framework being used, ensuring compatibility and ease of use within standard training routines.

## Getting Started

To get started with training an agent in the energy management environment, ensure you have the required dependencies installed. Then, you can run the `BasicEMS_agents_main.py` and `CarbonAwareEMS_agents_main.py` scripts to begin the training process. Make sure to adjust the data input paths and any hyperparameters according to your experimental setup.
