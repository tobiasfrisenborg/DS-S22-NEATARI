# NEATARI - Learning to play Atari Breakout from RAM using NeuroEvolution of Augmenting Topolgies
*Created by Tobias Frisenborg Christensen (Student ID: 201806880)*  
  
This repository contains the code for my Data Science exam project for the spring semester of 2022 on the Masters in Cognitive Science. More specifically, it uses NEAT (Stanley & Miikkulainen, 2002) to train an agent capable of playing the classic Atari arcade game "Breakout" using RAM-values as the only input.  
The overall steps to acheive this are as follows:
* Normalization of RAM values to limit the possible input sizes.
* Creating a buffer of 4 frames. The agent needs information about the velocity and direction of the game objects. The Atari 2600 had 128 bytes of RAM, which yields a total number of inputs of 512 (4*128).
* Training NEAT for a set number of durations and logging the results. The built-in rewards from OpenAI's Gym environment is used for the fitness function.

# Repository structure
The main code is placed in the `breakout-neat.ipynb` notebook. The notebook imports wrapper classes from `/src/environment.py`, which are used to instantiate an object that binds together NEAT and Atari Breakout, along with relevant preprocessing, logging, and other utility functionality. The configuration settings for NEAT are specified in the `neat_config.ini` file - this includes important parameters such as the number of input- and output nodes, as well as many hyperparameters. The `visualizations` folder contain simple R and Python scripts for creating the visualiations used in the paper, some additional processing in Photoshop, and the plots themselves. Upon running the main notebook, a `logs` folder will be created containing data relevant to the specific run.

# Setting up an environment
*Note that the `python` and `pip` commands should be replaced with `python3` and `pip3` respectively on MacOS - depending on the installation type.*
1. Clone the repository to a desired location and navigate to it in a CLI.
2. Create a python virtual environment: ```pyton -m venv .neatari_venv```
3. Activate the virtual environment: (on Windows: `.neatari_venv\Scripts\Activate.ps1`, and on MacOS: `source .neatari_venv/bin/activate`)
4. Install requirements: `pip install -r requirements.txt`

**The requirements for this repository include `accept-rom-license`, meaning that you accept the license associated with Atari environments and [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment).**

# References
Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting Topologies. Evolutionary Computation, 10(2), 99–127. https://doi.org/10.1162/106365602320169811

**Packages:**  
[neat-python](https://github.com/CodeReclaimers/neat-python)  
[OpenAI Gym](https://github.com/openai/gym)  
[The Atari Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)  

