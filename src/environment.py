from __future__ import annotations
from typing import Tuple
import neat, gym, os, time, uuid, csv
from gym.core import Wrapper, ObservationWrapper
from gym.spaces import Box
import numpy as np
from numpy import array
import pandas as pd
import pickle as pkl
import multiprocessing
from multiprocessing import Pool


class BreakoutPreprocessor(ObservationWrapper):
    """Wrapper for processing Breakout RAM observations.
    
    Attributes:
        ram_dims (int): The dimensions of the RAM used.
        observation_space (Box): A Gym Box object of the observation space.
    """
    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): The Gym Breakout environment, which
                must contain obs_type="ram".
        """
        ObservationWrapper.__init__(self, env)
        self.ram_dims = 3
        # Limit observation space to within 0-1, and according to number of RAM bytes
        low = np.zeros(self.ram_dims, 'float32')
        high = np.zeros(self.ram_dims, 'float32')
        high.fill(1)
        self.observation_space = Box(low, high)

    def normalization(self, val: int, centering: int, max: int) -> float:
        """Normalization by centering to 1 and dividing by a normalization factor
        
        Args:
            centering (int): The to subtract for centering.
            max (int): The max possible value of the variable, used for normalization.
        
        Returns:
            normalized_val (float): The normalized value.
        """
        
        return float((val - centering + 1) / (max-centering + 1))
    
    def observation(self, ram_obs: array) -> array:
        """Apply normalization to RAM values.
        
        Args:
            ram_obs (array): The RAM observation.
        
        Returns:
            ram_obs (array): The normalized RAM observation. 
        """
        # Retrieve and normalize the needed RAM adresses
        paddle_x = self.normalization(ram_obs[72], 55, 191)
        ball_x   = self.normalization(ram_obs[99], 0, 255)
        ball_y   = self.normalization(ram_obs[101], 0, 255)
        
        return np.array([paddle_x, ball_x, ball_y])

class BreakoutFrameBuffer(Wrapper):
    """Wrapper for creating a n-frame buffer of RAM observations.
    
    Attributes:
        observation_space (Box): A Gym Box object of the observation space.
        buffer (array): A 2-dimensional numpy array containing the current
            buffer state. Dimensions are equal to (frames, ram dimensions).
    """
    def __init__(self, env: gym.Env, frames: int=4):
        """
        Args:
            env (gym.Env): The Gym Breakout environment, which
                must contain obs_type="ram".
            frames (int): Number of frames to store in the buffer.
                (default is 4)
        """
        super(BreakoutFrameBuffer, self).__init__(env)
        # Define observation space
        ram_dims = env.observation_space.shape[0]
        self.observation_space = Box(0.0, 1.0, (frames, ram_dims))
        # Setup frame buffer
        self.buffer = np.zeros([frames, ram_dims], 'float32')
        
    def reset(self) -> array:
        """Reset the environment, including the buffer.
        
        Returns:
            buffer (array): The values contained in the buffer, corresponding
                to the observation_space of the environment.
        """
        # Reset buffer and add first frame RAM data
        self.buffer = np.zeros_like(self.buffer)
        self.update_buffer(self.env.reset())
        
        return self.buffer
    
    def step(self, action: int) -> Tuple[array, float, bool, dict]:
        """Perform a step in the environment and return results.
        
        Args:
            action (int): The action to perform.
        
        Returns:
            buffer (array): The values contained in the buffer, corresponding
                to the observation_space of the environment.
            reward (float): The environment reward.
            done (bool): Boolean indicating whether the environment is finished.
            info (dict): Various info related to the specific environment.
        """
        observation, reward, done, info = self.env.step(action)
        self.update_buffer(observation)
        
        return self.buffer, reward, done, info
    
    def update_buffer(self, ram_obs: array):
        """Remove the last frame of RAM observations from the buffer
        and prepend a new observation.
        
        Args:
            ram_obs (array): The RAM observation to prepend.
        """
        # Remove last RAM observation and append the new one
        cropped_buffer = self.buffer[:-1]
        self.buffer = np.vstack([ram_obs, cropped_buffer])

class BreakoutNEAT(Wrapper):
    """Wrapper class for binding together NEAT and Atari Breakout with
    RAM-based training.
    
    Attributes:
        env_id (str): A unique identifier for the environment. Used for logging.
        config (neat.Config): The NEAT configuration settings.
        pool (neat.Pool): The NEAT pool object, which contains all species, genomes, etc.
        stats (neat.StatisticsReporter): Statistics about the NEAT run.
        paths (dict): Various paths used for logging.
        
        Also extends attributes and methods from Gym environments.
    """
    def __init__(
        self, env: gym.Env, config_path: str,
        from_checkpoint: Tuple[str, int]=None,
        checkpoint_interval: int=10):
        """
        Args:
            env (gym.Env): The Gym Breakout environment, which
                must contain obs_type="ram".
            config_path (str): The path to the config.ini file used by NEAT.
                Read more here: https://neat-python.readthedocs.io/en/latest/config_file.html
            from_checkpoint (Tuple[str, int]): If not none, loads existing environment from checkpoint
                by providing a tuple with the env ID at position 0 and the checkpoint at position 1.
            deterministic (bool): Set whether sticky actions should be used (default is True).
                Read more here: https://www.gymlibrary.ml/environments/atari/#stochasticity
            checkpoint_interval (int): If not None, determines the interval of storing NEAT
                generation checkpoints (default is 10).
                Read more here: https://neat-python.readthedocs.io/en/latest/_modules/checkpoint.html
        """
        super(BreakoutNEAT, self).__init__(env)

        # Create a unique ID for the environment
        if from_checkpoint is None:
            self.env_id = str(uuid.uuid4())
        else:
            self.env_id = from_checkpoint[0]
        print(f"NEAT Atari Breakout environment (ID: {self.env_id})")
        
        # Setup NEAT
        self.config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_path)
        
        if from_checkpoint is None:
            self.pool = neat.Population(self.config)
        else:
            self.pool = neat.Checkpointer.restore_checkpoint(
                os.path.join('logs', self.env_id, 'checkpoints', f'neat-checkpoint-{from_checkpoint[1]}'))
        
        # Add NEAT reporter objects
        self.pool.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.pool.add_reporter(self.stats)
        
        # Setup folder structure for logging
        self.env_path = os.path.join('logs', self.env_id)
        checkpoints_path = os.path.join(self.env_path, 'checkpoints')

        # Create subfolders and files if they don't already exist
        if not os.path.exists(self.env_path):
            os.makedirs(self.env_path)
            os.makedirs(checkpoints_path)
            with open(os.path.join(self.env_path, 'fitness_scores.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i for i in range(self.pool.config.pop_size)])
                f.close()
        
        # Setup NEAT checkpoints (basically just pickle objects of the env pool object)
        if checkpoint_interval is not None:
            self.pool.add_reporter(
                neat.Checkpointer(
                    checkpoint_interval,
                    filename_prefix=os.path.join(checkpoints_path, 'neat-checkpoint-')))
        
    def action(self, prediction: array) -> int:
        """Returns the highest activation node as a mapping to the correct game input.
        
        Args:
            prediction (array): The network prediction array.
        
        Returns:
            action (int): The action to perform.
        """
        # Get the highest value of the output layer.
        action = np.array(prediction).argmax()
        # Button 0 is "no action", 2 is "left", and 3 is "right".
        # We don't include button 1 ("fire"), as it does nothing.
        mapping = {0:0, 1:2, 2:3}
        action = mapping[action]
        
        return action
    
    def evaluate_genome(self, genome: neat.DefaultGenome, max_steps: int=50000, lives: int=1) -> float:
        """Evaluate a single genome by playing Breakout.
        
        Args:
            genome (neat.DefaultGenome): The genome to evaluate.
            max_steps (int): The maximum number of allowed steps (default is 50k).
            lives (int): How many lives to evaluate for.
        
        Returns:
            fitness (float): The fitness score acheived by the genome.
        """
        # Prepare fitness, environment, and the phenotype/network
        fitness = 0.0
        network = neat.nn.FeedForwardNetwork.create(genome, self.config)
        observation = self.reset()
        
        # To start the game, the "fire" button must be pressed.
        observation, reward, done, info = self.step(1)
        life = info["lives"]
        
        # Run the environment
        for t in range(max_steps):
            # Start the game after dying
            if life != info["lives"]:
                life = info["lives"]
                observation, reward, done, info = self.step(1)
            
            # This conditional ends evluation according to the lives argument.
            if abs(life - 5) == lives:
                break
            
            # Process the observation, feed it to the network,
            # and map the action to the game
            observation = observation.flatten()
            prediction = network.activate(observation)
            action = self.action(prediction)
            # Perform a game step and update fitness
            observation, reward, done, info = self.step(action)
            fitness += reward
        
        return fitness
    
    def fitness_function(self, genomes: list[neat.DefaultGenome], parallelize: bool=True):
        """The fitness function used by neat.pool.run() to score (all) genomes in a generation.
        
        Args:
            genomes (list[neat.DefaultGenome]): A list of neat genomes to be evaluated.
            parallelize (bool): Indicates whether to use multiple CPU cores during training.
                (default is True).
        """
        
        genomes_list = [genome for _, genome in genomes]
        # Evaluate using multiprocessing.map()
        if parallelize:
            p = Pool(processes=os.cpu_count())
            fitness_scores = p.map(self.evaluate_genome, genomes_list)
        # Or just regular training
        else:
            fitness_scores = [self.evaluate_genome(genome) for genome in genomes_list]
        # Update the fitness scores with the results from the evaluation.
        for i, genome in enumerate(genomes_list):
                genome.fitness = fitness_scores[i]
        
        # Write fitness scores to the .csv
        with open(os.path.join(self.env_path, 'fitness_scores.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fitness_scores)
            f.close()
    
    def train(self, generations=10, parallelize=True):
        """Perform NEAT evolutionary training for a set number of generations.
        
        Args:
            generations (int): The number of generations to train for (default is 10).
            parallelize (bool): Indicates whether to use multiple CPU cores during training.
                (default is True).
        """
        # The pool function requires a function with only 2 inputs (genomes, config),
        # so we use the lambda function as a little trick to give the function what it
        # wants. Since this is inside a class that requires the self input parameter,
        # it wouldn't work otherwise.
        self.best_genome = self.pool.run(
            lambda genomes, config : self.fitness_function(genomes, parallelize),
            generations)


def make_neat_breakout(
    config_path: str, from_checkpoint: Tuple[str, int]=None,
    deterministic: bool=True, checkpoint_interval: int=10) -> BreakoutNEAT:
    """Pipeline for creating a BreakoutNEAT object.
    
    Args:
        config_path (str): The path to the config.ini file used by NEAT.
            Read more here: https://neat-python.readthedocs.io/en/latest/config_file.html
        from_checkpoint (Tuple[str, int]): If not none, loads existing environment from checkpoint
            by providing a tuple with the env ID at position 0 and the checkpoint at position 1.
        deterministic (bool): Set whether sticky actions should be used (default is True).
            Read more here: https://www.gymlibrary.ml/environments/atari/#stochasticity
        checkpoint_interval (int): If not None, determines the interval of storing NEAT
            generation checkpoints (default is 10).
            Read more here: https://neat-python.readthedocs.io/en/latest/_modules/checkpoint.html
    
    Returns:
        env (Gym.Env): The Breakout NEAT environment.
    """
    env = gym.make(
        "ALE/Breakout-v5",
        repeat_action_probability=0.0 if deterministic else 0.25,
        frameskip=4 if deterministic else 5,
        full_action_space=False,
        obs_type='ram')
    env = BreakoutPreprocessor(env)
    env = BreakoutFrameBuffer(env)
    env = BreakoutNEAT(
        env, config_path=config_path,
        from_checkpoint=from_checkpoint,
        checkpoint_interval=checkpoint_interval)
    
    return env
