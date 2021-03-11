import logging
import time
from typing import Tuple

import numpy as np
from sqlalchemy import create_engine

import gym
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, LazyFrames

from examples.gym.tf_model import policy_model_2
from examples.gym.gym_problem import GymEnvProblem
from examples.gym.gym_state import GymEnvState
from examples.gym.alphazero_gym import AlphaZeroGymEnv

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# NOTE: These class definitions need to stay outside of construct_problem
# or you will error out on not being able to pickle/serialize them.
class PongEnv(AlphaZeroGymEnv, gym.ObservationWrapper):
    def __init__(self, **kwargs):
        env = gym.envs.make("PongNoFrameskip-v4")
        env = GrayScaleObservation(env) # Turns RGB image to gray scale
        env = ResizeObservation(env, shape=84) # resizes image on a square with side length == shape
        env = FrameStack(env, num_stack=4) # collect num_stack number of frames and feed them to policy network
        super().__init__(env, **kwargs)
    
    def observation(self, obs) -> np.ndarray:
        return 2 * (np.array(LazyFrames(list(obs), self.lz4_compress))/255 - 0.5)

    def get_obs(self) -> np.ndarray:
        return self.observation(self.frames)


class PongProblem(GymEnvProblem):
    def __init__(self, 
                 engine: "sqlalchemy.engine.Engine",
                 **kwargs) -> None:
        env = PongEnv()
        super().__init__(engine, env, **kwargs)

    def policy_model(self) -> "tf.keras.Model":
        obs_shape = self.env.reset().shape
        return policy_model_2(obs_dim = obs_shape,
                              hidden_layers = 1,
                              conv_layers = 3,
                              filters_dim = [32, 64, 64],
                              kernel_dim = [8, 4, 3],
                              strides_dim = [4, 2, 1],
                              hidden_dim = 512,)

    def get_policy_inputs(self, state: GymEnvState) -> dict:
        return {"obs": state.env.get_obs()}

    def get_reward(self, state: GymEnvState) -> Tuple[float, dict]:
        return state.env.cumulative_reward, {}


def construct_problem():

    from rlmolecule.tree_search.reward import RankedRewardFactory

    engine = create_engine(f'sqlite:///pong_data.db',
                           connect_args={'check_same_thread': False},
                           execution_options = {"isolation_level": "AUTOCOMMIT"})

    run_id = "pong_example"

    reward_factory = RankedRewardFactory(
            engine=engine,
            run_id=run_id,
            reward_buffer_min_size=10,
            reward_buffer_max_size=50,
            ranked_reward_alpha=0.75
    )

    problem = PongProblem(
        engine,
        run_id=run_id,
        reward_class=reward_factory,
        min_buffer_size=15,
        policy_checkpoint_dir='pong_policy_checkpoints'
    )

    return problem


def run_games(use_az=True, num_mcts_samples=50):

    if use_az:
        from rlmolecule.alphazero.alphazero import AlphaZero
        game = AlphaZero(construct_problem(), dirichlet_noise=False)
    else:
        from rlmolecule.mcts.mcts import MCTS
        game = MCTS(construct_problem(ranked_reward=False))

    while True:
        path, reward = game.run(num_mcts_samples=num_mcts_samples)

        print(path)

        print("REWARD:", reward.__dict__)
        if use_az:
            logger.info(f'Game Finished -- Reward {reward.raw_reward:.3f} -- Final state {path[-1][0]}')
        

def train_model():
    construct_problem().train_policy_model(steps_per_epoch=100,
                                           game_count_delay=20,
                                           verbose=2)


def monitor():

    from rlmolecule.sql.tables import RewardStore
    problem = construct_problem()

    while True:
        best_reward = problem.session.query(RewardStore) \
            .filter_by(run_id=problem.run_id) \
            .order_by(RewardStore.reward.desc()).first()

        num_games = len(list(problem.iter_recent_games()))

        if hasattr(best_reward, "data") and "position" in best_reward.data:
            print(f"Best Reward: {best_reward.reward:.3f} for final position "
                  f"{best_reward.data['position']} with {num_games} games played")

        time.sleep(5)


if __name__ == "__main__":

    import multiprocessing

    jobs = [multiprocessing.Process(target=monitor)]
    jobs[0].start()
    time.sleep(1)

    for i in range(5):
        jobs += [multiprocessing.Process(target=run_games)]

    jobs += [multiprocessing.Process(target=train_model)]

    for job in jobs[1:]:
        job.start()

    for job in jobs:
        job.join(300)

