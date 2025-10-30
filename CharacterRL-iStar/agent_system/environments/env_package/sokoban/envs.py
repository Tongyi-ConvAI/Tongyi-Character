import torch.multiprocessing as mp
import gym
from agent_system.environments.env_package.sokoban.sokoban import SokobanEnv
import numpy as np
import sys

def _worker(remote, mode, env_kwargs):
    """
    Core loop for each subprocess. 
    Each subprocess holds its own independent instance of SokobanEnv.
    It receives instructions (cmd, data) from the main process,
    executes the corresponding environment operations, and sends back the result.
    """
    env = SokobanEnv(mode, **env_kwargs)

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            action = data
            obs, reward, done, info = env.step(action)

            remote.send((obs, reward, done, info))

        elif cmd == 'reset':
            seed_for_reset = data
            obs, info = env.reset(seed=seed_for_reset)

            remote.send((obs, info))

        elif cmd == 'render':
            mode_for_render = data
            rendered = env.render(mode=mode_for_render)
            remote.send(rendered)

        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class SokobanMultiProcessEnv(gym.Env):
    """
    Multi-process wrapper for the Sokoban environment.
    Each subprocess creates an independent SokobanEnv instance.
    The main process communicates with subprocesses via Pipe to collect step/reset results.
    """

    def __init__(self,
                 seed=0, 
                 env_num=1, 
                 group_n=1, 
                 mode='rgb_array',
                 is_train=True,
                 env_kwargs=None):
        """
        - env_num: Number of different environments
        - group_n: Number of same environments in each group (for GRPO and GiGPO)
        - env_kwargs: Dictionary of parameters for initializing SokobanEnv
        - seed: Random seed for reproducibility
        """
        super().__init__()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.mode = mode
        np.random.seed(seed)

        if env_kwargs is None:
            env_kwargs = {}


        self.parent_remotes = []
        self.workers = []

        if sys.platform.startswith("win"):
            ctx = mp.get_context('spawn')
        else:
            ctx = mp.get_context('fork')

        for i in range(self.num_processes):
            parent_remote, child_remote = mp.Pipe()
            worker = ctx.Process(
                target=_worker,
                args=(child_remote, self.mode, env_kwargs)
            )
            worker.daemon = True
            worker.start()
            child_remote.close()

            self.parent_remotes.append(parent_remote)
            self.workers.append(worker)

    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list[int], length must match self.num_processes
        :return:
            obs_list, reward_list, done_list, info_list
            Each is a list of length self.num_processes
        """
        assert len(actions) == self.num_processes

        for remote, action in zip(self.parent_remotes, actions):
            remote.send(('step', action))

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for remote in self.parent_remotes:
            obs, reward, done, info = remote.recv()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Perform reset in parallel.
        :return: obs_list and info_list, the initial observations for each environment
        """
        # randomly generate self.env_num seeds
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # repeat the seeds for each group
        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()
        for i, remote in enumerate(self.parent_remotes):
            remote.send(('reset', seeds[i]))

        obs_list = []
        info_list = []
        for remote in self.parent_remotes:
            obs, info = remote.recv()
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def render(self, mode='rgb_array', env_idx=None):
        """
        Request rendering from subprocess environments.
        Can specify env_idx to get render result from a specific environment,
        otherwise returns a list from all environments.
        """
        if env_idx is not None:
            self.parent_remotes[env_idx].send(('render', mode))
            return self.parent_remotes[env_idx].recv()
        else:
            for remote in self.parent_remotes:
                remote.send(('render', mode))
            results = [remote.recv() for remote in self.parent_remotes]
            return results

    def close(self):
        """
        Close all subprocesses
        """
        for remote in self.parent_remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()

    def __del__(self):
        self.close()


def build_sokoban_envs(
        seed=0,
        env_num=1,
        group_n=1,
        mode='rgb_array',
        is_train=True,
        env_kwargs=None):
    return SokobanMultiProcessEnv(seed, env_num, group_n, mode, is_train, env_kwargs=env_kwargs)