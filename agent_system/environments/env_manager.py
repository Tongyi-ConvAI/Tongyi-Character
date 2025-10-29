import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.prompts import *


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, env_name):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.buffers = None
        super().__init__(envs, projection_f, env_name)

    def reset(self):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(infos))]
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.save_to_history_buffer(self.pre_text_obs, actions)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.save_to_history_buffer(self.pre_text_obs, actions)
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if init or history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    if self.is_multi_modal:
                        action_history += f"\n[Action {step_number}: '{record['action']}']"
                    else:
                        action_history += f"\n[Text Observation {step_number}: \n{record['text_obs']}\nAction {step_number}: '{record['action']}']"

                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': self.ACTION_LOOKUP[actions[i]]})


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name):
        self.buffers = None
        super().__init__(envs, projection_f, env_name)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(infos))]
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.save_to_history_buffer(self.pre_text_obs, actions)
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions

    def save_to_history_buffer(self, text_obs, actions):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.buffers[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.buffers[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                # TODO Check in Sotopia
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return


from .env_package.sotopia import SotopiaEnv
class SotopiaEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, 
                 judge_model: str = "gpt-4o",
                 opponent_model: str = "gpt-3.5-turbo", 
                 max_turns: int = 20,
                 actor_role: str = "agent1",
                 **kwargs):
        super().__init__()
        
        self.judge_model = judge_model
        self.opponent_model = opponent_model
        self.max_turns = max_turns
        self.actor_role = actor_role
        self.env_kwargs = kwargs
        
    def make_env(self, env_id: int = 0) -> SotopiaEnv:
        """Create a Sotopia environment instance"""
        return SotopiaEnv(
            judge_model=self.judge_model,
            opponent_model_name=self.opponent_model,
            max_turns=self.max_turns,
            actor_role=self.actor_role,
            **self.env_kwargs
        )
    
    def get_prompt(self, observation, **kwargs) -> str:
        """Convert observation to prompt for the language model"""
        if hasattr(observation, 'current_prompt'):
            return observation.current_prompt
        else:
            # Fallback for string observations#
            return str(observation)
    
    def parse_action(self, action_str: str) -> str:
        """Parse action string to extract the actual response"""
        # Simple parsing - in practice might want more sophisticated extraction
        if isinstance(action_str, str):
            # Remove any action tags or formatting
            action_str = action_str.strip()
            if action_str.startswith("Response:"):
                action_str = action_str[9:].strip()
            return action_str
        return str(action_str)
    
    def is_terminal(self, observation) -> bool:
        """Check if the episode has terminated"""
        if hasattr(observation, 'is_done'):
            return observation.is_done
        return False
    
    def get_reward_function(self):
        """Get reward function for Sotopia environment"""
        # Reward is computed by the judge model at episode end
        return None  # Environment handles reward internally
    
    def get_success_rate(self, episode_data) -> float:
        """Calculate success rate based on judge evaluation"""
        if 'reward' in episode_data and episode_data['reward'] > 0.6:
            return 1.0
        return 0.0


def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import (
            build_sokoban_envs, sokoban_projection)
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import (
            build_webshop_envs, webshop_projection)
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "sotopia" in config.env.env_name.lower():
        from agent_system.environments.env_package.sotopia import build_sotopia_envs
        _envs = build_sotopia_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_sotopia_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)

        envs = SotopiaEnvironmentManager(_envs, config.env.env_name)
        val_envs = SotopiaEnvironmentManager(_val_envs, config.env.env_name)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)


if __name__ == "__main__":
    env_name = "webshop"
    # Test WebshopEnvironmentManager
    import time

    from agent_system.environments.env_package.webshop import (
        build_webshop_envs, webshop_projection)
    from agent_system.environments.env_package.webshop.webshop.web_agent_site.models import \
        RandomPolicy
    env_num = 2
    group_n = 5
    time1 = time.time()
    file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
    attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
    env_kwargs = {
                'observation_mode': 'text', 
                'num_products': None, 
                'human_goals': True,
                'file_path': file_path,
                'attr_path': attr_path
                }
    envs = build_webshop_envs(seed=1, env_num=env_num, group_n=group_n, env_kwargs=env_kwargs, is_train=True)
    # val_envs = build_webshop_envs(1000, 4)
    env_manager = WebshopEnvironmentManager(envs, webshop_projection, 'webshop')
    policy = RandomPolicy()
    time2 = time.time()
    print(f"env_num: {env_num}, group_n: {group_n}, init time: ", time2 - time1)
    # val_env_manager = AlfWorldEnvironmentManager(val_envs, alfworld_projection, 'alfworld/AlfredTWEnv')
    for k in range(10):
        time1 = time.time()
        obs, infos = env_manager.reset()
        for i in range(20):
            # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
            print("step: ", i)
            random_actions = ['<action>'+policy.forward(None, info['available_actions'])+'</action>' for info in infos]
            # step
            obs, rewards, dones, infos = env_manager.step(random_actions)
            if np.array(dones).any():
                print("Episode completed")

            if obs['image'] is not None:
                env_manager.save_image(obs['image'], i)
            # print("obs['image'].shape: ", obs['image'].shape)
        time2 = time.time()
        print(f"env_num: {env_num}, group_n: {group_n}, Time elapsed: ", time2 - time1)
    print("completed")