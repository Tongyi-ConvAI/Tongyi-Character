import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ControlLLM import ControlLLM
from typing import Literal
from sotopia_utils.utils import ScriptBackground, EnvResponse, get_bio, format_bad_output
from langchain.output_parsers import PydanticOutputParser
import random
from openai import OpenAI
import os
import numpy as np
import re
import json
import pickle
import time
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu

current_file_path = os.path.abspath(__file__)
directory = os.path.dirname(current_file_path)


class SotopiaEnv():
    def __init__(self, model_name: str,
                env_model: str = "gpt-4",
                opponent_model: str = "",  #"meta-llama/Meta-Llama-3-8B-Instruct",
                prefix_size: int = 8,
                prefix_embedding_size: int = 64, 
                max_turns: int = 20,
                temperature: float = 0.7,
                prefix_pos: Literal ['start', 'mid', 'end'] = 'start',
                actor_role: Literal ['agent1', 'agent2'] = 'agent1',
                test_baseline: bool = False,
                test_gpt: bool = False,
                saving_dir: str = None
                ):
        # super(self).__init__()
        if opponent_model == "":
            self.reuse_model = True
        else:
            self.reuse_model = False
            self.opponent_model = AutoModelForCausalLM.from_pretrained(opponent_model, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map="auto")
            # self.opponent_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            self.opponent_tokenizer = AutoTokenizer.from_pretrained(opponent_model)
        self.model = ControlLLM(
            model_name,
            prefix_size,
            prefix_embedding_size,
            prefix_pos
        )

        self.env_model = env_model
        self.cur_turn = 0
        self.max_turns = max_turns
        self.temperature = temperature
        self.test_baseline = test_baseline
        self.test_gpt = test_gpt
        self.agent1_intro = ''
        self.agent2_intro = ''
        self.complete_intro = ''
        self.cur_dialog = []
        self.cur_rewards = []
        self.actions = []
        self.cur_states = []
        self.next_states = []
        self.terminations = []
        self.p1_scores = None
        self.p2_scores = None
        self.agent1_name = ''
        self.agent2_name = ''
        self.additional_instruction = "You can say something to interact or just say 'left the conversation' to stop continuing.\n"


        self.load_scenarios()
        self.actor_role = actor_role

        self.saving_dir = saving_dir

    def load_scenarios(self):
        with open(os.path.join(directory, 'sotopia_utils/sotopia_data/env_agent_combos.json'), 'r') as f:
            self.env_agent_combos = json.load(f)

        with open(os.path.join(directory, 'sotopia_utils/sotopia_data/envs.json'), 'r') as f:
            self.envs = json.load(f)

        with open(os.path.join(directory, 'sotopia_utils/sotopia_data/agents.json'), 'r') as f:
            self.agent_profiles = json.load(f)

    def get_current_prompt(self):
        # accumulate the current dialog into a prompt
        prompt = "\n".join(self.cur_dialog)
        prompt = self.agent1_intro + prompt if self.cur_turn % 2 == 0 else self.agent2_intro + prompt
        return prompt

    def get_current_input_tensors(self):
        # get the input_ids of the current prompt
        prompt = self.get_current_prompt()
        tensors = self.model.tokenizer(prompt, return_tensors='pt')

        return tensors['input_ids'].cuda(), tensors['attention_mask'].cuda()

    def get_state(self):
        # return the current state of the environment
        # use base model's forward method to get the hidden state of the last layer of the last token of the input
        input_ids, attention_mask = self.get_current_input_tensors()
        with torch.no_grad():
            output = self.model.base_model(
                attention_mask=attention_mask,
                input_ids=input_ids,
                output_hidden_states=True,
            )
        hidden_states = output.hidden_states
        last_hidden_state = hidden_states[-1][:, -1, :].to(torch.float32)

        return last_hidden_state
    
    def _clean_tags(self, text):
        # use regex to remove the tags in the text
        return re.sub(r'<.*?>', '', text)
    
    def _mask_intro(self, intro, role: int):
        # Mask the opponent's goal and secret in the intro
        opponent_name = self.agent2_name if role == 0 else self.agent1_name
        # mask secrets
        if f"<p viewer='agent_{1-role}'>" in intro:
            # replace the text contained in the tag
            first_name, last_name = opponent_name.split()
            intro = re.sub(fr"<p viewer='agent_{1-role}'>.*?<\/p>", "", intro)

        # mask goal
        intro = re.sub(fr"{opponent_name}'s goal: .*?(\n|$)", f"{opponent_name}'s goal: Unknown\n", intro)

        return intro
    
    def _judge_terminate(self) -> bool:
        # 1. exceeds the max_turns
        if self.cur_turn >= self.max_turns:
            return True
        elif any([_ in self.cur_dialog[-1].lower() for _ in ["left", "leave", "bye", "have a good"]]):
            return True
        else:
            return False
        
    def save_conversation_history(self, seed):
        # the seed is how the action is randomized in the actor not how the engine works
        # convert to numpy arrays
        self.cur_rewards = np.array(self.cur_rewards)
        self.cur_states = np.array(self.cur_states)
        self.actions = np.array(self.actions)
        self.next_states = np.array(self.next_states)
        self.terminations = np.array(self.terminations)

        tbd = dict(
            complete_intro=self.complete_intro, 
            dialog=self.cur_dialog, 
            rewards=self.cur_rewards,
            observations=self.cur_states,
            actions=self.actions,
            next_observations=self.next_states,
            terminals=self.terminations,
            agent1_scores=self.p1_scores,
            agent2_scores=self.p2_scores,
        )
        with open(os.path.join(self.saving_dir, f"env_id={self.env_id}_role={self.actor_role}_seed={seed}.pkl"), 'wb') as f:
            pickle.dump(tbd, f)

    def reset(self, env_id = None, actor_role = "agent1"):
        # Reset the environment and return the initial state
        self.cur_dialog = []
        self.cur_rewards = []
        self.cur_states = []
        self.actions = []
        self.next_states = []
        self.terminations = []

        # Sample a new scenario
        self.actor_role = actor_role
        if env_id is None:
            env_id = random.randint(0, 449)

        env_agent_combo_storage = self.env_agent_combos[env_id]
        self.env_id = env_id
        env_id = env_agent_combo_storage['env_id']
        
        env_profile = self.envs[env_id]

        agent_ids = env_agent_combo_storage['agent_ids']
        agent_profiles = [self.agent_profiles[id] for id in agent_ids]

        self.agent1_name = agent_profiles[0]['first_name'] + " " + agent_profiles[0]['last_name']
        self.agent2_name = agent_profiles[1]['first_name'] + " " + agent_profiles[1]['last_name']

        background = ScriptBackground(
            scenario=env_profile['scenario'],
            p1_background=get_bio(
                env_profile['relationship'],
                agent_profiles[0],
                agent_id=0,
            ),
            p2_background=get_bio(
                env_profile['relationship'],
                agent_profiles[1],
                agent_id=1,
            ),
            p1_goal=f"{env_profile['agent_goals'][0]}",
            p2_goal=f"{env_profile['agent_goals'][1]}",
            p1_name=self.agent1_name,
            p2_name=self.agent2_name,
        )

        # form the prompt
        # TODO: hide goal of the opponent
        self.agent1_intro = self._mask_intro(background.to_natural_language(), 0) + self.additional_instruction
        self.agent2_intro = self._mask_intro(background.to_natural_language(), 1) + self.additional_instruction

        self.complete_intro = self._clean_tags(background.to_natural_language())
        self.agent1_intro = self._clean_tags(self.agent1_intro)
        self.agent2_intro = self._clean_tags(self.agent2_intro)

        self.cur_turn = 0
        self.cur_dialog.append(f"Turn {self.cur_turn}: {self.agent1_name} said:")

        if self.actor_role == 'agent2':
            self._act(action=None, actor=False)
            self.cur_dialog.append(f"Turn {self.cur_turn}: {self.agent2_name} said:")

        # get the state of the prompt
        state = self.get_state()

        return state
    
    def get_final_reward(self):
        template = """{history}{schema}"""
        history = self.complete_intro + "\n".join(self.cur_dialog)
        with open(os.path.join(directory, 'sotopia_utils/output_schema.txt'), 'r') as f:
            schema = f.read()

        prompt = template.format(history=history, schema=schema)

        # call self.env_model to get the reward
        client = OpenAI(
            api_key = os.environ.get("OPENAI_API_KEY")
        )
        
        response = client.chat.completions.create(
            model=self.env_model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        output_text = response.choices[0].message.content

        # parse output_text to get the reward
        output_parser = PydanticOutputParser[EnvResponse](
            pydantic_object=EnvResponse
        )
        try:
            parsed_result = output_parser.parse(output_text)
    
        except Exception as e:
            
            reformat_parsed_result = format_bad_output(
                output_text, format_instructions=output_parser.get_format_instructions()
            )
            parsed_result = output_parser.parse(reformat_parsed_result)

        d = parsed_result.agent_1_evaluation.dict() if self.actor_role == 'agent1' else parsed_result.agent_2_evaluation.dict()
        overall_score = 0
        for dimension in d:
            overall_score += d[dimension][1]

        overall_score /= len(d)
        return overall_score, parsed_result.agent_1_evaluation.dict(), parsed_result.agent_2_evaluation.dict()
    
    def _act(self, action, actor: bool = False):
        with torch.no_grad():
            # upmapping the action to get input embeddings
            input_ids, attention_mask = self.get_current_input_tensors()

            input_ids = input_ids.to(self.model.base_model.device)
            if actor:  # time for the controlled LM to talk
                if (not self.test_baseline):  # time for the controlled LM to act
                    new_embeddings = self.model.embed_action(action, input_ids).to(dtype=torch.bfloat16)
                    output = self.model.base_model.generate(
                        inputs_embeds=new_embeddings,
                        temperature=self.temperature,
                    )
                    result = self.model.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                else:  # time for the base model to act
                    if (not self.test_gpt):  # base model is the controlled LM
                        output = self.model.base_model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            temperature=self.temperature,
                        )
                        result = self.model.tokenizer.batch_decode(output, skip_special_tokens=True)[0][len(self.get_current_prompt()):]
                    else:  # base model is the GPT model
                        prompt = self.get_current_prompt()
                        # call self.env_model to get the reward
                        client = OpenAI(
                            api_key = os.environ.get("OPENAI_API_KEY")
                        )
                        
                        response = client.chat.completions.create(
                            model=self.env_model,
                            messages=[
                                {
                                    'role': 'user',
                                    'content': prompt,
                                }
                            ],
                            temperature=1.0,  # Default value
                            top_p=1.0,  # Default value, no nucleus sampling
                            frequency_penalty=0.0,  # Default value
                            presence_penalty=0.0  # Default value
                        )

                        result = response.choices[0].message.content
            else:  # time for the base model to talk
                if self.reuse_model:  # time for the base model to act
                    output = self.model.base_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        temperature=self.temperature,
                    )
                    result = self.model.tokenizer.batch_decode(output, skip_special_tokens=True)[0][len(self.get_current_prompt()):]
                else:  # time for the opponent model to act
                    output = self.opponent_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        temperature=self.temperature,
                    )
                    result = self.opponent_tokenizer.batch_decode(output, skip_special_tokens=True)[0][len(self.get_current_prompt()):]

        # TODO: add stopping token
        # Currently: only get the first line.
        result = result.split('\n')[0]

        # update the dialog
        self.cur_turn += 1
        self.cur_dialog[-1] += result + '\n'
        

    def step(self, cur_state, action):
        self.cur_states.append(cur_state.detach().cpu().numpy())
        self.actions.append(action.detach().cpu().numpy())
        # take an action
        self._act(action, actor=True) if self.actor_role == 'agent1' else self._act(action, actor=False)

        # judge whether terminate or not
        done = self._judge_terminate()

        if done:
            reward, p1_scores, p2_scores = self.get_final_reward()
            self.cur_rewards.append(reward)
            self.p1_scores = p1_scores
            self.p2_scores = p2_scores

            # TODO: encode the final state
            next_state = self.get_state()
            self.next_states.append(next_state.detach().cpu().numpy())
            self.terminations.append(True)

            return next_state, reward, done, None
        
        self.cur_dialog.append(f"Turn {self.cur_turn}: {self.agent2_name} said:" if self.cur_turn % 2 != 0 else f"Turn {self.cur_turn}: {self.agent1_name} said:")
        
        # let base model generate the response
        self._act(action=None, actor=False) if self.actor_role == 'agent1' else self._act(action=action, actor=True)

        done = self._judge_terminate()
        if done:
            reward, p1_scores, p2_scores = self.get_final_reward()
            self.cur_rewards.append(reward)
            self.p1_scores = p1_scores
            self.p2_scores = p2_scores

            # TODO: encode the final state
            next_state = self.get_state()
            self.next_states.append(next_state.detach().cpu().numpy())
            self.terminations.append(True)

            return next_state, reward, done, None

        self.cur_dialog.append(f"Turn {self.cur_turn}: {self.agent2_name} said:" if self.cur_turn % 2 != 0 else f"Turn {self.cur_turn}: {self.agent1_name} said:")

        # get the state of the new prompt
        next_state = self.get_state()

        # save the state
        self.cur_rewards.append(0)
        self.next_states.append(next_state.detach().cpu().numpy())
        self.terminations.append(False)

        # return the new state, reward, done, info
        return next_state, 0, done, None

    