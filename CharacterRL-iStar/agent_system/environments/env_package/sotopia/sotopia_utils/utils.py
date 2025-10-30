# Code in this file is adapted from Sotopia: https://github.com/sotopia-lab/sotopia
import sys
sys.path.insert(0, '../../../')

import re
from enum import IntEnum
from pydantic import BaseModel, Field, validator
from verl.envs.utils.generate import fetch_api_result
from verl.envs.utils.utils import load

prefix_prompt = """Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
You can find {agent}'s goal (or background) in the 'Here is the context of this interaction' field.
Note that {agent}'s goal is only visible to you.
You should try your best to achieve {agent}'s goal in a way that align with their character traits.
Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).\n
"""

additional_instruction = """You are at Turn {turn}. You can say something to interact or just say 'left the conversation' to stop continuing.
Note: You can 'left the conversation' if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.
Please provide your response within quotation marks: \"your response content\"
"""

def format_docstring(docstring: str) -> str:
    """Format a docstring for use in a prompt template."""
    return re.sub("\n +", "\n", docstring).strip()

class RelationshipType(IntEnum):
    stranger = 0
    know_by_name = 1
    acquaintance = 2
    friend = 3
    romantic_relationship = 4
    family_member = 5

class Message(BaseModel):
    """
    An interface for messages.
    There is only one required method: to_natural_language
    """

    def to_natural_language(self) -> str:
        raise NotImplementedError


class ScriptBackground(Message):
    scenario: str = Field(description="scenario of the episode")
    p1_name: str = Field(description="name of participant 1")
    p2_name: str = Field(description="name of participant 2")
    p1_background: str = Field(description="background of participant 1")
    p2_background: str = Field(description="background of participant 2")
    p1_goal: str = Field(description="goal of participant 1")
    p2_goal: str = Field(description="goal of participant 2")

    def to_natural_language(self) -> str:
        return format_docstring(
            f"""Here is the context of this interaction:
        Scenario: {self.scenario}
        Participants: {self.p1_name} and {self.p2_name}
        {self.p1_name}'s background: {self.p1_background}
        {self.p2_name}'s background: {self.p2_background}
        {self.p1_name}'s goal: {self.p1_goal}
        {self.p2_name}'s goal: {self.p2_goal}
        """
        )
    
class EvaluationBySocialDimensions(BaseModel):
    believability: tuple[str, int] = Field(
        ...,
        description="Reasoning requirement: 1. Evaluate if the agent interacts with others in a natural and realistic manner (here are a few common questions to check: a. whether the agent is confusing with its own identity? b. whether the agent repeats others' words/actions without any reason? c. whether the agent is being overly polite considering the context?). Start the analysis with tag <naturalness> "
        "2. Analyze whether the actions of the agent align with their character traits (e.g., personality, values, and etc.). Start the analysis with tag <consistency>. "
        "Output your reasoning process to the 'reasoning' field. Output an integer score ranging from 0 and 10 in the 'score' field. A higher score indicates that the agent is more believable.",
    )
    relationship: tuple[str, int] = Field(
        ...,
        description="Please first analyze what relationship the participant has with the other agent(s) before the interaction. "
        "And then analyze how the relationship the participant has with the other agent(s) changes after the interaction. "
        "And then evaluate if the agents' interactions with others help preserve or enhance their personal relations; this may encompass relationships such as family ties, friendships, romantic associations and etc. "
        "Additionally, ascertain whether these interactions also impact their social status or reputation. "
        "In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from -5 to 5 in the 'score' field. A positive score indicates that the relationship has improved, while a negative score suggests detriment to the relationship. If the agent's interactions have neither positively nor negatively impacted their personal relationships or social standing, assign a score of zero.",
    )
    knowledge: tuple[str, int] = Field(
        ...,
        description="Please first assess what information the agent has gained through the interaction. "
        "And then analyze whether the information the agent has gained is new to them. "
        "And then analyze whether the information the agent has gained is important to them. "
        "In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from 0 and 10 in the 'score' field. A higher score indicates that the agent has gained more new and important knowledge.",
    )
    secret: tuple[str, int] = Field(
        ...,
        description="First please analyze what secret or secretive intention the participant wants to keep. "
        "And then analyze whether they keep it successfully. Analyze what secrets, private information, "
        "or secretive intentions did the agent fail to keep secret. "
        "Please output the reasoning in the reasoning field. "
        "Give an integer score ranging from -10 to 0 (-10 indicates participant leaks critical secrets/intentions, 0 indicates no secret/intention is revealed)",
    )
    social_rules: tuple[str, int] = Field(
        ...,
        description="Please evaluate whether the agent violates any moral rules or laws in their interactions with others. "
        "In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from -10 and 0 in the 'score' field. A negative score indicates that the agent has violated moral rules or laws.",
    )
    financial_and_material_benefits: tuple[str, int] = Field(
        ...,
        description="Please evaluate whether the agent's interactions with others contribute towards financial and material benefits. Analyze what the agent would gain/lose after the interactions. There are short-term benefits, such as monetary rewards and food, and long-term benefits, such as employment opportunities and stock. "
        "In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from -5 and 5 in the 'score' field. Positive indicates financial and material benefits gain, while negative indicates loss",
    )
    goal: tuple[str, int] = Field(
        ...,
        description="Please first reiterate agent's social goals. "
        "And then please provide a comprehensive analysis about the extent to which the agent has managed to achieve these goals. "
        "In the 'reasoning' field, provide a comprehensive account of the logic or thought process that led you to your conclusion. Further, provide an integer score ranging from 0 and 10 in the 'score' field. 0 represents minimal goals achievement, 10 represents complete goal achievement, and a higher score indicates that the agent is making progress towards their social goals.",
    )

    @validator("believability", "knowledge", "goal")
    def zero_to_ten_validator(cls, v: tuple[str, int]) -> tuple[str, int]:
        assert v[1] >= 0 and v[1] <= 10
        return v

    @validator("relationship", "financial_and_material_benefits")
    def minus_five_to_five_validator(
        cls, v: tuple[str, int]
    ) -> tuple[str, int]:
        assert v[1] >= -5 and v[1] <= 5
        return v

    @validator("secret", "social_rules")
    def minus_ten_to_zero_validator(
        cls, v: tuple[str, int]
    ) -> tuple[str, int]:
        assert v[1] >= -10 and v[1] <= 0
        return v


class EnvResponse(BaseModel):
    agent_1_evaluation: EvaluationBySocialDimensions
    agent_2_evaluation: EvaluationBySocialDimensions


def _map_gender_to_adj(gender: str) -> str:
    gender_to_adj = {
        "Man": "male",
        "Woman": "female",
        "Nonbinary": "nonbinary",
    }
    if gender:
        return gender_to_adj[gender]
    else:
        return ""

def _agent_profile_to_stranger_self(
    profile, agent_id: int
) -> str:
    return f"<root><p viewer='agent_{agent_id}'>{profile['first_name']} {profile['last_name']} is a {profile['age']}-year-old {_map_gender_to_adj(profile['gender'])} {profile['occupation'].lower()}. {profile['gender_pronoun']} pronouns. {profile['public_info']} Personality and values description: {profile['personality_and_values']} {profile['first_name']}'s secrets: {profile['secret']}</p></root>"


def _agent_profile_to_name_self(profile, agent_id: int) -> str:
    return f"{profile['first_name']} {profile['last_name']} <p viewer='agent_{agent_id}'>is a {profile['age']}-year-old {_map_gender_to_adj(profile['gender'])} {profile['occupation'].lower()}. {profile['gender_pronoun']} pronouns. {profile['public_info']} Personality and values description: {profile['personality_and_values']} {profile['first_name']}'s secrets: {profile['secret']}</p>"


def _agent_profile_to_aquaintance_self(
    profile, agent_id: int
) -> str:
    return f"{profile['first_name']} {profile['last_name']} is a {profile['age']}-year-old {_map_gender_to_adj(profile['gender'])} {profile['occupation'].lower()}. {profile['gender_pronoun']} pronouns. {profile['public_info']} <p viewer='agent_{agent_id}'>Personality and values description: {profile['personality_and_values']} {profile['first_name']}'s secrets: {profile['secret']}</p>"


def _agent_profile_to_friendabove_self(
    profile, agent_id: int
) -> str:
    return f"{profile['first_name']} {profile['last_name']} is a {profile['age']}-year-old {_map_gender_to_adj(profile['gender'])} {profile['occupation'].lower()}. {profile['gender_pronoun']} pronouns. {profile['public_info']} Personality and values description: {profile['personality_and_values']} <p viewer='agent_{agent_id}'>{profile['first_name']}'s secrets: {profile['secret']}</p>"

def get_bio(
    relationship: RelationshipType, profile, agent_id: int
) -> str:
    match relationship:
        case RelationshipType.stranger:
            return _agent_profile_to_stranger_self(profile, agent_id=agent_id)
        case RelationshipType.know_by_name:
            return _agent_profile_to_name_self(profile, agent_id=agent_id)
        case RelationshipType.acquaintance:
            return _agent_profile_to_aquaintance_self(
                profile, agent_id=agent_id
            )
        case RelationshipType.friend | RelationshipType.romantic_relationship | RelationshipType.family_member:
            return _agent_profile_to_friendabove_self(
                profile, agent_id=agent_id
            )
        case _:
            raise ValueError(f"Unknown relationship {relationship}")


def extract_leading_int(input_string):
    if input_string.startswith(str(10)):
        return 10
    for i in range(1, 10):
        if input_string.startswith(str(i)):
            return i
    return 0

def format_bad_output(
    ill_formed_output: str,
    format_instructions: str,
    model_name: str = "gpt-3.5-turbo",
) -> str:
    template = """
    Given the string that can not be parsed by json parser, reformat it to a string that can be parsed by json parser.
    Original string: {ill_formed_output}

    Format instructions: {format_instructions}

    Please only generate the JSON:
    """
    config = model_name
    prompt = template.format(
        ill_formed_output=ill_formed_output, format_instructions=format_instructions
    )
    prompt_input = {"prompt": {"user": prompt}}
    reformat = fetch_api_result(prompt_input, config, 8000)
    return reformat