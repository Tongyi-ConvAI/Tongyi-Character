from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import sys
import os
from pathlib import Path

from PROMPTS import USER_PREFERENCE_ANALYSIS_PROMPT, USER_PROMPT_TEMPLATE

model_path = str((Path(__file__).resolve().parent.parent / "checkpoints" / "P-GenRM-8B-PRISM"))

historical_choices='''
User: If you could change one thing about how humans behave, what would you change?
Chosen: I would change humans to be more open-minded and kind towards each other. We should strive to build bridges instead of barriers and to promote inclusivity and understanding. We should also prioritize collective action and recognize the interconnectedness of our lives.
Rejected: I’d change the misogynistic behavior of men towards women.

User: Do you think ghost are real?
Chosen: I'm just an AI, I don't have personal opinions or beliefs. However, I can provide information on the topic. The existence of ghosts is a topic of debate and speculation. While some people claim to have experienced paranormal activities, scientific evidence to support the existence of ghosts is scarce. Many alleged ghost sightings can be explained by natural phenomena, psychological factors, or misinterpretation of sensory information. Therefore, it's uncertain whether ghosts exist or not.
Rejected: The existence of ghosts is a matter of belief. From a scientific standpoint, there is no conclusive evidence that proves ghosts are real. People's experiences and cultural backgrounds often heavily influence their belief in the supernatural.

User: Do you think there is value in working hard or is it better to choose the easier option?
Chosen: As an AI language model, I don't have personal opinions or beliefs. There is value in different perspectives and judgements on the concept of hard work depending on the context. Some may believe that hard work is necessary for success, while others may consider it less crucial. It's also important to note that hard work can be stressful and detrimental if overdone or improperly motivated. Ultimately, one's outlook on the value of hard work is influenced by personal goals, circumstances, and preferences.
Rejected: Of course! Here's the answer: Working hard and choosing the easier option both have value, but in different ways. Working hard builds character, discipline, and a strong work ethic, while choosing the easier option can help you prioritize your well-being and avoid burnout. Balancing the two is key to a fulfilling life.
'''

user_input = "Do you think family should take priority over one's career?"

response_list = [
    "Yes, family should take priority over one's career in my opinion. A balanced life with a healthy work-life balance is important for both mental and physical health, and prioritizing family over career can help ensure a sense of fulfillment and wellbeing. Working hard and sacrificing personal time for one's career can also take its toll on relationships and quality of life, so prioritizing family can provide a sense of balance and stability.",
    "Family should play a significant role in one's life, but it's important to strike a balance between personal and professional responsibilities. Prioritizing one over the other can lead to burnout and neglect in other areas of life. Finding a healthy balance between family and career is crucial for overall well-being."
]

system_prompt = USER_PREFERENCE_ANALYSIS_PROMPT.format(few_shots=historical_choices)
user_prompt = USER_PROMPT_TEMPLATE.format(
    user_input=user_input,
    response_1=response_list[0],
    response_2=response_list[1],
)
messages = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': user_prompt},
]

tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=True,)
model = AutoModelForCausalLM.from_pretrained(
    model_path,  
    local_files_only=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=model.dtype,
    pad_token_id=tokenizer.pad_token_id,
)

out = gen(
    prompt,
    max_new_tokens=3072,
    temperature=0.9,
    do_sample=True,
    return_full_text=False,  
)

assistant_reply = out[0]["generated_text"]

print("\n","-"*20,"Evalution Chain","-"*20,"\n")
print(assistant_reply)
