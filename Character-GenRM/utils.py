import json
import pandas as pd
from collections import Counter
import random
import os
from pathlib import Path
import csv


def dict2csv(res, path):
    """Append a dictionary to a CSV file."""
    if not res:  # If result is empty, do nothing
        return
        
    csv_filename = Path(path)
    # Ensure parent directory exists
    csv_filename.parent.mkdir(parents=True, exist_ok=True)
    
    headers = res.keys()
    file_exists = csv_filename.exists()

    with open(csv_filename, 'a', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(res)

def extract_content(response):
    try:
        # Find the start position of JSON
        json_start = response.index("<JSON_START>")
        # Extract analysis text (everything before the JSON markers)
        analysis_text = response.strip()
        
        # Extract JSON
        json_start = json_start + len("<JSON_START>")
        json_end = response.index("<JSON_END>")
        json_str = response[json_start:json_end].strip()
        
        return analysis_text, json_str
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error extracting content: {e}")
        return None, None

def generate_valid_userids(data, num_ids=10, min_rows=10):
    valid_userids = []
    all_userids = list(data.keys()) 

    results = {}

    while len(valid_userids) < num_ids:
        cur_conv=''''''
        
        userid = random.choice(all_userids)
        
        # Check if this user ID is already selected
        if userid in valid_userids:
            continue

        user_data_count = sum(len(sublist) for sublist in data[userid])
        
        if user_data_count >= min_rows:
            valid_userids.append(userid)

    return valid_userids


def get_fewshots_tests(data,valid_userids,n_examples=3):
    
    random.seed(42)
    
    result = {}
    for user in valid_userids:
        # Get all data for this user
        user_data = data[user]
        cur_all_conv=[item for sublist in user_data for item in sublist]
        # Randomly choose n_examples as few-shot examples
        few_shot = random.sample(cur_all_conv,n_examples)
        
        # Remaining data as test examples
        few_shot_set = set(str(item) for item in few_shot)  # Convert dicts to strings to create a set (dicts are unhashable)
        test_data = [item for item in cur_all_conv if str(item) not in few_shot_set]
        
        result[user] = {
            'few_shot': few_shot,
            'test_data': test_data
        }
    
    return result

import random

def get_formal_train_val(all_user_data, n_examples=3, test_split_name='test'):

    
    random.seed(42)
    
    result = {}

    # Validate test_split_name
    if test_split_name not in ['val', 'test']:
        raise ValueError("test_split_name must be 'val' or 'test'")

    # Iterate over each user and their data in all_user_data
    for user_id, data_tuple in all_user_data.items():
        # 1. Unpack train, val, test lists
        train_records, val_records, test_records = data_tuple

        # 2. Draw few-shot examples from train_records
        # If there are no training records, skip this user
        if not train_records:
            continue
        
        # Use min() to keep sample size within bounds
        few_shot = random.sample(train_records, min(n_examples, len(train_records)))

        # 3. Choose which split to use as test data
        if test_split_name == 'val':
            test_data = val_records
        else:  # test_split_name == 'test'
            test_data = test_records

        # 4. Assemble into the requested format and store in result
        result[user_id] = {
            'few_shot': few_shot,
            'test_data': test_data
        }
        
    return result

def get_full_train_val(all_user_data, n_examples, test_split_name):
    """
    Return format:
    {
        user_id: {
            'few_shot': [...],
            'test_data': [...]
        },
        ...
    }
    """
    random.seed(42)

    if test_split_name not in ('val', 'test'):
        raise ValueError("test_split_name must be 'val' or 'test'")

    result = {}

    for user_id, (train_records, val_records, test_records) in all_user_data.items():
        # Skip if no train samples
        if not train_records:
            continue

        # ---- 1. Sample few-shot (by index) ----
        n_shot = min(n_examples, len(train_records))
        shot_idx = random.sample(range(len(train_records)), n_shot)

        few_shot        = [train_records[i] for i in shot_idx]
        remaining_train = [train_records[i] for i in range(len(train_records)) if i not in shot_idx]

        # ---- 2. Build test_data ----
        base_test = val_records if test_split_name == 'val' else test_records
        # Deep or shallow copy is fine; just don't modify the originals

        if test_split_name == 'val':
            test_data = list(base_test)+remaining_train
        elif test_split_name == 'test':
            test_data = list(base_test)
        # test_data = list(base_test) + remaining_train

        # ---- 3. Write result ----
        result[user_id] = {
            'few_shot' : few_shot,
            'test_data': test_data
        }

    return result

from typing import Dict, List, Tuple, Any

def get_full_train_val_rotate(
    all_user_data: Dict[Any, Tuple[List[Any], List[Any], List[Any]]],
    n_examples: int,
    test_split_name: str,
    seed: int = 42
) -> Dict[Any, List[Dict[str, Any]]]:
    """
    Conventions:
    - When test_split_name == 'val':
        For each user, treat every sample in (train + val) as test_data;
        few_shot is sampled from the remaining samples in (train + val) after excluding the current one,
        randomly take n_examples (or all if fewer).
    - When test_split_name == 'test':
        For each user, treat every sample in test as test_data;
        few_shot is sampled only from train, randomly take n_examples (or all if fewer; empty if no train).

    Args:
        all_user_data: { user_id: (train_records, val_records, test_records), ... }
        n_examples:    target number of few_shot examples
        test_split_name: 'val' or 'test'
        seed:          RNG seed for reproducibility

    Returns:
        {
            user_id: [
                {'few_shot': [...], 'test_data': sample_0},
                {'few_shot': [...], 'test_data': sample_1},
                ...
            ],
            ...
        }
    """
    if test_split_name not in ('val', 'test'):
        raise ValueError("test_split_name must be 'val' or 'test'")

    random.seed(seed)
    result: Dict[Any, List[Dict[str, Any]]] = {}

    for user_id, (train_records, val_records, test_records) in all_user_data.items():
        entries: List[Dict[str, Any]] = []

        if test_split_name == 'val':
            # —— Handle train + val: leave-one-out few-shot ——
            pool = list(train_records) + list(val_records)
            if not pool:
                # This user has no train/val; skip
                continue

            total = len(pool)
            for i in range(total):
                current = pool[i]
                candidates = pool[:i] + pool[i+1:]  # Exclude current sample
                k = min(n_examples, len(candidates))
                few = random.sample(candidates, k) if k > 0 else []
                # Deduplicate
                unique_few = list({str(d["chosen"]): d for d in few}.values())

                entries.append({'few_shot': unique_few, 'test_data': current})

        else:  # test_split_name == 'test'
            # —— For test: few-shot comes only from train ——
            test_list = list(test_records)
            train_pool = list(train_records)
            if not test_list:
                # If no test samples, skip this user
                continue

            for t in test_list:
                k = min(n_examples, len(train_pool))
                few = random.sample(train_pool, k) if k > 0 else []
                entries.append({'few_shot': few, 'test_data': t})

        if entries:
            result[user_id] = entries

    return result

def fewshot_formatter(result):
    
    final_string = ""

    # Iterate over each entry (dict) in the list
    for i, item in enumerate(result):
        # Extract the content of context
        # item['context'] is an array; take the 'content' of the first element
        context_text = item['context'][0]['content']

        # Extract the content of chosen and rejected
        # Adjust if their structures differ
        chosen_text = item['chosen']['content'] 
        rejected_text = item['rejected']['content']

        # Format and concatenate to the final string
        final_string += f"User: {context_text}\n"
        final_string += f"Chosen: {chosen_text}\n"
        final_string += f"Rejected: {rejected_text}\n\n"

    return final_string

def fewshot_formatter_with_choice_attributes(result):
    
    final_string = ""

    # Iterate over each entry (dict) in the list
    for i, item in enumerate(result):
        # Extract the content of context
        # item['context'] is an array; take the 'content' of the first element
        context_text = item['context'][0]['content']

        # Extract the content of chosen and rejected
        # Adjust if their structures differ
        chosen_text = item['chosen']['content'] 
        rejected_text = item['rejected']['content']

        choice_attributes=item['formatted_choice_attributes']

        # Format and concatenate to the final string
        final_string += f"User: {context_text}\n"
        final_string += f"Chosen: {chosen_text}\n"
        final_string += f"Rejected: {rejected_text}\n"
        final_string += f"User's Choice Attributes: {choice_attributes}\n\n"

    return final_string

def fewshot_formatter_with_choice_attributes_chosen_rejected_score(result):
    
    final_string = ""

    # Iterate over each entry (dict) in the list
    for i, item in enumerate(result):
        # Extract the content of context
        # item['context'] is an array; take the 'content' of the first element
        context_text = item['context'][0]['content']

        # Extract the content of chosen and rejected
        # Adjust if their structures differ
        chosen_text = item['chosen']['content'] 
        rejected_text = item['rejected']['content']

        choice_attributes=item['formatted_choice_attributes']

        chosen_score=item['chosen_score']
        rejected_score=item['rejected_score']

        # Format and concatenate to the final string
        final_string += f"User: {context_text}\n"
        final_string += f"Chosen: {chosen_text}\n"
        final_string += f"Chosen Score: {chosen_score}\n"
        final_string += f"Rejected: {rejected_text}\n"
        final_string += f"Rejected Score: {rejected_score}\n"
        final_string += f"User's Choice Attributes: {choice_attributes}\n\n"

    return final_string
