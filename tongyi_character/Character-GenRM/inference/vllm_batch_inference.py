import json
import requests
import concurrent.futures
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pre_process.dataset import PrismDataset, ChatbotArenaDataset,Prism_personal_align_Dataset
from PROMPTS import USER_PREFERENCE_ANALYSIS_PROMPT,USER_PROMPT_TEMPLATE
import random
from utils import fewshot_formatter,fewshot_formatter_with_choice_attributes,generate_valid_userids,get_fewshots_tests,fewshot_formatter_with_choice_attributes_chosen_rejected_score,get_formal_train_val,get_full_train_val,get_full_train_val_rotate,extract_content,dict2csv
import os
import csv
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import csv

session = requests.Session()
retries = Retry(total=3, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(pool_connections=200, pool_maxsize=200, max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)

def request_local_vllm(messages, model):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {'Content-Type': 'application/json'}
    data = {'model': model, 'messages': messages}
    try:
        # Connect timeout 10s, read timeout 360s (adjust based on generation length)
        r = session.post(url, headers=headers, json=data, timeout=(10, 360))
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"Error calling local VLLM API: {e}")
        return {"choices": [{"message": {"content": ""}}], "error": str(e)}

def call_llm_and_prepare_data(userid, few_shot_examples, user_input, response_list, chosen_score, rejected_score, model):
    """
    Core function that integrates data preparation, LLM invocation, and result parsing.
    """
    # Record the original chosen and rejected for later accuracy calculation
    golden_chosen = response_list[0]
    # Shuffle responses randomly to avoid position bias
    random.shuffle(response_list)

    # Format few-shots
    fewshots_formatted = fewshot_formatter(few_shot_examples)

    # Format System Prompt
    system_prompt = USER_PREFERENCE_ANALYSIS_PROMPT.format(
        few_shots=fewshots_formatted,
    )

    # Format User Prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        user_input=user_input,
        response_1=response_list[0],
        response_2=response_list[1],
    )

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]

    # Call local VLLM
    response = request_local_vllm(messages, model)

    # Parse LLM response
    try:
        # Check for errors, e.g., VLLM service not started
        if "error" in response:
            print(f"One task for user {userid} failed to call the LLM. Response: {response}")
            return None
        content = response['choices'][0]['message']['content']

        # Assume extract_content is still applicable
        examples_analysis_process, parsed_content = extract_content(content)
        if parsed_content is None:
            print(f"One task for user {userid} failed to parse; skipping. Content: {content}")
            return None

        parsed_content = json.loads(parsed_content)
        llm_scores = parsed_content.get('better_response', {})
        llm_rationale = parsed_content.get('rationale', 'N/A')

        if not llm_scores or len(llm_scores) != 2:
            print(f"One task for user {userid} returned improperly formatted scores; skipping.")
            return None

        # Determine LLM's choice
        score1 = llm_scores.get('response_1', -1)
        score2 = llm_scores.get('response_2', -1)

        if score1 == score2:
            return None

        llm_choice_idx = 0 if score1 > score2 else 1
        llm_chosen_response = response_list[llm_choice_idx]
        is_correct = (llm_chosen_response == golden_chosen)

    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        print(f"Error processing LLM response: {e}. User: {userid}. Response: {response}")
        return None

    # Prepare data to be written to CSV
    res = {
        'userid': userid,
        'total_messages': json.dumps(messages, ensure_ascii=False),
        'examples_analysis_process': examples_analysis_process,
        'rationale': llm_rationale,
        'chose': llm_chosen_response,
        'golden_chosen': golden_chosen,
        'response_list': json.dumps(response_list, ensure_ascii=False),
        'chosen_score': chosen_score,
        'rejected_score': rejected_score,
        'Correct/Wrong': is_correct
    }
    return res

def _normalize_user_entries(user_entries):
    """
    Accepts two input styles:
    - New format: List[{'few_shot': [...], 'test_data': {...}}]
    - Old format: {'few_shot': [...], 'test_data': List[{...}] or {...}}
    Returns unified: List[{'few_shot': [...], 'test_data': {...}}]
    """
    # Old-format dict
    if isinstance(user_entries, dict) and 'few_shot' in user_entries and 'test_data' in user_entries:
        few = user_entries.get('few_shot', [])
        tests = user_entries.get('test_data', [])
        if isinstance(tests, list):
            return [{'few_shot': few, 'test_data': t} for t in tests]
        else:
            return [{'few_shot': few, 'test_data': tests}]
    # New-format list
    if isinstance(user_entries, list):
        if all(isinstance(e, dict) and 'few_shot' in e and 'test_data' in e for e in user_entries):
            return user_entries
        else:
            raise ValueError("user_entries is a list, but elements do not contain few_shot/test_data.")
    raise ValueError("Unknown user_entries format.")

def process_user_data_concurrently(userid, user_entries, csvpath, model, max_workers):
    """
    Supports the new format returned by get_full_train_val(..., 'test'):
    user_entries: List[{'few_shot': List[Any], 'test_data': Dict}]
    """
    user_entries = _normalize_user_entries(user_entries)
    correct_count = 0
    total_processed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for entry in user_entries:
            few_shot_examples = entry.get('few_shot', [])
            test_item = entry.get('test_data', None)
            if not test_item:
                continue
            try:
                user_input = test_item['context'][0]['content']
                chosen_text = test_item['chosen']['content']
                rejected_text = test_item['rejected']['content']
                responses_list = [chosen_text, rejected_text]
                # Scores may be missing; set to None if absent
                raw_chosen = test_item.get('chosen_score', None)
                raw_rejected = test_item.get('rejected_score', None)
                chosen_score = (raw_chosen / 10) if isinstance(raw_chosen, (int, float)) else None
                rejected_score = (raw_rejected / 10) if isinstance(raw_rejected, (int, float)) else None

                fut = executor.submit(
                    call_llm_and_prepare_data,
                    userid, few_shot_examples, user_input, responses_list, chosen_score, rejected_score, model
                )
                futures.append(fut)
            except Exception as e:
                print(f"Error submitting task for user {userid}: {e}")

        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing User {userid}"):
            try:
                output = fut.result()
                if output:
                    dict2csv(output, csvpath)
                    total_processed += 1
                    if output['Correct/Wrong']:
                        correct_count += 1
            except Exception as exc:
                print(f"An LLM call task errored during execution: {exc}")

    accuracy = (correct_count / total_processed) * 100 if total_processed > 0 else 0
    print(f"--- User {userid} finished ---")
    print(f" - Total processed samples: {total_processed}")
    print(f" - Correct count: {correct_count}")
    print(f" - Accuracy: {accuracy:.2f}%")
    return total_processed, correct_count

def run_experiment(run_index, all_data, output_csv_path,test_split_name):
    """
    Run a single complete experiment.

    Args:
        run_index (int): Which run number (starting from 1).
        all_data (dict): Dict containing all user data.
        output_csv_path (str): Output path for this run.

    Returns:
        tuple: (total samples processed this run, correct samples this run, accuracy this run)
    """
    print("\n" + f"--- Starting run {run_index}/{NUM_RUNS} ---")

    # Data preparation
    # Note: To ensure consistent few-shot/test splits across runs, we fix the seed.
    # If you want different random splits each run, remove random.seed(42) or use different seeds.
    random.seed(42)
    gft = get_full_train_val_rotate(all_data, n_examples=NUM_FEWSHOTS, test_split_name=test_split_name)
    print(f"Run {run_index}: dataset prepared; found {len(gft)} users.")

    # Initialize counters for this run
    total_samples_run = 0
    total_correct_run = 0

    # Ensure the output directory exists
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Main loop
    for userid, data in gft.items():
        # process_user_data_concurrently returns (processed sample count, correct count)
        processed, correct = process_user_data_concurrently(userid, data, csvpath=output_csv_path, model=MODEL_NAME, max_workers=MAX_WORKERS)
        total_samples_run += processed
        total_correct_run += correct

    # Summary for this run
    run_accuracy = (total_correct_run / total_samples_run) * 100 if total_samples_run > 0 else 0
    print(f"--- End of run {run_index}/{NUM_RUNS} ---")
    print(f"Samples processed this run: {total_samples_run}")
    print(f"Correct this run: {total_correct_run}")
    print(f"Accuracy this run: {run_accuracy:.2f}%")
    print(f"Results saved to: {output_csv_path}")
    return total_samples_run, total_correct_run, run_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to score data after GRPO, using a served model."
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help="The name of the served model (e.g., from $SERVED_MODEL_NAME)."
    )
    parser.add_argument(
        '--fs', type=int, required=True,
        help="Number of few-shot examples (e.g., from $FEWSHOTS)."
    )
    parser.add_argument(
        '--max-workers', type=int, default=36,
        help="Number of concurrent client threads."
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        help="Target dataset: 'prism' or 'chatbot'."
    )
    parser.add_argument("--test_split_name", type=str, default="test", help="Name of the test split")

    args = parser.parse_args()

    MODEL_NAME = args.model
    # Concurrency (client-side) thread count
    MAX_WORKERS = args.max_workers
    # Number of few-shot examples
    NUM_FEWSHOTS = args.fs
    target_dataset = args.dataset
    NUM_RUNS = 3
    # Get the absolute path of the current file (vllm_batch_inference.py)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Create a path to ../vllm_experiments relative to the project root
    OUTPUT_DIR = os.path.join(CURRENT_DIR, '..', 'vllm_experiments')

    # Ensure the folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_split_name = args.test_split_name
    # Save results to that folder
    BASE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_{test_split_name}.csv')

    # Data loading (only once)
    print("Loading and preparing datasets...")
    if target_dataset=='prism':
        dataset = Prism_personal_align_Dataset()
    elif target_dataset=='chatbotarena':
        dataset = ChatbotArenaDataset()
    else:
        raise ValueError(f"Unsupported dataset: {target_dataset}. Only 'prism' and 'chatbotarena' are supported.")
    all_data = dataset.get_all_user_data()
    print("Datasets loaded.")

    # Store results from each run
    all_run_accuracies = []
    total_samples_all_runs = 0
    total_correct_all_runs = 0

    # Execute experiments repeatedly
    for i in range(NUM_RUNS):
        run_num = i + 1
        # Generate a unique output file name for each run to avoid overwriting
        # e.g., .../model_name_run_1.csv, .../model_name_run_2.csv
        output_path_for_run = BASE_OUTPUT_PATH.replace('.csv', f'_run_{run_num}.csv')

        samples, corrects, accuracy = run_experiment(
            run_index=run_num,
            all_data=all_data,
            output_csv_path=output_path_for_run,
            test_split_name=test_split_name
        )

        # Collect results
        total_samples_all_runs += samples
        total_correct_all_runs += corrects
        all_run_accuracies.append(accuracy)

    # Final summary
    # Compute average accuracy
    average_accuracy = sum(all_run_accuracies) / len(all_run_accuracies) if all_run_accuracies else 0

    print("\n" + "="*50)
    print(f"All {NUM_RUNS} runs completed!")
    print("-" * 50)
    print(f"Accuracies per run: {[f'{acc:.2f}%' for acc in all_run_accuracies]}")
    print(f"Total samples across all runs: {total_samples_all_runs}")
    print(f"Total correct across all runs: {total_correct_all_runs}")
    print("-" * 50)
    print(f"Average accuracy: {average_accuracy:.2f}%")
    print(f"Detailed results for each run are saved to the corresponding ..._run_N.csv files.")
    print("="*50)
