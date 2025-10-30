import random

import torch

import verl
from verl import DataProto

def compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta):
    cur_scores = ((token_level_scores * response_mask).sum(dim=1) * beta).sigmoid()
    cur_dpo_loss = torch.nn.functional.binary_cross_entropy(cur_scores, acc) # weakness: recognize current step as a positive sample when its trajectory is successful
    return cur_dpo_loss

def compute_detach_dpo_loss_rm(token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"):
    # we always assume that the BoN size equals n_samples
    # mode1: use acc as rm
    # mode2: use Q as rm
    cur_Q = (token_level_scores * response_mask).sum(dim=1) * beta
    other_Q = torch.zeros_like(cur_Q)
    for i in range(token_level_scores.shape[0]):
        Q_chosen = Q_bc[i][acc_bc[i] < acc[i]] if acc[i] > 0 else Q_bc[i][acc_bc[i] > acc[i]]
        if len(Q_chosen) > 0:
            other_Q[i] = Q_chosen.mean() * beta
        else:
            other_Q[i] = 0 # reduce to ce_loss but may cause numerical instability
    dpo_loss = -torch.log(torch.sigmoid((cur_Q - other_Q) * ((acc > 0).float() * 2 - 1)))
    if bon_mode == "none":
        dpo_loss = dpo_loss.mean()
    else: # higher weight for samples that outperform more references
        weight = torch.zeros_like(dpo_loss)
        n_samples = acc_bc.shape[1]
        if bon_mode == "bon_rm":
            for i in range(token_level_scores.shape[0]):
                weight[i] = n_samples * torch.pow((Q_bc[i] * beta <= cur_Q[i]).float().mean(), n_samples - 1)
        elif bon_mode == "bon_acc":
            for i in range(token_level_scores.shape[0]):
                weight[i] = n_samples * torch.pow((acc_bc[i] <= acc[i]).float().mean(), n_samples - 1)
        else:
            raise NotImplementedError
        dpo_loss = (dpo_loss * weight).sum()

    return dpo_loss

def compute_eto_loss_rm(data: DataProto, token_level_scores, beta, env_name):
    """
    Compute DPO loss using trajectory-level preferences with one-to-one pairing.
    Each successful trajectory is paired with exactly one unsuccessful trajectory,
    avoiding reuse of trajectories in subsequent comparisons.
    
    Args:
        data: DataProto containing batch information
        token_level_scores: Q-values for each step (batch_size, response_length)
        beta: Temperature parameter for DPO
    Returns:
        dpo_loss: Computed DPO loss
    """
    if not token_level_scores.requires_grad:
        token_level_scores = token_level_scores.requires_grad_(True)
    
    # Extract trajectory information
    traj_uids = data.non_tensor_batch['traj_uid']
    uid_batch = data.non_tensor_batch['uid']  # Environment group IDs
    episode_rewards = data.batch['acc']  # Trajectory-level success
    active_masks = data.non_tensor_batch['active_masks']
    
    # Step 1: Aggregate step-level Q-values to trajectory-level
    traj_q_values = {}  # Maps traj_uid -> trajectory Q-value
    traj_success = {}   # Maps traj_uid -> trajectory success (episode_rewards)
    traj_env_group = {} # Maps traj_uid -> environment group (uid)
    
    # Compute Q-values for each step
    prompt_ids = data.batch['prompts']
    prompt_length = prompt_ids.shape[-1]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, prompt_length:]
    step_q_values = (token_level_scores * response_mask).sum(dim=1) * beta
    
    for i in range(len(traj_uids)):
        if not active_masks[i]:  # Skip inactive steps
            continue
            
        traj_uid = traj_uids[i]
        env_group = uid_batch[i]
        
        if traj_uid not in traj_q_values:
            traj_q_values[traj_uid] = []
            traj_success[traj_uid] = episode_rewards[i]
            traj_env_group[traj_uid] = env_group
        
        traj_q_values[traj_uid].append(step_q_values[i])
    
    for traj_uid in traj_q_values:
        traj_q_values[traj_uid] = torch.stack(traj_q_values[traj_uid]).mean()
    
    # Step 2: Group trajectories by environment group
    env_groups = {}  # Maps env_group -> list of traj_uids
    for traj_uid, env_group in traj_env_group.items():
        if env_group not in env_groups:
            env_groups[env_group] = []
        env_groups[env_group].append(traj_uid)
    
    # Step 3: Construct one-to-one preference pairs within each environment group
    total_loss = 0.0
    total_pairs = 0
    
    for env_group, traj_list in env_groups.items():
        if len(traj_list) < 2:  
            continue
            
        # Separate successful and unsuccessful trajectories
        successful_trajs = [uid for uid in traj_list if traj_success[uid] > 0]
        unsuccessful_trajs = [uid for uid in traj_list if traj_success[uid] <= 0]
        
        # Handle edge cases
        if len(successful_trajs) == 0 or len(unsuccessful_trajs) == 0:
            continue  
        
        for i in range(len(successful_trajs)):
            for j in range(len(unsuccessful_trajs)):
                succ_traj = successful_trajs[i]
                unsucc_traj = unsuccessful_trajs[j]
                
                succ_q = traj_q_values[succ_traj]
                unsucc_q = traj_q_values[unsucc_traj]
                
                # DPO loss: successful trajectory should have higher Q-value
                pair_loss = -torch.log(torch.sigmoid(succ_q - unsucc_q))
                total_loss += pair_loss
                total_pairs += 1
    
    if total_pairs == 0:
        # No valid pairs found, return zero loss
        print("Warning: No valid trajectory pairs found for DPO training")
        return torch.tensor(0.0, device=token_level_scores.device, requires_grad=True)
    
    return total_loss / total_pairs

def compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples):
    dpo_acc = []
    for start_id in range(0, token_level_scores.shape[0], n_samples):
        cur_scores = (token_level_scores[start_id : start_id + n_samples] * response_mask[start_id : start_id + n_samples]).sum(dim=1)

        def get_upper_triangle(tensor_x):
            diff_matrix = tensor_x.unsqueeze(1) - tensor_x.unsqueeze(0)
            upper_tri_indices = torch.triu(torch.ones_like(diff_matrix).bool(), diagonal=1)
            return diff_matrix[upper_tri_indices]

        cur_acc_diff = get_upper_triangle(acc[start_id : start_id + n_samples])  # in range [-1,1]
        cur_score_diff = get_upper_triangle(cur_scores)  # in R
        cur_score_prediction = (cur_score_diff > 0).float()  # in [0,1]
        if cur_acc_diff.abs().sum() == 0:
            cur_acc = torch.zeros_like(cur_score_prediction[0]) + 0.5
        else:
            cur_acc = (((cur_score_diff > 0) == (cur_acc_diff > 0)).float() * cur_acc_diff.abs()).sum() / cur_acc_diff.abs().sum()

        dpo_acc.append(cur_acc.unsqueeze(0))

    return torch.cat(dpo_acc, dim=0).mean()

def compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples):
    return (torch.sign((token_level_scores * response_mask).sum(dim=-1)) == torch.sign(acc * 2 - 1)).float().mean()
