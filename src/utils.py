import re
import json
import torch
import math
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch.nn.functional as F
from os import path
from sklearn.cluster import KMeans

def word_tokenize(sent):
    pat = re.compile(r'[\w]+|[.,!?;|]')
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []
    
def items_count(config):
    smap = json.load(open(path.join(config.data_dir, config.dataset, "smap.json")))
    return len(smap)


def kmeans(samples, num_clusters, num_iters = 10):
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().double().numpy()

    cluster = KMeans(n_clusters = num_clusters, max_iter = num_iters).fit(x)

    centers = cluster.cluster_centers_

    return centers

def output_pos(token_ids, ground_truth, vocab, reverse_vocab, num_beams):
    pos = []
    batch_size = len(ground_truth)
    pred_sem_ids = []
    ground_sem_ids = []

    pad_token_id = vocab["pad"]
    eos_token_id = vocab["eos"]

    token_ids_detached = token_ids.detach().cpu().tolist()
    
    for seq in token_ids_detached:
        # Filter out special tokens before processing
        valid_tokens = [token_id for token_id in seq if token_id not in {pad_token_id, eos_token_id}]
        sem_id = "-".join([str(token_id) for token_id in valid_tokens]) # Use .get for safety
        pred_sem_ids.append(sem_id)
    for seq in ground_truth:
        valid_tokens = [token_id.item() for token_id in seq if token_id.item() not in {pad_token_id, eos_token_id}]
        sem_id = "-".join([str(token_id) for token_id in valid_tokens])
        ground_sem_ids.append(sem_id)

    for b in range(batch_size):
        this_beam = np.array(pred_sem_ids[b*num_beams:(b+1)*num_beams])
        this_beam_judge = this_beam == ground_sem_ids[b]
        pos.append(this_beam_judge)
    return torch.tensor(np.stack(pos, axis=0), device=token_ids.device)


def get_steady_then_decay_scheduler(optimizer, num_warmup_steps, num_training_steps):
    steady_steps = num_warmup_steps
    if not isinstance(steady_steps, int) or not isinstance(num_training_steps, int):
        raise TypeError("steady_steps and num_training_steps must be int")
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must > 0")
    if steady_steps < 0:
        raise ValueError("steady_steps cannot be negative")
    if steady_steps > num_training_steps:
        raise ValueError("steady_steps cannot be larger than num_training_steps")
    decay_steps = max(1, num_training_steps - steady_steps) 
    def lr_lambda(current_step):
        if current_step < steady_steps:
            return 1.0
        if current_step <= num_training_steps:
            progress = (current_step - steady_steps) / decay_steps
            return max(0.0, 1.0 - progress)
        return 0.0
    return LambdaLR(optimizer, lr_lambda)


def all_possible_ids(vocab, semantic_ids, codebook_offset=0):
    sidlist = []
    for i, sid in enumerate(semantic_ids):
        wordpieces = [
            vocab[f'cb{k + codebook_offset}_{w}'] for k, w in enumerate(sid)
        ]
        sidlist.append([vocab['pad']] + wordpieces + [vocab['eos']])
    return sidlist

def get_cosine_annealing_with_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    if not 0 <= num_warmup_steps < num_training_steps:
        raise ValueError("num_warmup_steps must be in [0, num_training_steps)")

    def lr_lambda(current_step):
        # Stage 1: linear warmup (0 to 1)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Stage 2: cosine annealing (1 to 0)
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(1.0, progress)
        return max(0.0, 0.5 * (1.0 + math.cos(2.0 * math.pi * num_cycles * progress)))
    
    return LambdaLR(optimizer, lr_lambda)
