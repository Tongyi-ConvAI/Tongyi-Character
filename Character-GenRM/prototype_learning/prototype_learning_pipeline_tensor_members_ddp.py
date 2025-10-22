from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
from torch.distributed.algorithms.join import Join

# ============================
# Local embedding backend (sentence-transformers, local path)
# ============================

class LocalSentenceTransformerBackend:
    def __init__(self, model_path: str, batch_size: int = 128, device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError("`sentence-transformers` is required. pip install sentence-transformers") from e
        kwargs = {"device": device} if device is not None else {}
        self.model = SentenceTransformer(model_path, **kwargs)
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> np.ndarray:
        # single-process/local encoding
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=(dist.get_rank()==0 if dist.is_initialized() else True),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)

# -------- DDP helpers for embedding --------

def _shard_bounds(n: int, world_size: int, rank: int) -> Tuple[int, int]:
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end

def embed_texts_distributed(backend: LocalSentenceTransformerBackend, texts: List[str]) -> np.ndarray:
    """
    Each rank only encodes its own slice; then all_gather to all ranks, ensuring every process gets the complete E.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return backend.embed(texts)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    n = len(texts)

    s, e = _shard_bounds(n, world_size, rank)
    local_texts = texts[s:e]
    local_arr = backend.embed(local_texts) if len(local_texts) > 0 else None

    # Collect results from each rank (variable length) using all_gather_object
    gathered: List[Optional[np.ndarray]] = [None] * world_size
    dist.all_gather_object(gathered, local_arr)

    # Assemble
    parts = [g for g in gathered if g is not None]
    if len(parts) == 0:
        return np.empty((0, 0), dtype=np.float32)
    E = np.concatenate(parts, axis=0)
    return E

# ============================
# Utils & regex extractors
# ============================

def l2n(x: np.ndarray, axis=-1, eps: float = 1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

def clean_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in s.split("\n"):
        line = re.sub(r"[\t ]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)

P1_RE = re.compile(
    r"(?i)"                                     # ignore case
    r"part[\u00A0\u202F\s]*1[\u00A0\u202F\s]*[:：][\u00A0\u202F\s]*"
    r"(?:"
        # Original form: ... user preference model analysis (Chain-of-Thought)
        r"user[\s\u00A0\u202F]*preference[\s\u00A0\u202F]*model[\s\u00A0\u202F]*analysis[\s\u00A0\u202F]*"
        r"\(\s*chain[\-\u2010-\u2015]of[\-\u2010-\u2015]thought\s*\)"
    r"|"
        # New form: Chain-of-Thought Analysis
        r"chain[\-\u2010-\u2015]of[\-\u2010-\u2015]thought[\s\u00A0\u202F]*analysis\b"
    r")"
)

P2_RE = re.compile(
    r"part\s*[\u00A0\u202F\s]*2\s*[:：]\s*final\s*scoring\s*and\s*json\s*output",
    flags=re.IGNORECASE
)

def _normalize(s: str) -> str:
    return (s or "")\
        .replace("\u00A0", " ")   \
        .replace("\u202F", " ")   \
        .replace("\u200B", "")    \
        .replace("\uFEFF", "")    \
        .replace("：", ":")       \
        .replace("（", "(").replace("）", ")") \
        .replace("–", "-").replace("-", "-")

def extract_between_parts(raw_text: str):
    text = _normalize(raw_text)
    m1 = P1_RE.search(text)
    m2 = P2_RE.search(text)
    reason = None
    if not m1:
        reason = "missing_part1"
        return "", {"found_p1": False, "found_p2": bool(m2), "order_ok": False, "reason": reason}
    if not m2:
        reason = "missing_part2"
        return "", {"found_p1": True, "found_p2": False, "order_ok": False, "reason": reason}
    if m2.start() <= m1.end():
        reason = "bad_order"
        return "", {"found_p1": True, "found_p2": True, "order_ok": False, "reason": reason}
    between = raw_text[m1.end():m2.start()].strip()
    if len(between) == 0:
        reason = "empty_between"
    return between, {
        "found_p1": True,
        "found_p2": True,
        "order_ok": True,
        "reason": reason or "ok",
        "extracted_len": len(between),
    }

# Tagged blocks in "user input"

def extract_block(text: str, start_tag: str, end_tag: str) -> str:
    s_pat = re.escape(start_tag).replace("\\ ", r"\s*")
    e_pat = re.escape(end_tag).replace("\\ ", r"\s*")
    m = re.search(s_pat + r"(.*?)" + e_pat, text, flags=re.S | re.I)
    return clean_text(m.group(1)) if m else ""

def parse_user_input_cell(cell: str) -> Dict[str, str]:
    t = str(cell or "")
    q  = extract_block(t, "[The Start of User Input]",  "[The End of User Input]")
    r1 = extract_block(t, "[The Start of Response 1]", "[The End of Response 1]")
    r2 = extract_block(t, "[The Start of Response 2]", "[The End of Response 2]")
    return {"q": q, "r1": r1, "r2": r2}

CHOICE_MAP = {
    "a": 1, "a.": 1, "1": 1, "response 1": 1, "choice a": 1, "option a": 1,
    "b": 2, "b.": 2, "2": 2, "response 2": 2, "choice b": 2, "option b": 2,
}

def chosen_idx_from_value(v: str) -> Optional[int]:
    s = (v or "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    if s in CHOICE_MAP:
        return CHOICE_MAP[s]
    if re.search(r"\b1\b", s): return 1
    if re.search(r"\b2\b", s): return 2
    if re.search(r"\ba\b", s): return 1
    if re.search(r"\bb\b", s): return 2
    return None

# History parser: flexible length; missing history allowed
HIST_S_RE = re.compile(r"\[\s*The\s*Start\s*of\s*User\'?s\s*preference\s*history\s*\]", re.I)
HIST_E_RE = re.compile(r"\[\s*The\s*End\s*of\s*User\'?s\s*preference\s*history\s*\]", re.I)
LINE_USER_RE   = re.compile(r"User\s*:\s*(.*)")
LINE_CHOSEN_RE = re.compile(r"Chosen\s*:\s*(.*)")
LINE_REJECT_RE = re.compile(r"Rejected\s*:\s*(.*)")

def parse_history_cell(cell: str) -> List[Tuple[str, str, str]]:
    t = str(cell or "")
    ms = HIST_S_RE.search(t); me = HIST_E_RE.search(t)
    if not (ms and me and me.start() > ms.end()):
        return []
    block = t[ms.end():me.start()]
    parts = re.split(r"Rejected\s*Score\s*:\s*", block, flags=re.I)
    parts = [p.strip() for p in parts if p.strip()]
    events: List[Tuple[str, str, str]] = []
    for seg in parts:
        usr = LINE_USER_RE.search(seg)
        chs = LINE_CHOSEN_RE.search(seg)
        rej = LINE_REJECT_RE.search(seg)
        if not (usr and chs and rej):
            continue
        h  = clean_text(usr.group(1))
        rp = clean_text(chs.group(1))
        rn = clean_text(rej.group(1))
        events.append((h, rp, rn))
    return events


# ============================
# Data model
# ============================

@dataclass
class Sample:
    idx: int
    user_id: str
    e: Optional[np.ndarray]            # PART1→PART2 middle embedding
    middle_text: str                   # cleaned middle text
    q: Optional[np.ndarray] = None
    r_pos: Optional[np.ndarray] = None
    r_neg: Optional[np.ndarray] = None
    history: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None

class HistoryDataset(Dataset):
    def __init__(self, samples: List[Sample], assign: np.ndarray, target_proto: int, oversample_ratio: int = 1):
        members = [s for s, a in zip(samples, assign) if a == target_proto and s.q is not None]
        self.items = members * max(1, oversample_ratio)
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        s = self.items[i]
        q = torch.from_numpy(s.q).float()
        r_pos = torch.from_numpy(s.r_pos).float()
        r_neg = torch.from_numpy(s.r_neg).float()
        hist = [(torch.from_numpy(h).float(), torch.from_numpy(rp).float(), torch.from_numpy(rn).float()) for (h, rp, rn) in (s.history or [])]
        return q, r_pos, r_neg, hist

def collate_hist(batch):
    qs, rps, rns, hists = zip(*batch)
    return torch.stack(qs), torch.stack(rps), torch.stack(rns), list(hists)

# ============================
# Prototype updater
# ============================

class PrototypeUpdater(nn.Module):
    def __init__(self, prototypes: np.ndarray, d: int, rho: float = 0.5, lambda_q: float = 1.0, lambda_s: float = 1.0):
        super().__init__()
        k, d0 = prototypes.shape
        assert d0 == d
        self.d = d; self.rho = rho; self.lambda_q = lambda_q; self.lambda_s = lambda_s
        self.prototypes = nn.Parameter(torch.from_numpy(prototypes.astype(np.float32)))
        self.W_hist = nn.Linear(2 * d, d)
        self.Wq = nn.Linear(d, d)
        self.Ws = nn.Linear(d, d)
        self.register_buffer("pbar", torch.from_numpy(prototypes.astype(np.float32)))
    def step_ema(self, decay: float = 0.98):
        with torch.no_grad():
            self.pbar.mul_((decay)).add_((1 - decay) * self.prototypes)
    def forward_batch(self, q, r_pos, r_neg, hists, proto_idx):
        device = q.device
        B, d = q.shape
        a_j = self.prototypes[proto_idx].unsqueeze(0).expand(B, -1)
        o_list = []
        for i in range(B):
            events = hists[i]
            if len(events) == 0:
                o_list.append(torch.zeros(1, d, device=device)); continue
            h_stack = torch.stack([h for (h, _, _) in events]).to(device)
            rp_stack = torch.stack([rp for (_, rp, _) in events]).to(device)
            rn_stack = torch.stack([rn for (_, _, rn) in events]).to(device)
            diff = rp_stack - rn_stack
            inp = torch.cat([h_stack, diff], dim=-1)
            o = torch.tanh(self.W_hist(inp))
            o_list.append(o)
        s_q_list = []
        for i in range(B):
            o = o_list[i]
            logits = (o @ a_j[i].unsqueeze(-1)).squeeze(-1) / math.sqrt(d)
            logits = logits + self.rho * ((o @ q[i].unsqueeze(-1)).squeeze(-1) / math.sqrt(d))
            alpha = torch.softmax(logits, dim=0)
            s_q = (alpha.unsqueeze(-1) * o).sum(dim=0)
            s_q_list.append(s_q)
        s_q = torch.stack(s_q_list, dim=0)
        z = a_j + self.lambda_q * self.Wq(q) + self.lambda_s * self.Ws(s_q)
        y_pos = (z * r_pos).sum(dim=-1)
        y_neg = (z * r_neg).sum(dim=-1)
        delta = y_pos - y_neg
        l_pair = F.binary_cross_entropy_with_logits(delta, torch.ones_like(delta))
        return l_pair, {"y_pos_mean": y_pos.mean().detach(), "y_neg_mean": y_neg.mean().detach()}
    def reg_losses(self, proto_idx: int, mu_k: torch.Tensor, lambda_cent: float = 1e-3, lambda_tr: float = 1e-3):
        p_k = self.prototypes[proto_idx]
        l_cent = lambda_cent * torch.mean((p_k - mu_k) ** 2)
        l_tr = lambda_tr * torch.mean((p_k - self.pbar[proto_idx]) ** 2)
        return l_cent + l_tr

# ============================
# Steps
# ============================

def step1_parse_and_embed_for_clustering(
    df: pd.DataFrame,
    backend: LocalSentenceTransformerBackend,
    examples_col: str = "examples_analysis_process",
    user_id_col: str = "user_id",
):
    middles_all: List[str] = []
    mid_reports: List[Dict[str, object]] = []
    for i, cell in enumerate(df[examples_col].fillna("").astype(str).tolist()):
        middle, rep = extract_between_parts(cell)
        rep["row_index"] = i
        rep["kept_for_clustering"] = bool(middle)
        middles_all.append(middle)
        mid_reports.append(rep)
    # keep only rows with non-empty middle
    keep_indices = [r["row_index"] for r in mid_reports if r["kept_for_clustering"]]
    parse_fail_df = pd.DataFrame([r for r in mid_reports if not r["kept_for_clustering"]])
    if len(keep_indices) == 0:
        return np.empty((0,), dtype=np.float32), [], parse_fail_df, []
    df_kept = df.iloc[keep_indices].reset_index(drop=True)
    middles_kept = [middles_all[i] for i in keep_indices]

    # -- Distributed Embedding -- #
    if dist.is_initialized():
        E = embed_texts_distributed(backend, middles_kept)
    else:
        E = backend.embed(middles_kept)
    E = l2n(E)

    user_ids = df_kept[user_id_col].astype(str).tolist() if user_id_col in df_kept.columns else [str(i) for i in range(len(df_kept))]
    samples_stub = [Sample(idx=i, user_id=user_ids[i], e=E[i], middle_text=middles_kept[i]) for i in range(len(df_kept))]
    return E, samples_stub, parse_fail_df, keep_indices


def step1b_parse_training_fields(
    df_kept: pd.DataFrame,
    backend: LocalSentenceTransformerBackend,
    user_input_col: str = "user input",
    fewshots_col: str = "few shots",
    chose_col: str = "chose",
) -> Tuple[List[Sample], pd.DataFrame, pd.DataFrame, List[List[Tuple[str,str,str]]]]:
    ui_series = df_kept[user_input_col].fillna("").astype(str).tolist() if user_input_col in df_kept.columns else [""]*len(df_kept)
    ch_series = df_kept[chose_col].fillna("").astype(str).tolist() if chose_col in df_kept.columns else [""]*len(df_kept)
    fs_series = df_kept[fewshots_col].fillna("").astype(str).tolist() if fewshots_col in df_kept.columns else [""]*len(df_kept)

    q_texts: List[str] = []
    rpos_texts: List[str] = []
    rneg_texts: List[str] = []
    histories_text: List[List[Tuple[str,str,str]]] = []

    train_keep_mask: List[bool] = []
    drop_rows: List[Dict[str, object]] = []

    for i, (ui, ch, fs) in enumerate(zip(ui_series, ch_series, fs_series)):
        parsed = parse_user_input_cell(ui)
        q = parsed["q"]; r1 = parsed["r1"]; r2 = parsed["r2"]
        idx = chosen_idx_from_value(ch)
        reasons = []
        ok = True
        if not q: ok = False; reasons.append("q-empty")
        if not r1 or not r2: ok = False; reasons.append("resp-missing")
        if idx not in (1, 2): ok = False; reasons.append("chose-unrecognized")
        if ok:
            train_keep_mask.append(True)
            q_texts.append(q)
            rpos_texts.append(r1 if idx==1 else r2)
            rneg_texts.append(r2 if idx==1 else r1)
            histories_text.append(parse_history_cell(fs))  # flexible, may be []
        else:
            train_keep_mask.append(False)
            drop_rows.append({"processed_index": i, "reasons": ",".join(reasons)})
            q_texts.append(""); rpos_texts.append(""); rneg_texts.append(""); histories_text.append([])

    kept_idx = [i for i, k in enumerate(train_keep_mask) if k]
    if len(kept_idx) == 0:
        return [], pd.DataFrame({"processed_index": list(range(len(df_kept))), "train_kept": train_keep_mask}), pd.DataFrame(drop_rows), histories_text

    # Embeddings for the training phase are done locally (texts are short, distributed overhead is not worthwhile)
    q_emb  = backend.embed([q_texts[i] for i in kept_idx])
    rp_emb = backend.embed([rpos_texts[i] for i in kept_idx])
    rn_emb = backend.embed([rneg_texts[i] for i in kept_idx])

    # history embedding pool
    pool: List[str] = []
    idx_map: Dict[Tuple[int, int, str], int] = {}
    for i in kept_idx:
        evs = histories_text[i]
        for j, (h, rp, rn) in enumerate(evs):
            idx_map[(i, j, "h")] = len(pool); pool.append(h)
            idx_map[(i, j, "rp")] = len(pool); pool.append(rp)
            idx_map[(i, j, "rn")] = len(pool); pool.append(rn)
    pool_emb: Optional[np.ndarray] = None
    if pool:
        pool_emb = backend.embed(pool)

    # build training samples (subset)
    samples_tr: List[Sample] = []
    user_ids = df_kept["user_id"].astype(str).tolist() if "user_id" in df_kept.columns else [str(i) for i in range(len(df_kept))]
    for n, i in enumerate(kept_idx):
        hist_vecs: List[Tuple[np.ndarray,np.ndarray,np.ndarray]] = []
        for j, (h, rp, rn) in enumerate(histories_text[i]):
            h_e  = pool_emb[idx_map[(i, j, "h")]] if pool_emb is not None else None
            rp_e = pool_emb[idx_map[(i, j, "rp")]] if pool_emb is not None else None
            rn_e = pool_emb[idx_map[(i, j, "rn")]] if pool_emb is not None else None
            hist_vecs.append((h_e, rp_e, rn_e))
        samples_tr.append(Sample(
            idx=i,
            user_id=user_ids[i],
            e=None,
            middle_text="",
            q=q_emb[n], r_pos=rp_emb[n], r_neg=rn_emb[n],
            history=hist_vecs,
        ))

    parsed_summary = pd.DataFrame({
        "processed_index": list(range(len(df_kept))),
        "train_kept": train_keep_mask,
        "num_history": [len(h) for h in histories_text],
    })
    drop_df = pd.DataFrame(drop_rows)
    return samples_tr, parsed_summary, drop_df, histories_text


def step2_cluster(E: np.ndarray, k: int, seed: int = 0):
    if not isinstance(E, np.ndarray) or E.ndim != 2 or E.shape[0] == 0:
        raise ValueError(f"[step2_cluster] No embeddings to cluster, got shape={None if not isinstance(E,np.ndarray) else E.shape}")
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    assign = km.fit_predict(E)
    A = l2n(km.cluster_centers_.astype(np.float32))
    return A, assign, km


def step3_update(samples_tr: List[Sample], A: np.ndarray, assign_all: np.ndarray, train_indices: List[int],
                 epochs=1, batch_size=32, lr=1e-3, rho=0.5, lambda_q=1.0, lambda_s=1.0,
                 lambda_cent=1e-3, lambda_tr=1e-3, ema_decay=0.98, oversample=1,
                 device="cpu", distributed=False, local_rank=0):
    assign_tr = assign_all[np.array(train_indices, dtype=np.int64)]
    d = A.shape[1]
    base_model = PrototypeUpdater(A, d=d, rho=rho, lambda_q=lambda_q, lambda_s=lambda_s).to(device)
    model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) if distributed else base_model
    k = A.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(1, epochs+1):
        epoch_loss, batches = 0.0, 0
        ctx = Join([model]) if distributed else nullcontext()
        with ctx:
            for proto_idx in range(k):
                ds = HistoryDataset(samples_tr, assign_tr, target_proto=proto_idx, oversample_ratio=oversample)
                if len(ds) == 0: continue
                if distributed:
                    sampler = DistributedSampler(ds, shuffle=True, drop_last=False)
                    sampler.set_epoch(epoch)
                    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, collate_fn=collate_hist)
                else:
                    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_hist)
                for q, r_pos, r_neg, hists in loader:
                    q, r_pos, r_neg = q.to(device), r_pos.to(device), r_neg.to(device)
                    opt.zero_grad()
                    l_pair, _ = (model.module if isinstance(model, DDP) else model).forward_batch(q, r_pos, r_neg, hists, proto_idx)
                    loss = l_pair
                    loss.backward(); opt.step()
                    with torch.no_grad():
                        mm = model.module if isinstance(model, DDP) else model
                        p = mm.prototypes[proto_idx]
                        mm.prototypes[proto_idx].copy_(F.normalize(p, dim=0))
                    epoch_loss += loss.item(); batches += 1
                (model.module if isinstance(model, DDP) else model).step_ema(decay=ema_decay)
        if dist.get_rank()==0 if dist.is_initialized() else True:
            print(f"Epoch {epoch}: avg loss = {epoch_loss/max(1,batches):.4f}")
    with torch.no_grad():
        A_new = F.normalize((model.module if isinstance(model, DDP) else model).prototypes, dim=1).cpu().numpy()
    return A_new

# ============================
# DDP helpers
# ============================

def setup_distributed(args):
    args.rank = int(os.environ.get("RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = args.distributed or (args.world_size > 1)
    if args.distributed:
        torch.cuda.setDevice = torch.cuda.set_device  # alias for safety
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
        args.device = f"cuda:{args.local_rank}"
    else:
        args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()

def is_main():
    return (not dist.is_initialized()) or (dist.get_rank()==0)

# ============================
# History savers (training subset)
# ============================

def save_history_artifacts(samples_tr: List[Sample], processed_indices_tr: List[int], row_index_t_all: torch.Tensor, save_dir: str) -> None:
    if len(samples_tr)==0:
        return
    d = samples_tr[0].q.shape[0]
    H_list: List[torch.Tensor] = []; Rp_list: List[torch.Tensor] = []; Rn_list: List[torch.Tensor] = []
    offsets = [0]
    for s in samples_tr:
        evs = s.history or []
        for (h, rp, rn) in evs:
            H_list.append(torch.from_numpy(h).float())
            Rp_list.append(torch.from_numpy(rp).float())
            Rn_list.append(torch.from_numpy(rn).float())
        offsets.append(offsets[-1] + len(evs))
    H  = torch.stack(H_list, dim=0) if H_list else torch.empty(0, d)
    Rp = torch.stack(Rp_list, dim=0) if Rp_list else torch.empty(0, d)
    Rn = torch.stack(Rn_list, dim=0) if Rn_list else torch.empty(0, d)
    offsets_t = torch.tensor(offsets, dtype=torch.long)
    row_index_np = row_index_t_all.cpu().numpy()
    ori_rows = [int(row_index_np[i]) for i in processed_indices_tr]
    user_ids = [s.user_id for s in samples_tr]
    torch.save({
        "H": H, "Rp": Rp, "Rn": Rn,
        "offsets": offsets_t,
        "processed_index": torch.tensor(processed_indices_tr, dtype=torch.long),
        "original_row_index": torch.tensor(ori_rows, dtype=torch.long),
        "user_id": user_ids,
    }, os.path.join(save_dir, "history_vectors.pt"))
    payload = []
    for pi, s in zip(processed_indices_tr, samples_tr):
        evs = s.history or []
        payload.append({
            "processed_index": int(pi),
            "original_row_index": int(row_index_np[pi]),
            "user_id": s.user_id,
            "events": [{"User": "", "Chosen": "", "Rejected": ""} for _ in evs],
        })
    torch.save(payload, os.path.join(save_dir, "history_texts.pt"))

# ============================
# Members saver
# ============================

def save_members_pt(save_dir: str, assign_pt: str, row_index_pt: str, row_index_map_csv: str, out_pt: str) -> None:
    assign = torch.load(os.path.join(save_dir, assign_pt)).cpu().numpy()
    row_index = torch.load(os.path.join(save_dir, row_index_pt)).cpu().numpy()
    if os.path.exists(os.path.join(save_dir, row_index_map_csv)):
        map_df = pd.read_csv(os.path.join(save_dir, row_index_map_csv))
        user_ids = map_df.get("user_id", pd.Series([""]*len(assign))).astype(str).tolist()
    else:
        user_ids = [""] * len(assign)
    k = int(assign.max()) + 1 if assign.size > 0 else 0
    members = []
    for j in range(k):
        proc = np.where(assign == j)[0]
        members.append({
            "processed_index": torch.tensor(proc, dtype=torch.long),
            "original_row_index": torch.tensor(row_index[proc], dtype=torch.long),
            "user_id": [user_ids[i] for i in proc.tolist()],
        })
    torch.save({"members": members}, os.path.join(save_dir, out_pt))

# ============================
# CLI
# ============================

def main():
    ap = argparse.ArgumentParser(description="Prototype pipeline (cluster by PART1→PART2; strict training; flexible history) with DDP embeddings + caching")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--local_model_path", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--embed_batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--lambda_q", type=float, default=1.0)
    ap.add_argument("--lambda_s", type=float, default=1.0)
    ap.add_argument("--lambda_cent", type=float, default=1e-3)
    ap.add_argument("--lambda_tr", type=float, default=1e-3)
    ap.add_argument("--ema_decay", type=float, default=0.98)
    ap.add_argument("--oversample", type=int, default=1)
    ap.add_argument("--user_input_col", default="user input")
    ap.add_argument("--fewshots_col", default="few shots")
    ap.add_argument("--examples_col", default="examples_analysis_process")
    ap.add_argument("--chose_col", default="chose")
    ap.add_argument("--user_id_col", default="user_id")
    # DDP
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--dist_backend", default="nccl")
    # cache
    ap.add_argument("--reuse_cached", action="store_true")
    # device
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # Paths & DDP
    args.csv = os.path.abspath(os.path.expanduser(args.csv))
    args.save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    os.makedirs(args.save_dir, exist_ok=True)

    # DDP setup
    setup_distributed(args)
    if is_main():
        print(f"Distributed={args.distributed}, world_size={getattr(args,'world_size',1)}, device={args.device}")
        print(f"CSV={args.csv}\nSaveDir={args.save_dir}")
    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    # Backend
    backend = LocalSentenceTransformerBackend(
        args.local_model_path,
        batch_size=args.embed_batch,
        device=args.device if torch.cuda.is_available() and (args.device or '').startswith('cuda') else None
    )

    # Cache detection
    cache_ok = args.reuse_cached and all(os.path.exists(os.path.join(args.save_dir, f)) for f in ["E.pt","A_init.pt","assign_init.pt","row_index.pt"])

    if cache_ok:
        if is_main():
            print("Reusing cached E/A/assign from save_dir.")
        E_np = torch.load(os.path.join(args.save_dir, "E.pt")).cpu().numpy()
        row_index = torch.load(os.path.join(args.save_dir, "row_index.pt")).cpu().numpy()
        df_kept = df.iloc[row_index].reset_index(drop=True)
        A_np = torch.load(os.path.join(args.save_dir, "A_init.pt")).cpu().numpy()
        assign_np = torch.load(os.path.join(args.save_dir, "assign_init.pt")).cpu().numpy()
    else:
        # Step 1A: parse middles for clustering only
        E_np, samples_stub, parse_fail_df, keep_indices = step1_parse_and_embed_for_clustering(
            df, backend, examples_col=args.examples_col, user_id_col=args.user_id_col
        )
        if E_np.shape[0] == 0:
            if is_main():
                print("No rows kept for clustering (all middles empty). See parse_failures.csv")
                parse_fail_df.to_csv(os.path.join(args.save_dir, "parse_failures.csv"), index=False)
            cleanup_distributed(); return

        # Only rank0 writes to cache; other ranks wait
        if is_main():
            torch.save(torch.from_numpy(E_np), os.path.join(args.save_dir, "E.pt"))
            row_index_t = torch.tensor(keep_indices, dtype=torch.long)
            torch.save(row_index_t, os.path.join(args.save_dir, "row_index.pt"))
            uid_all = df[args.user_id_col].astype(str).tolist() if args.user_id_col in df.columns else [""]*len(df)
            map_df = pd.DataFrame({
                "processed_index": list(range(len(keep_indices))),
                "original_row_index": keep_indices,
                "user_id": [uid_all[i] for i in keep_indices],
            })
            map_df.to_csv(os.path.join(args.save_dir, "row_index_map.csv"), index=False)
            parse_fail_df.to_csv(os.path.join(args.save_dir, "parse_failures.csv"), index=False)
        if dist.is_initialized(): dist.barrier()

        # Step 2: clustering on all kept rows (each rank does it, results are consistent; or can be changed to only rank0 computes and broadcasts)
        A_np, assign_np, _ = step2_cluster(E_np, k=args.k)
        if is_main():
            torch.save(torch.from_numpy(A_np), os.path.join(args.save_dir, "A_init.pt"))
            torch.save(torch.from_numpy(assign_np.astype(np.int64)), os.path.join(args.save_dir, "assign_init.pt"))
            save_members_pt(args.save_dir, "assign_init.pt", "row_index.pt", "row_index_map.csv", "members_init.pt")
        df_kept = df.iloc[keep_indices].reset_index(drop=True)
        if dist.is_initialized(): dist.barrier()

    # Step 1B: parse training fields on df_kept (strict q/resp/chose; flexible history)
    samples_tr, parsed_summary, drop_df, histories_text = step1b_parse_training_fields(
        df_kept, backend,
        user_input_col=args.user_input_col,
        fewshots_col=args.fewshots_col,
        chose_col=args.chose_col,
    )
    if is_main():
        parsed_summary.to_csv(os.path.join(args.save_dir, "parsed_summary.csv"), index=False)
        drop_df.to_csv(os.path.join(args.save_dir, "train_drop_reasons.csv"), index=False)
        print(f"Training kept: {(parsed_summary['train_kept']).sum()} / {len(parsed_summary)}; dropped: {len(drop_df)}")

    # Build training subset indices relative to clustering set
    train_indices = [i for i, k in enumerate(parsed_summary["train_kept"].tolist()) if k]

    # Save history artifacts for training subset
    if is_main():
        row_index_t_all = torch.load(os.path.join(args.save_dir, "row_index.pt"))
        save_history_artifacts(samples_tr, train_indices, row_index_t_all, args.save_dir)

    # Step 3: training (optional)
    has_train = len(samples_tr) > 0 and args.epochs > 0
    if has_train:
        A_tr = step3_update(samples_tr, A_np, assign_np, train_indices,
                            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            rho=args.rho, lambda_q=args.lambda_q, lambda_s=args.lambda_s,
                            lambda_cent=args.lambda_cent, lambda_tr=args.lambda_tr,
                            ema_decay=args.ema_decay, oversample=args.oversample,
                            device=args.device, distributed=args.distributed, local_rank=getattr(args,'local_rank',0))
        if is_main():
            torch.save(torch.from_numpy(A_tr), os.path.join(args.save_dir, "A_trained.pt"))
            # Re-assign **all N rows** using trained prototypes
            E_all = torch.load(os.path.join(args.save_dir, "E.pt")).cpu().numpy()
            assign_tr_all = (E_all @ A_tr.T).argmax(axis=1)
            torch.save(torch.from_numpy(assign_tr_all.astype(np.int64)), os.path.join(args.save_dir, "assign_trained.pt"))
            save_members_pt(args.save_dir, "assign_trained.pt", "row_index.pt", "row_index_map.csv", "members_trained.pt")
            print("Saved trained prototypes and per-prototype members (all rows reassigned).")
    else:
        if is_main():
            print("Step 3 skipped (epochs=0 or no rows passed strict training filter).")

    cleanup_distributed()

if __name__ == "__main__":
    main()
