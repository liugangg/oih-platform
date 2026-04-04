"""
ML Tools Router - ESM & Chemprop
"""
# --- SYNC_NOTES (auto-generated from CLAUDE.md, do not edit) ---
# CHEMPROP+ESM notes (from CLAUDE.md, do not manually edit):
#   - 1. **Dockerfile must include**: `RUN ln -sf /usr/bin/python3.11 /usr/bin/python3` (container python3 defaults to 3.10)
#   - 2. **predict calls must add**: `--accelerator cpu` (small batches on GPU cause OOM), assigned to `_CPU_TOOLS` queue
#   - 3. **`--devices 1` = "use 1 GPU device"**, not device index=1. Train uses `--accelerator gpu --devices 1`
#   - Root cause: container `python3` symlink points to 3.10, but all packages (torch/numpy/chemprop) installed under python3.11 path → `No module` errors
#   - Fix: `docker exec oih-chemprop bash -c "rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3"`
#   - Router fix: `routers/ml_tools.py` adds `--accelerator cpu` to avoid GPU OOM; `task_manager.py` assigns `chemprop_predict` to CPU queue
#   - Verified: 3-molecule prediction completed, CPU queue, 5 seconds
#   - Container NVIDIA_VISIBLE_DEVICES=1, always use device=0 / gpu_id=0 (not 1)
# --- /SYNC_NOTES ---

import os, json
from fastapi import APIRouter
from schemas.models import ESMRequest, ESMScoreRequest, ESMMutantScanRequest, ChempropRequest, TaskRef
from core.task_manager import task_manager
from core.docker_client import run_in_container_streaming
from core.config import settings

router = APIRouter()

@router.post("/esm/embed", response_model=TaskRef, summary="ESM2 protein embedding")
async def run_esm_embed(req: ESMRequest):
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "esm")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        task.progress = 10
        task.progress_msg = "Preparing ESM2 script..."
        script_path = f"/data/oih/tmp/{req.job_name}_esm_run.py"
        result_path = f"/data/oih/outputs/{req.job_name}/esm/result.json"
        script = f"""
import esm, torch, json, os
model_name = "{req.model_name}"
repr_layer = {req.repr_layer}
sequences = {json.dumps(req.sequences)}
model, alphabet = esm.pretrained.__dict__[model_name]()
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
batch_converter = alphabet.get_batch_converter()
data = [(f"seq_{{i}}", s) for i, s in enumerate(sequences)]
_, _, tokens = batch_converter(data)
if torch.cuda.is_available():
    tokens = tokens.cuda()
with torch.no_grad():
    results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
token_reps = results["representations"][repr_layer]
mean_reps = []
for i, (_, seq) in enumerate(data):
    mean_reps.append(token_reps[i, 1:len(seq)+1].mean(0).cpu().tolist())
output = {{"task": "{req.task.value}", "num_sequences": len(sequences),
           "embedding_dim": len(mean_reps[0]) if mean_reps else 0,
           "mean_embeddings": mean_reps}}
if "{req.task.value}" == "similarity" and len(sequences) >= 2:
    import torch.nn.functional as F
    vecs = torch.tensor(mean_reps)
    sim = F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0), dim=-1)
    output["similarity_matrix"] = sim.tolist()
os.makedirs(os.path.dirname("{result_path}"), exist_ok=True)
with open("{result_path}", "w") as f:
    json.dump(output, f)
print("Done:", output["num_sequences"], "seqs, dim=", output["embedding_dim"])
"""
        os.makedirs(f"{settings.DATA_ROOT}/tmp", exist_ok=True)
        with open(script_path.replace("/data/oih", settings.DATA_ROOT), "w") as f:
            f.write(script)
        task.progress = 20
        task.progress_msg = f"Running ESM2 on {len(req.sequences)} sequence(s)..."
        retcode, out = await run_in_container_streaming(
            settings.CONTAINER_ESM, ["python3", script_path], task,
            timeout=settings.TIMEOUT_ESM)
        if retcode != 0:
            raise RuntimeError(f"ESM failed: {out}")
        local_result = result_path.replace("/data/oih", settings.DATA_ROOT)
        with open(local_result) as f:
            data = json.load(f)
        task.progress_msg = f"ESM done: {data['num_sequences']} seqs, dim={data['embedding_dim']}"
        return {"mean_embeddings": data.get("mean_embeddings"),
                "similarity_matrix": data.get("similarity_matrix"),
                "output_dir": output_dir, "task": req.task.value}

    task = await task_manager.submit("esm", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="esm",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


@router.post("/esm/score", response_model=TaskRef, summary="ESM2 pseudo-perplexity scoring")
async def run_esm_score(req: ESMScoreRequest):
    """Score sequences by ESM2 pseudo-perplexity. Lower = more natural/foldable."""
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "esm_score")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        task.progress = 10
        task.progress_msg = f"Scoring {len(req.sequences)} sequences with ESM2..."
        script = f"""
import esm, torch, json, os, math

model_name = "{req.model_name}"
sequences = {json.dumps(req.sequences)}

model, alphabet = esm.pretrained.__dict__[model_name]()
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
batch_converter = alphabet.get_batch_converter()

results = []
for i, seq in enumerate(sequences):
    data = [(f"seq_{{i}}", seq)]
    _, _, tokens = batch_converter(data)
    if torch.cuda.is_available():
        tokens = tokens.cuda()

    # Masked marginal probability: mask each position, predict, get log-prob
    log_probs = []
    with torch.no_grad():
        for pos in range(1, len(seq) + 1):  # skip BOS token at 0
            masked = tokens.clone()
            masked[0, pos] = alphabet.mask_idx
            out = model(masked)["logits"]
            log_p = torch.log_softmax(out[0, pos], dim=-1)
            true_token = tokens[0, pos]
            log_probs.append(log_p[true_token].item())

    ppl = math.exp(-sum(log_probs) / len(log_probs)) if log_probs else 999.0
    mean_log_prob = sum(log_probs) / len(log_probs) if log_probs else -999.0
    results.append({{
        "sequence_index": i,
        "length": len(seq),
        "pseudo_perplexity": round(ppl, 4),
        "mean_log_probability": round(mean_log_prob, 4),
    }})

output = {{
    "scores": results,
    "model": model_name,
    "num_sequences": len(sequences),
}}
result_path = "/data/oih/outputs/{req.job_name}/esm_score/result.json"
os.makedirs(os.path.dirname(result_path), exist_ok=True)
with open(result_path, "w") as f:
    json.dump(output, f)
print(json.dumps({{"n": len(results), "best_ppl": min(r["pseudo_perplexity"] for r in results)}}))
"""
        script_path = f"/data/oih/tmp/{req.job_name}_esm_score.py"
        local_script = script_path.replace("/data/oih", settings.DATA_ROOT)
        os.makedirs(os.path.dirname(local_script), exist_ok=True)
        with open(local_script, "w") as f:
            f.write(script)

        task.progress = 20
        retcode, out = await run_in_container_streaming(
            settings.CONTAINER_ESM, ["python3", script_path], task,
            timeout=settings.TIMEOUT_ESM)
        if retcode != 0:
            raise RuntimeError(f"ESM score failed: {out}")

        local_result = f"{settings.OUTPUT_DIR}/{req.job_name}/esm_score/result.json"
        with open(local_result) as f:
            data = json.load(f)
        task.progress_msg = f"Scored {data['num_sequences']} sequences, best PPL={min(s['pseudo_perplexity'] for s in data['scores']):.2f}"
        return data

    task = await task_manager.submit("esm", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="esm_score",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


@router.post("/esm/mutant_scan", response_model=TaskRef, summary="ESM-1v in-silico mutant scanning")
async def run_esm_mutant_scan(req: ESMMutantScanRequest):
    """Predict effect of every point mutation at specified positions using ESM-1v."""
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "esm_mutant_scan")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        n_pos = len(req.scan_positions) if req.scan_positions else len(req.sequence)
        task.progress = 10
        task.progress_msg = f"Scanning {n_pos} positions with ESM-1v..."
        script = f"""
import esm, torch, json, os

sequence = "{req.sequence}"
scan_positions = {json.dumps(req.scan_positions)}  # 1-based or None=all
model_name = "{req.model_name}"

model, alphabet = esm.pretrained.__dict__[model_name]()
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
batch_converter = alphabet.get_batch_converter()

data = [("wt", sequence)]
_, _, tokens = batch_converter(data)
if torch.cuda.is_available():
    tokens = tokens.cuda()

# Get WT log-probs at all positions
with torch.no_grad():
    wt_logits = model(tokens)["logits"]  # [1, L+2, vocab]
    wt_log_probs = torch.log_softmax(wt_logits[0], dim=-1)  # [L+2, vocab]

# Standard amino acids
AA = "ACDEFGHIKLMNPQRSTVWY"
aa_indices = [alphabet.get_idx(a) for a in AA]

positions = scan_positions if scan_positions else list(range(1, len(sequence) + 1))
results = []
for pos in positions:  # 1-based
    tok_pos = pos  # tokens have BOS at 0, so position p maps to index p
    wt_aa = sequence[pos - 1]
    wt_idx = alphabet.get_idx(wt_aa)
    wt_score = wt_log_probs[tok_pos, wt_idx].item()

    mutations = []
    for aa, aa_idx in zip(AA, aa_indices):
        if aa == wt_aa:
            continue
        mut_score = wt_log_probs[tok_pos, aa_idx].item()
        ddg_proxy = -(mut_score - wt_score)  # positive = destabilizing
        mutations.append({{
            "mutation": f"{{wt_aa}}{{pos}}{{aa}}",
            "score_wt": round(wt_score, 4),
            "score_mut": round(mut_score, 4),
            "delta": round(mut_score - wt_score, 4),
            "ddg_proxy": round(ddg_proxy, 4),
        }})
    mutations.sort(key=lambda m: m["delta"], reverse=True)
    results.append({{
        "position": pos,
        "wt_residue": wt_aa,
        "top_stabilizing": mutations[:3],
        "top_destabilizing": mutations[-3:],
        "all_mutations": mutations,
    }})

output = {{
    "positions_scanned": len(results),
    "sequence_length": len(sequence),
    "model": model_name,
    "scan_results": results,
}}
result_path = "/data/oih/outputs/{req.job_name}/esm_mutant_scan/result.json"
os.makedirs(os.path.dirname(result_path), exist_ok=True)
with open(result_path, "w") as f:
    json.dump(output, f)
print(json.dumps({{"positions": len(results)}}))
"""
        script_path = f"/data/oih/tmp/{req.job_name}_esm_mutant.py"
        local_script = script_path.replace("/data/oih", settings.DATA_ROOT)
        os.makedirs(os.path.dirname(local_script), exist_ok=True)
        with open(local_script, "w") as f:
            f.write(script)

        task.progress = 20
        retcode, out = await run_in_container_streaming(
            settings.CONTAINER_ESM, ["python3", script_path], task,
            timeout=settings.TIMEOUT_ESM)
        if retcode != 0:
            raise RuntimeError(f"ESM mutant scan failed: {out}")

        local_result = f"{settings.OUTPUT_DIR}/{req.job_name}/esm_mutant_scan/result.json"
        with open(local_result) as f:
            data = json.load(f)
        task.progress_msg = f"Scanned {data['positions_scanned']} positions"
        return data

    task = await task_manager.submit("esm", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="esm_mutant_scan",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


@router.post("/chemprop/predict", response_model=TaskRef, summary="Chemprop property prediction")
async def run_chemprop_predict(req: ChempropRequest):
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "chemprop")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        task.progress = 10
        task.progress_msg = "Preparing input CSV..."
        smiles_csv = f"/data/oih/tmp/{req.job_name}_smiles.csv"
        local_csv = smiles_csv.replace("/data/oih", settings.DATA_ROOT)
        os.makedirs(os.path.dirname(local_csv), exist_ok=True)
        with open(local_csv, "w") as f:
            f.write("smiles\n")
            for s in req.smiles:
                f.write(s + "\n")
        result_csv = f"/data/oih/outputs/{req.job_name}/chemprop/predictions.csv"
        cmd = ["python3", "-m", "chemprop.cli.predict",
               "--test-path", smiles_csv,
               "--model-path", req.model_path,
               "--preds-path", result_csv,
               "--accelerator", "cpu"]
        task.progress = 20
        task.progress_msg = f"Predicting {len(req.smiles)} molecule(s)..."
        retcode, out = await run_in_container_streaming(
            settings.CONTAINER_CHEMPROP, cmd, task,
            timeout=settings.TIMEOUT_CHEMPROP)
        if retcode != 0:
            raise RuntimeError(f"Chemprop predict failed: {out}")
        import csv
        predictions = []
        local_result = result_csv.replace("/data/oih", settings.DATA_ROOT)
        if os.path.exists(local_result):
            with open(local_result) as f:
                for row in csv.DictReader(f):
                    predictions.append(dict(row))
        task.progress_msg = f"Predicted {len(predictions)} molecule(s)."
        return {"predictions": predictions, "model_output_dir": None,
                "output_dir": output_dir, "task": "predict"}

    task = await task_manager.submit("chemprop_predict", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="chemprop",
                   poll_url=f"/api/v1/tasks/{task.task_id}")


@router.post("/chemprop/train", response_model=TaskRef, summary="Train Chemprop model")
async def run_chemprop_train(req: ChempropRequest):
    output_dir = os.path.join(settings.OUTPUT_DIR, req.job_name, "chemprop_model")
    os.makedirs(output_dir, exist_ok=True)

    async def _run(task):
        task.progress = 10
        task.progress_msg = "Starting Chemprop training..."
        model_save_dir = f"/data/oih/outputs/{req.job_name}/chemprop_model"
        cmd = ["python3", "-m", "chemprop.cli.train",
               "--data-path", f"/data/oih/inputs/{req.train_csv}",
               "--save-dir", model_save_dir,
               "--epochs", str(req.epochs),
               "--accelerator", "gpu", "--devices", "1"]  # devices=1 means "use 1 GPU", NOT device index
        if req.target_columns:
            cmd += ["--target-columns"] + req.target_columns
        retcode, out = await run_in_container_streaming(
            settings.CONTAINER_CHEMPROP, cmd, task,
            timeout=settings.TIMEOUT_CHEMPROP)
        if retcode != 0:
            raise RuntimeError(f"Chemprop train failed: {out}")
        task.progress_msg = f"Training done. Model: {model_save_dir}"
        return {"predictions": None, "model_output_dir": output_dir,
                "output_dir": output_dir, "task": "train"}

    task = await task_manager.submit("chemprop_train", req.model_dump(), _run)
    return TaskRef(task_id=task.task_id, status=task.status, tool="chemprop_train",
                   poll_url=f"/api/v1/tasks/{task.task_id}")
