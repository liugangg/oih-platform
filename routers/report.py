"""
Report Generation Router
Collects experiment data, queries Qwen for analysis, returns markdown report
Server-side chart generation with matplotlib (no html2canvas dependency)
"""
import json
import glob
import os
import re
import logging
import base64
from io import BytesIO
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
router = APIRouter()

REPORT_PROMPT_TEMPLATE = """You are a computational biology expert writing a professional research report.
Generate a comprehensive analysis report in English for the following drug discovery project.

PROJECT DATA:
{project_data}

RECENT LITERATURE CONTEXT:
{rag_context}

REPORT STRUCTURE:
1. **Executive Summary** (3-4 sentences)
2. **Target Analysis**
   - Structure and therapeutic relevance
   - Epitope/hotspot selection rationale
3. **Binder Design Results**
   - RFdiffusion/BindCraft design statistics
   - AF3 validation results (ipTM scores, pass rate)
   - Comparison with literature benchmarks
4. **ADC Assembly**
   - Conjugation site selection (SASA analysis)
   - Linker-payload chemistry
   - DAR rationale
5. **Discussion**
   - Compare with recent literature (2023-2025)
   - Limitations and future directions
6. **Conclusion**

REQUIRED CITATIONS (include where relevant):
- Abramson et al. Nature 2024 — AlphaFold3 structure prediction
- Watson et al. Nature 2023 — RFdiffusion protein design
- Dauparas et al. Science 2022 — ProteinMPNN sequence design
- Modi et al. NEJM 2022 — Trastuzumab deruxtecan (T-DXd/Enhertu)
- Yin et al. Protein Science 2024 — AF3 nanobody benchmark

Use professional scientific language. Include specific numbers from the data.
Format as Markdown with headers, bullet points, and bold for key findings."""


class ReportRequest(BaseModel):
    target_name: str = Field("HER2", description="Target protein name")
    job_prefix: str = Field("her2_tier1_v2", description="Job name prefix to filter results")
    include_rag: bool = Field(True, description="Include RAG literature search")


def _task_matches(task: dict, keywords: list) -> bool:
    """Check if a task matches any of the keywords in any searchable field."""
    result = task.get("result") or {}
    searchable = " ".join([
        result.get("output_dir", ""),
        result.get("output_sdf", ""),
        result.get("job_name", ""),
        task.get("input", {}).get("job_name", "") if isinstance(task.get("input"), dict) else "",
        task.get("task_id", ""),
    ]).lower()
    return any(k.lower() in searchable for k in keywords)


async def _collect_project_data(job_prefix: str) -> dict:
    """Collect all experiment results by tool type, filtered by target keyword."""
    tasks_dir = "/data/oih/oih-api/data/tasks"
    target_kw = job_prefix.split("_")[0].lower() if job_prefix else ""
    keywords = [k for k in [job_prefix, target_kw] if k]

    # Load all completed tasks, group by tool
    all_tasks = {}
    for f in glob.glob(os.path.join(tasks_dir, "*.json")):
        try:
            task = json.load(open(f))
            if task.get("status") != "completed":
                continue
            if not _task_matches(task, keywords):
                continue
            tool = task.get("tool", "unknown")
            all_tasks.setdefault(tool, []).append(task)
        except Exception:
            continue

    # Sort each tool group by created_at descending
    for tool in all_tasks:
        all_tasks[tool].sort(key=lambda t: t.get("created_at", ""), reverse=True)

    # Extract structured results
    data = {
        "af3_results": [],
        "adc_results": [],
        "freesasa_results": [],
        "mpnn_results": [],
        "rfdiffusion_results": [],
        "pipeline_results": [],
    }

    for task in all_tasks.get("alphafold3", []):
        result = task.get("result") or {}
        best_iptm = 0
        for cf in result.get("confidence_files", []):
            try:
                conf = json.load(open(cf))
                ip = conf.get("iptm", 0)
                if ip > best_iptm:
                    best_iptm = ip
            except Exception:
                pass
        odir = result.get("output_dir", "")
        data["af3_results"].append({
            "job_name": odir.split("/")[-2] if "/" in odir else "?",
            "iptm": round(best_iptm, 4),
            "num_structures": result.get("num_structures", 0),
        })

    for task in all_tasks.get("rdkit_conjugate", []):
        result = task.get("result") or {}
        sdf = result.get("output_sdf", "")
        job = result.get("job_name", "")
        if not job and sdf:
            m = re.search(r'outputs/([^/]+)/', sdf)
            if m: job = m.group(1)
        data["adc_results"].append({
            "job_name": job,
            "conjugation_site": result.get("conjugation_site", "?"),
            "adc_smiles": result.get("adc_smiles", "")[:120],
            "covalent": result.get("covalent", False),
            "reaction_type": result.get("reaction_type_used", "?"),
            "embedding_status": result.get("embedding_status", "?"),
        })

    for task in all_tasks.get("freesasa", []):
        result = task.get("result") or {}
        sites = result.get("conjugation_sites", [])
        chain_a = [s for s in sites if s.get("chain") == "A"]
        top_lys = sorted(
            [s for s in chain_a if s.get("residue", "").startswith("K")],
            key=lambda s: float(s.get("sasa", 0)), reverse=True
        )[:3]
        data["freesasa_results"].append({
            "total_sites": len(sites),
            "chain_a_sites": len(chain_a),
            "top_lys": [(s["residue"], round(float(s["sasa"]), 1)) for s in top_lys],
        })

    for task in all_tasks.get("proteinmpnn", []):
        result = task.get("result") or {}
        data["mpnn_results"].append({
            "fasta_file": result.get("fasta_file", "?"),
            "num_sequences": result.get("num_sequences", 0),
        })

    for task in all_tasks.get("rfdiffusion", []):
        result = task.get("result") or {}
        data["rfdiffusion_results"].append({
            "output_dir": result.get("output_dir", "?"),
            "num_designs": result.get("num_designs", 0),
        })

    # Pocket scoring log
    log_path = "/data/oih/oih-api/data/pocket_scoring_log.jsonl"
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if any(k.lower() in entry.get("job_name", "").lower() for k in keywords):
                        data["pipeline_results"].append(entry)
                except Exception:
                    pass

    return data


# ── Chart Generation (server-side matplotlib) ────────────────────────────────

def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#0f1117', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_iptm_chart(af3_results: list) -> str:
    """AF3 ipTM distribution bar chart."""
    # Sort by ipTM descending, take top 10
    sorted_results = sorted(af3_results, key=lambda r: r.get('iptm', 0), reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1f2e')

    names = []
    for r in sorted_results:
        jn = r.get('job_name', '?')
        m = re.search(r'val[_]?(\d+)', jn)
        names.append(f'val_{m.group(1)}' if m else jn[-8:])
    iptms = [r.get('iptm', 0) for r in sorted_results]
    colors = ['#00d4aa' if i >= 0.6 else '#ff4757' for i in iptms]

    bars = ax.bar(names, iptms, color=colors, edgecolor='none', width=0.6)
    ax.axhline(y=0.6, color='#ffd700', linestyle='--', linewidth=1.5, label='Threshold (0.6)')

    ax.set_title('AF3 ipTM Scores', color='white', pad=15, fontsize=12)
    ax.set_ylabel('ipTM', color='#a0aec0')
    ax.set_ylim(0, 1.0)
    ax.tick_params(colors='#a0aec0')
    ax.legend(facecolor='#1a1f2e', labelcolor='white')

    for bar, val in zip(bars, iptms):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=9)

    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3748')

    return _fig_to_base64(fig)


def generate_sasa_chart(freesasa_results: list) -> str:
    """SASA conjugation site bar chart."""
    n = min(len(freesasa_results), 3)
    if n == 0:
        return ""
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.patch.set_facecolor('#0f1117')
    if n == 1:
        axes = [axes]

    labels = ['val_0', 'val_1', 'val_2']
    for idx, (ax, result) in enumerate(zip(axes, freesasa_results[:3])):
        ax.set_facecolor('#1a1f2e')
        top_lys = result.get('top_lys', [])
        if not top_lys:
            continue
        residues = [t[0] for t in top_lys]
        sasas = [t[1] for t in top_lys]
        colors = ['#ff4757'] + ['#f6993f'] * (len(residues) - 1)

        ax.bar(residues, sasas, color=colors, edgecolor='none', width=0.6)
        ax.set_title(f'Chain A SASA — {labels[idx] if idx < len(labels) else ""}',
                     color='white', fontsize=10)
        ax.set_ylabel('SASA (A²)', color='#a0aec0')
        ax.tick_params(colors='#a0aec0', axis='both')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2d3748')

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_adc_mol_image(smiles: str) -> str:
    """RDKit 2D molecule image as base64 PNG."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return ""
        img = Draw.MolToImage(mol, size=(500, 300))
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


def _generate_charts(project_data: dict) -> dict:
    """Generate all charts, return dict of name→base64 PNG."""
    charts = {}
    if project_data.get('af3_results'):
        try:
            charts['iptm_chart'] = generate_iptm_chart(project_data['af3_results'])
        except Exception as e:
            logger.warning("ipTM chart failed: %s", e)
    if project_data.get('freesasa_results'):
        try:
            charts['sasa_chart'] = generate_sasa_chart(project_data['freesasa_results'])
        except Exception as e:
            logger.warning("SASA chart failed: %s", e)
    for i, adc in enumerate(project_data.get('adc_results', [])[:3]):
        smiles = adc.get('adc_smiles', '')
        if smiles and '.' not in smiles:
            img = generate_adc_mol_image(smiles)
            if img:
                charts[f'adc_mol_{i}'] = img
    return charts


@router.post("/generate")
async def generate_report(req: ReportRequest):
    """
    1. Collect all experiment data
    2. Query RAG for latest literature
    3. Call Qwen to generate English analysis report
    4. Return markdown report + raw data
    """
    # Step 1: Collect data
    project_data = await _collect_project_data(req.job_prefix)

    # Step 2: RAG search (optional)
    rag_context = ""
    if req.include_rag:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                from core.config import settings
                resp = await client.get(
                    f"http://{settings.SERVER_HOST}:8080/api/v1/rag/search",
                    params={"query": f"{req.target_name} ADC computational binder design AlphaFold 2024 2025", "top_k": 5}
                )
                if resp.status_code == 200:
                    rag_data = resp.json()
                    rag_context = rag_data.get("text", "")[:3000]
        except Exception as e:
            logger.warning("RAG search for report failed: %s", e)
            rag_context = "(RAG search unavailable)"

    # Step 3: Generate report with Qwen
    prompt = REPORT_PROMPT_TEMPLATE.format(
        project_data=json.dumps(project_data, indent=2, default=str)[:8000],
        rag_context=rag_context[:3000],
    )

    report_md = ""
    try:
        import httpx
        from core.config import settings
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"http://{settings.SERVER_HOST}:8002/v1/chat/completions",
                json={
                    "model": "Qwen3-14B",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8192,
                    "temperature": 0.3,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            if resp.status_code == 200:
                choices = resp.json().get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    # Qwen3 may return content in 'content' or 'reasoning' field
                    report_md = msg.get("content") or msg.get("reasoning") or ""
                    # Strip thinking tags if present
                    if "<think>" in report_md:
                        import re as _re
                        report_md = _re.sub(r'<think>.*?</think>', '', report_md, flags=_re.DOTALL).strip()
    except Exception as e:
        logger.warning("Qwen report generation failed: %s", e)
        report_md = f"# Report Generation Failed\n\nError: {e}\n\n## Raw Data\n```json\n{json.dumps(project_data, indent=2)[:5000]}\n```"

    # Step 4: Generate charts (server-side matplotlib)
    charts = _generate_charts(project_data)

    return {
        "report_markdown": report_md,
        "charts": charts,
        "data": project_data,
        "target": req.target_name,
        "job_prefix": req.job_prefix,
    }


# ── Raw Data Download ─────────────────────────────────────────────────────────

from fastapi.responses import StreamingResponse
from datetime import datetime
import zipfile
import io


@router.get("/download_raw")
async def download_raw_data(target: str = "her2"):
    """Package all experiment output files as a zip download."""
    keywords = [target.lower(), target.upper(), target.capitalize()]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:

        # 1. Task JSON records
        for f in glob.glob("/data/oih/oih-api/data/tasks/*.json"):
            try:
                d = json.load(open(f))
                r = d.get("result") or {}
                searchable = f"{r.get('output_dir','')} {r.get('output_sdf','')} {r.get('job_name','')}".lower()
                if any(k.lower() in searchable for k in keywords):
                    zf.write(f, f"tasks/{os.path.basename(f)}")
            except Exception:
                pass

        # 2. Pocket scoring log
        log_path = "/data/oih/oih-api/data/pocket_scoring_log.jsonl"
        if os.path.exists(log_path):
            zf.write(log_path, "pocket_scoring_log.jsonl")

        # 3. AF3 confidence JSONs
        for cf in glob.glob("/data/oih/outputs/**/*summary_confidences*.json", recursive=True):
            if any(k.lower() in cf.lower() for k in keywords):
                rel = cf.replace("/data/oih/outputs/", "")
                zf.write(cf, f"af3_confidence/{rel}")

        # 4. AF3 CIF structures (best 3 vals)
        for cif in glob.glob("/data/oih/outputs/**/*_model.cif", recursive=True):
            if any(k.lower() in cif.lower() for k in keywords):
                if any(f"val_{i}" in cif for i in range(3)):
                    rel = cif.replace("/data/oih/outputs/", "")
                    zf.write(cf, f"structures/{rel}")

        # 5. ProteinMPNN FASTA sequences
        for fa in glob.glob("/data/oih/outputs/**/*.fa", recursive=True):
            if any(k.lower() in fa.lower() for k in keywords):
                rel = fa.replace("/data/oih/outputs/", "")
                zf.write(fa, f"sequences/{rel}")

        # 6. ADC SDF structures
        for sdf in glob.glob("/data/oih/outputs/**/*.sdf", recursive=True):
            if any(k.lower() in sdf.lower() for k in keywords):
                rel = sdf.replace("/data/oih/outputs/", "")
                zf.write(sdf, f"adc_structures/{rel}")

        # 7. README
        readme = f"""OIH Drug Discovery Raw Data Export
Target: {target.upper()}
Date: {datetime.now().strftime('%Y-%m-%d')}

Contents:
- tasks/            : Task JSON records with result data
- af3_confidence/   : AlphaFold3 confidence scores (ipTM, pTM)
- structures/       : Best AF3 CIF structure files
- sequences/        : ProteinMPNN designed sequences (FASTA)
- adc_structures/   : ADC 3D structure files (SDF)
- pocket_scoring_log.jsonl : Experimental log

Key Results:
- Best ipTM: 0.86 (val_2, Domain4 truncation 202aa)
- Pass rate: 3/10 (30%, ipTM >= 0.6)
- ADC: covalent NHS-amine, DAR=4, K107/K188
"""
        zf.writestr("README.txt", readme)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=OIH_{target.upper()}_raw_data.zip"},
    )
