"""
Report Generation Router
Collects experiment data, queries Qwen for analysis, returns markdown report
"""
import json
import glob
import os
import logging
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List

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


async def _collect_project_data(job_prefix: str) -> dict:
    """Collect all experiment results matching the job prefix."""
    tasks_dir = "/data/oih/oih-api/data/tasks"
    data = {
        "af3_results": [],
        "adc_results": [],
        "mpnn_results": [],
        "freesasa_results": [],
        "pipeline_results": [],
    }

    for f in glob.glob(os.path.join(tasks_dir, "*.json")):
        try:
            task = json.load(open(f))
            if task.get("status") != "completed":
                continue
            result = task.get("result") or {}
            output_dir = result.get("output_dir", "")

            if job_prefix not in output_dir and job_prefix not in task.get("input", {}).get("job_name", ""):
                continue

            tool = task.get("tool", "")

            if tool == "alphafold3":
                best_iptm = 0
                for cf in result.get("confidence_files", []):
                    try:
                        conf = json.load(open(cf))
                        ip = conf.get("iptm", 0)
                        if ip > best_iptm:
                            best_iptm = ip
                    except Exception:
                        pass
                data["af3_results"].append({
                    "job_name": output_dir.split("/")[-2] if "/" in output_dir else "?",
                    "iptm": round(best_iptm, 4),
                    "num_structures": result.get("num_structures", 0),
                })

            elif tool == "rdkit_conjugate":
                data["adc_results"].append({
                    "job_name": task.get("input", {}).get("job_name", "?"),
                    "adc_smiles": result.get("adc_smiles", "")[:100],
                    "covalent": result.get("covalent", False),
                    "reaction_type": result.get("reaction_type_used", "?"),
                    "embedding_status": result.get("embedding_status", "?"),
                    "conjugation_site": task.get("input", {}).get("conjugation_site", "?"),
                })

            elif tool == "freesasa":
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

        except Exception as e:
            logger.debug("Report data collection error: %s", e)
            continue

    # Pocket scoring log
    log_path = "/data/oih/oih-api/data/pocket_scoring_log.jsonl"
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if job_prefix in entry.get("job_name", ""):
                        data["pipeline_results"].append(entry)
                except Exception:
                    pass

    return data


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
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"http://{settings.SERVER_HOST}:8002/v1/chat/completions",
                json={
                    "model": "Qwen3-14B",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                    "temperature": 0.3,
                },
            )
            if resp.status_code == 200:
                choices = resp.json().get("choices", [])
                if choices:
                    report_md = choices[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.warning("Qwen report generation failed: %s", e)
        report_md = f"# Report Generation Failed\n\nError: {e}\n\n## Raw Data\n```json\n{json.dumps(project_data, indent=2)[:5000]}\n```"

    return {
        "report_markdown": report_md,
        "data": project_data,
        "target": req.target_name,
        "job_prefix": req.job_prefix,
    }
