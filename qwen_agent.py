"""
Qwen3-14B Agent Integration
Complete example of connecting Qwen to the OIH tool chain

Deployment:
  GPU0 → vLLM serving Qwen3-14B
  GPU1 → OIH API Gateway (this FastAPI service)

Usage:
  python qwen_agent.py
  or call via HTTP POST /api/v1/agent/chat
"""

import asyncio
import json
import re
import httpx
from typing import Any, Dict, List, Optional

# ─── Config ───────────────────────────────────────────────────────────────────
import os as _os
_server_host  = _os.environ.get("OIH_SERVER_HOST", "localhost")
QWEN_BASE_URL = f"http://{_server_host}:8002/v1"   # vLLM on GPU0
QWEN_MODEL    = "Qwen3-14B"
OIH_API_BASE  = f"http://{_server_host}:8080/api/v1"

# ─── LLM Backend (provider-agnostic) ─────────────────────────────────────────
from core.llm_backend import create_backend, LLMBackend, LLMResponse
from core.config import settings as _cfg

_default_backend = create_backend(
    provider=_cfg.LLM_PROVIDER,
    api_key=_cfg.LLM_API_KEY,
    model=_cfg.LLM_MODEL,
    base_url=_cfg.LLM_BASE_URL,
    vllm_url=_cfg.QWEN_BASE_URL,
    vllm_model=_cfg.QWEN_MODEL,
)
print(f"[LLM] Default backend: {_default_backend.provider} ({getattr(_default_backend, 'model', '')})")

# ─── Load tool definitions ────────────────────────────────────────────────────
from tool_definitions.qwen_tools import ALL_TOOLS, QWEN_SYSTEM_PROMPT
from skills_loader import build_dynamic_system_prompt


# ─── Tool Executor ────────────────────────────────────────────────────────────

class OIHToolExecutor:
    """Executes tool calls by calling the OIH API endpoints"""

    # Map function name → HTTP method + endpoint template
    TOOL_MAP = {
        "fetch_pdb":               ("POST", f"{OIH_API_BASE}/structure/fetch_pdb"),
        "fetch_molecule":          ("POST", f"{OIH_API_BASE}/structure/fetch_molecule"),
        "alphafold3_predict":      ("POST", f"{OIH_API_BASE}/structure/alphafold3"),
        "rfdiffusion_design":      ("POST", f"{OIH_API_BASE}/design/rfdiffusion"),
        "proteinmpnn_sequence_design": ("POST", f"{OIH_API_BASE}/design/proteinmpnn"),
        "bindcraft_design":        ("POST", f"{OIH_API_BASE}/design/bindcraft"),
        "fpocket_detect_pockets":  ("POST", f"{OIH_API_BASE}/pocket/fpocket"),
        "p2rank_predict_pockets":  ("POST", f"{OIH_API_BASE}/pocket/p2rank"),
        "dock_ligand":             ("POST", f"{OIH_API_BASE}/docking/dock"),
        "diffdock_blind_dock":     ("POST", f"{OIH_API_BASE}/docking/diffdock"),
        "gromacs_md_simulation":   ("POST", f"{OIH_API_BASE}/md/gromacs"),
        "drug_discovery_pipeline": ("POST", f"{OIH_API_BASE}/pipeline/drug-discovery"),
        "binder_design_pipeline":  ("POST", f"{OIH_API_BASE}/pipeline/binder-design"),
        "pocket_guided_binder_pipeline": ("POST", f"{OIH_API_BASE}/pipeline/pocket-guided-binder"),
        "freesasa":                ("POST", f"{OIH_API_BASE}/design/freesasa"),
        "linker_select":           ("POST", f"{OIH_API_BASE}/adc/linker_select"),
        "rdkit_conjugate":         ("POST", f"{OIH_API_BASE}/adc/rdkit_conjugate"),
        "get_system_status":       ("GET",  f"{OIH_API_BASE}/system/status"),
        "poll_task_status":        ("GET",  f"{OIH_API_BASE}/tasks/{{task_id}}"),
        "esm_embed":               ("POST", f"{OIH_API_BASE}/ml/esm/embed"),
        "esm2_score_sequences":    ("POST", f"{OIH_API_BASE}/ml/esm/score"),
        "esm2_mutant_scan":        ("POST", f"{OIH_API_BASE}/ml/esm/mutant_scan"),
        "chemprop_predict":        ("POST", f"{OIH_API_BASE}/ml/chemprop/predict"),
        "rag_search":              ("GET",  f"{OIH_API_BASE}/rag/search"),
        "search_literature":       ("GET",  f"{OIH_API_BASE}/rag/search"),
        "execute_python":          ("POST", f"{OIH_API_BASE}/analysis/execute_python"),
        "read_results_file":       ("POST", f"{OIH_API_BASE}/analysis/read_results_file"),
        "discotope3_predict":      ("POST", f"{OIH_API_BASE}/immunology/discotope3"),
        "igfold_predict":          ("POST", f"{OIH_API_BASE}/immunology/igfold"),
        "extract_interface_residues": ("POST", f"{OIH_API_BASE}/structure/extract_interface"),
        "generate_report":         ("POST", f"{OIH_API_BASE}/report/generate"),
        "pesto_predict":           ("POST", f"{OIH_API_BASE}/structure/pesto_predict"),
        "ipsae_score":             ("POST", f"{OIH_API_BASE}/structure/ipsae_score"),
        "web_search":              ("POST", f"{OIH_API_BASE}/rag/web-search"),
    }

    async def execute(self, function_name: str, arguments: Dict[str, Any]) -> Dict:
        if function_name not in self.TOOL_MAP:
            return {"error": f"Unknown tool: {function_name}"}

        method, url_template = self.TOOL_MAP[function_name]

        # Fill URL template (for GET with path params)
        url = url_template.format(**arguments)

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                if method == "GET":
                    # Pass remaining arguments as query params
                    # (url.format already consumed path-template keys like {task_id})
                    query_params = {k: v for k, v in arguments.items()
                                    if f"{{{k}}}" not in url_template}
                    resp = await client.get(url, params=query_params)
                else:
                    resp = await client.post(url, json=arguments)
            if resp.status_code not in (200, 201):
                return {"error": f"API error {resp.status_code}: {resp.text[:500]}"}
            return resp.json()
        except httpx.ConnectError as e:
            return {"error": f"Connection failed: {str(e)}"}
        except httpx.TimeoutException as e:
            return {"error": f"Request timeout: {str(e)}"}
        except Exception as e:
            return {"error": f"Tool execution error: {str(e)}"}


# ─── Qwen Agent Loop ──────────────────────────────────────────────────────────

class QwenBioAgent:
    def __init__(self, llm_backend: Optional[LLMBackend] = None):
        self.executor = OIHToolExecutor()
        self.conversation_history: List[Dict] = []
        self._llm_backend: LLMBackend = llm_backend or _default_backend

    SIMPLE_PATTERNS = [
        "hello", "hi", "hey", "morning", "good night", "thanks", "thank you",
        "ok", "okay", "got it", "bye", "goodbye", "nice", "👍", "haha",
    ]
    BIO_KEYWORDS = [
        "protein", "pdb", "smiles", "fasta", "drug", "antibody", "design",
        "predict", "analysis", "structure", "docking", "sequence", "target", "molecule", "nanobody",
        "alphafold", "gromacs", "docking", "md", "adc", "ligand",
    ]

    # ── Known target PDB mapping ──
    TARGET_PDB = {
        "her2": "1N8Z", "erbb2": "1N8Z", "pd-l1": "5XXY", "pdl1": "5XXY",
        "egfr": "1YY9", "vegf": "1BJ1", "tnf": "3WD5", "cd36": "5LGD",
        "nectin-4": "4GJT", "nectin4": "4GJT", "trop2": "7PEE", "trka": "1HE7",
        "cox-2": "5XWR", "cox2": "5XWR", "bcl-2": "6O0K", "bcl2": "6O0K",
        "tp53": "2XWR", "p53": "2XWR",
    }

    @staticmethod
    def is_simple_message(msg: str) -> bool:
        msg_lower = msg.lower().strip()
        if len(msg_lower) < 20:
            if any(p in msg_lower for p in QwenBioAgent.SIMPLE_PATTERNS):
                return True
            if not any(k in msg_lower for k in QwenBioAgent.BIO_KEYWORDS):
                return True
        return False

    @staticmethod
    def detect_fast_route(msg: str):
        """Pattern-match fixed workflows. Returns list of tool calls or None."""
        m = msg.lower().strip()

        # Find target name and PDB ID
        target_name = None
        pdb_id = None
        for name, pid in QwenBioAgent.TARGET_PDB.items():
            if name in m:
                target_name = name.upper()
                pdb_id = pid
                break
        # Also supports direct PDB ID input (4 chars: digit + 3 alphanumeric)
        import re as _re
        pdb_match = _re.search(r'(?<![A-Za-z0-9])([0-9][A-Za-z0-9]{3})(?![A-Za-z0-9])', msg)
        if pdb_match and not pdb_id:
            pdb_id = pdb_match.group(1).upper()

        # ── Route 1: AF3 structure prediction ──
        if any(k in m for k in ["alphafold", "af3", "predict structure", "structure prediction"]):
            if pdb_id:
                return "af3_predict", [
                    {"name": "fetch_pdb", "args": {"pdb_id": pdb_id}},
                ], f"Downloading {pdb_id}, will submit AlphaFold3 prediction after extracting sequence..."

        # Common drug name aliases
        _MOL_ALIASES = {}

        # ── Route 2: Molecular docking ──
        if any(k in m for k in ["dock", "docking"]):
            mol_match = _re.search(r'(?:and|with)\s*(\S+)', m)
            mol_name = mol_match.group(1) if mol_match else None
            if mol_name:
                mol_name = _MOL_ALIASES.get(mol_name, mol_name)
            if pdb_id and mol_name:
                return "docking", [
                    {"name": "fetch_pdb", "args": {"pdb_id": pdb_id}},
                    {"name": "fetch_molecule", "args": {"query": mol_name}},
                ], f"Downloading {pdb_id} and fetching {mol_name}, preparing docking..."

        # ── Route 3: Binder/ADC design (no specified residues) ──
        if any(k in m for k in ["design binder",
                                  "design antibody", "design nanobody", "design adc",
                                  "target.*binder"]) or \
           (_re.search(r'target.*(binder|antibody|nanobody)', m)):
            if pdb_id or target_name:
                job = f"binder_{(target_name or pdb_id).lower()}_{int(__import__('time').time())}"
                args = {"job_name": job}
                if pdb_id:
                    args["pdb_id"] = pdb_id
                if target_name:
                    args["target_name"] = target_name
                return "binder_pipeline", [
                    {"name": "pocket_guided_binder_pipeline", "args": args},
                ], f"Starting {target_name or pdb_id} binder design pipeline..."

        # ── Route 4: ADMET prediction ──
        if any(k in m for k in ["admet", "toxicity", "solubility"]):
            # Extract molecule name
            mol_match = _re.search(r'(?:of|for|predict|assess)\s*([a-zA-Z0-9_-]+)', m)
            mol_name = mol_match.group(1) if mol_match else None
            if mol_name:
                mol_name = _MOL_ALIASES.get(mol_name, mol_name)
                if mol_name and mol_name.lower() not in ["admet", "toxicity", "solubility"]:
                    return "admet", [
                        {"name": "fetch_molecule", "args": {"query": mol_name}},
                    ], f"Fetching {mol_name}, preparing ADMET prediction..."

        # ── Route 5: Pocket detection ──
        if any(k in m for k in ["pocket", "binding site", "fpocket", "p2rank", "detect pocket"]):
            if pdb_id:
                return "pocket", [
                    {"name": "fetch_pdb", "args": {"pdb_id": pdb_id}},
                ], f"Downloading {pdb_id}, preparing pocket detection..."

        # ── Route 6: Drug discovery full pipeline ──
        if any(k in m for k in ["drug discovery", "full pipeline", "end to end"]):
            mol_match = _re.search(r'(?:and|with|ligand)\s*(\S+)', m)
            mol_name = mol_match.group(1) if mol_match else None
            if pdb_id:
                job = f"drugdisc_{pdb_id}_{int(__import__('time').time())}"
                args = {"job_name": job, "pdb_id": pdb_id}
                if mol_name:
                    mol_name = _MOL_ALIASES.get(mol_name, mol_name)
                    args["ligand_name"] = mol_name
                return "drug_discovery", [
                    {"name": "drug_discovery_pipeline", "args": args},
                ], f"Starting {pdb_id} drug discovery full pipeline..."

        # ── Route 7: MD simulation ──
        if any(k in m for k in ["md simulation", "gromacs", "molecular dynamics"]):
            if pdb_id:
                return "md_simulation", [
                    {"name": "fetch_pdb", "args": {"pdb_id": pdb_id}},
                ], f"Downloading {pdb_id}, preparing MD simulation..."

        # ── Route 8: Epitope prediction ──
        if any(k in m for k in ["epitope", "discotope", "b-cell", "epitope prediction"]):
            if pdb_id:
                return "epitope", [
                    {"name": "fetch_pdb", "args": {"pdb_id": pdb_id}},
                ], f"Downloading {pdb_id}, preparing epitope prediction..."

        # ── Route 9: PDB download ──
        if any(k in m for k in ["download pdb", "fetch pdb", "get pdb", "get structure"]) or \
           (any(k in m for k in ["download", "fetch"]) and pdb_id):
            if pdb_id:
                return "fetch_pdb", [
                    {"name": "fetch_pdb", "args": {"pdb_id": pdb_id}},
                ], f"Downloading PDB {pdb_id}..."

        # ── Route 10: Molecule retrieval ──
        if any(k in m for k in ["fetch molecule", "find molecule", "smiles", "find drug"]):
            mol_match = _re.search(r'(?:fetch|find|get)\s*(?:molecule|drug)?\s*(\S+)', m)
            mol_name = mol_match.group(1) if mol_match else None
            if mol_name and mol_name not in ["molecule", "drug"]:
                return "fetch_molecule", [
                    {"name": "fetch_molecule", "args": {"query": mol_name}},
                ], f"Searching for {mol_name}..."

        # ── Route 11: Literature search ──
        if any(k in m for k in ["literature", "paper", "pubmed", "search paper"]):
            query = m
            return "literature", [
                {"name": "search_literature", "args": {"query": query}},
            ], f"Searching literature..."

        return None

    async def chat(self, user_message: str) -> str:
        """Single turn: user message → agent response with tool use"""
        self._cancelled = False
        self.conversation_history.append({"role": "user", "content": user_message})

        # ── Simple message fast path: skip skills, thinking_budget=0 ──
        skip_skills = self.is_simple_message(user_message)

        _SIMPLE_PROMPT = "You are the AI assistant of the OIH computational biology platform. You help users with protein structure prediction, drug design, molecular docking, and other computational biology tasks. Reply concisely. Do not reveal server IPs, GPU models, or internal paths."
        if skip_skills:
            dynamic_system = _SIMPLE_PROMPT
            print(f"[Simple] Using short prompt")
        else:
            dynamic_system, detected_skills = build_dynamic_system_prompt(
                QWEN_SYSTEM_PROMPT, user_message
            )
            if detected_skills:
                print(f"[Skills] Injected: {detected_skills}")

        max_iterations = 20   # prevent infinite loops
        iteration = 0
        tool_fail_count: Dict[str, int] = {}   # track consecutive failures per tool

        # Thinking mode disabled — Qwen3 works fine for tool calls without thinking, much faster
        thinking_budget = 0
        print(f"[Thinking] disabled (budget=0)")

        while iteration < max_iterations:
            iteration += 1

            # Call Qwen — fit conversation history within vLLM context limit
            # Budget: 32768 tokens total - 8192 output - ~8000 system/tools ≈ 16000 tokens for history
            # ~3 chars/token → ~48000 chars budget for history
            MAX_HISTORY_CHARS = 48000
            history_window = self.conversation_history[-10:]
            total_chars = sum(len(str(m.get("content", ""))) for m in history_window)
            while total_chars > MAX_HISTORY_CHARS and len(history_window) > 1:
                dropped = history_window.pop(0)
                total_chars -= len(str(dropped.get("content", "")))

            llm_response = await self._llm_backend.chat(
                messages=history_window,
                tools=None if skip_skills else ALL_TOOLS,
                system_prompt=dynamic_system,
                max_tokens=1024 if skip_skills else 8192,
                temperature=0.1,
            )

            # Add assistant message to history
            self.conversation_history.append(self._llm_backend.build_history_message(llm_response))

            # No tool calls → final answer
            if not llm_response.has_tool_calls:
                raw = (llm_response.content or "") or (llm_response.reasoning or "") or ""
                clean = re.sub(r"<[^>]+>", "", raw).strip()
                return clean

            # Execute tool calls
            print(f"\n[Agent] Calling {len(llm_response.tool_calls)} tool(s)...")

            for tc in llm_response.tool_calls:
                fn_name = tc.name
                fn_args = tc.arguments

                print(f"  → {fn_name}({list(fn_args.keys())})")

                # ── Skip tools that have failed ≥2 times consecutively ────
                if tool_fail_count.get(fn_name, 0) >= 2:
                    skip_msg = (
                        f"Tool {fn_name} has failed {tool_fail_count[fn_name]} times consecutively. "
                        f"Please skip this step and continue with the remaining workflow. Do not call this tool again."
                    )
                    print(f"     SKIPPED (consecutive failures={tool_fail_count[fn_name]})")
                    self.conversation_history.append(
                        self._llm_backend.build_tool_result_message(
                            tc.id, fn_name, json.dumps({"error": skip_msg, "skipped": True})))
                    continue

                # Execute
                result = await self.executor.execute(fn_name, fn_args)

                # If it's a task submission, auto-poll until complete
                if "task_id" in result and fn_name != "poll_task_status":
                    task_id = result["task_id"]
                    print(f"     Task submitted: {task_id}, polling...")
                    result = await self._poll_until_done(task_id)

                print(f"     Result: {str(result)[:200]}...")

                # ── Track failures ────────────────────────────────────────
                if result.get("error"):
                    tool_fail_count[fn_name] = tool_fail_count.get(fn_name, 0) + 1
                    print(f"     FAIL #{tool_fail_count[fn_name]} for {fn_name}")
                else:
                    tool_fail_count[fn_name] = 0  # reset on success

                # Add tool result to history (truncate large results to avoid context overflow)
                result_str = json.dumps(result, ensure_ascii=False)
                if len(result_str) > 4000:
                    result_str = result_str[:4000] + f"\n... [truncated, {len(result_str)} chars total]"
                self.conversation_history.append(
                    self._llm_backend.build_tool_result_message(tc.id, fn_name, result_str))

        return "Agent exceeded max iterations. Please check task status manually."

    async def _poll_until_done(self, task_id: str, max_wait: int = 7200) -> Dict:
        """Poll task until completed or failed"""
        elapsed = 0
        poll_interval = 15  # seconds

        while elapsed < max_wait:
            if getattr(self, "_cancelled", False):
                return {"error": "cancelled by reset"}
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            result = await self.executor.execute("poll_task_status", {"task_id": task_id})

            status = result.get("status")
            progress = result.get("progress", 0)
            msg = result.get("progress_msg", "")
            print(f"     [{elapsed}s] {status} {progress}% - {msg}")

            if status == "completed":
                return result
            elif status == "failed":
                error_msg = result.get("error", "Task failed")
                return {
                    "error": error_msg,
                    "task_id": task_id,
                    "instruction": "Analyze the error above to find the root cause, modify parameters and retry. Do not retry with the same parameters."
                }

        return {"error": f"Task {task_id} timed out after {max_wait}s"}

    async def chat_stream(self, user_message: str):
        """Streaming version of chat() — yields SSE event dicts at each step."""
        self._cancelled = False
        self.conversation_history.append({"role": "user", "content": user_message})

        # ── Fast route: execute fixed workflows directly, skip Qwen ──
        fast = self.detect_fast_route(user_message)
        if fast:
            route_name, tool_sequence, status_msg = fast
            yield {"type": "status", "content": f"⚡ Fast route: {status_msg}"}

            all_results = {}
            for step in tool_sequence:
                fn_name, fn_args = step["name"], step["args"]
                yield {"type": "tool_call", "tool": fn_name, "args": fn_args}
                result = await self.executor.execute(fn_name, fn_args)
                # Auto-poll async tasks
                if "task_id" in result and fn_name != "poll_task_status":
                    task_id = result["task_id"]
                    yield {"type": "task_submitted", "tool": fn_name, "task_id": task_id}
                    elapsed = 0
                    while elapsed < 7200:
                        if getattr(self, "_cancelled", False):
                            result = {"error": "cancelled"}
                            break
                        await asyncio.sleep(15)
                        elapsed += 15
                        result = await self.executor.execute("poll_task_status", {"task_id": task_id})
                        status = result.get("status")
                        progress = result.get("progress", 0)
                        msg = result.get("progress_msg", "")
                        yield {"type": "task_progress", "tool": fn_name, "task_id": task_id,
                               "status": status, "progress": progress, "message": msg}
                        if status in ("completed", "failed"):
                            break

                is_error = bool(result.get("error"))
                summary = str(result)[:300]
                yield {"type": "tool_result", "tool": fn_name, "success": not is_error, "summary": summary}
                all_results[fn_name] = result

                if is_error:
                    yield {"type": "answer", "content": f"Tool {fn_name} execution failed: {result.get('error', '')[:200]}"}
                    return

            # After fast route: pass results to Qwen for a summary round
            result_summary = json.dumps(all_results, ensure_ascii=False)[:3000]
            self.conversation_history.append({"role": "assistant", "content": None,
                "tool_calls": [{"id": "fast_0", "type": "function",
                    "function": {"name": tool_sequence[0]["name"], "arguments": json.dumps(tool_sequence[0]["args"])}}]})
            self.conversation_history.append({"role": "tool", "tool_call_id": "fast_0",
                "name": tool_sequence[0]["name"], "content": result_summary})

            # If AF3 route, extract sequence from PDB after fetch_pdb succeeds, then submit AF3
            if route_name == "af3_predict" and "fetch_pdb" in all_results:
                pdb_result = all_results["fetch_pdb"]
                pdb_path = pdb_result.get("output_pdb", "")
                # Extract sequence from PDB file
                seq = ""
                if pdb_path:
                    try:
                        aa3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
                                  "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
                                  "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"}
                        seen = set()
                        with open(pdb_path) as f:
                            for line in f:
                                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                                    resname = line[17:20].strip()
                                    resid = line[21:26].strip()
                                    if resid not in seen:
                                        seen.add(resid)
                                        seq += aa3to1.get(resname, "X")
                    except Exception:
                        pass
                    yield {"type": "status", "content": f"Sequence extracted: {len(seq)}aa"}
                if seq and len(seq) > 10:
                    job_name = f"{pdb_result.get('pdb_id','af3')}_af3_predict"
                    af3_args = {"job_name": job_name, "chains": [{"type": "protein", "sequence": seq}], "num_seeds": 1}
                    yield {"type": "tool_call", "tool": "alphafold3_predict", "args": {"job_name": job_name}}
                    af3_result = await self.executor.execute("alphafold3_predict", af3_args)
                    if "task_id" in af3_result:
                        task_id = af3_result["task_id"]
                        yield {"type": "task_submitted", "tool": "alphafold3_predict", "task_id": task_id}
                        elapsed = 0
                        while elapsed < 7200:
                            if getattr(self, "_cancelled", False):
                                break
                            await asyncio.sleep(15)
                            elapsed += 15
                            af3_result = await self.executor.execute("poll_task_status", {"task_id": task_id})
                            yield {"type": "task_progress", "tool": "alphafold3_predict", "task_id": task_id,
                                   "status": af3_result.get("status"), "progress": af3_result.get("progress", 0),
                                   "message": af3_result.get("progress_msg", "")}
                            if af3_result.get("status") in ("completed", "failed"):
                                break
                    yield {"type": "tool_result", "tool": "alphafold3_predict",
                           "success": af3_result.get("status") == "completed", "summary": str(af3_result)[:300]}
                    yield {"type": "answer", "content": f"AF3 structure prediction {'completed' if af3_result.get('status')=='completed' else 'failed'}."}
                else:
                    yield {"type": "answer", "content": f"PDB {pdb_result.get('pdb_id')} downloaded successfully, but no sequence could be extracted. Please provide an amino acid sequence to continue AF3 prediction."}
                return

            # Docking route: auto-call dock_ligand after both fetch_pdb + fetch_molecule succeed
            if route_name == "docking" and "fetch_pdb" in all_results and "fetch_molecule" in all_results:
                pdb_r = all_results["fetch_pdb"]
                mol_r = all_results["fetch_molecule"]
                dock_args = {
                    "input_pdb": pdb_r.get("output_pdb", ""),
                    "ligand_smiles": mol_r.get("smiles", ""),
                    "engine": "gnina",
                    "job_name": f"dock_{pdb_r.get('pdb_id','')}_{mol_r.get('cid','')}",
                }
                yield {"type": "tool_call", "tool": "dock_ligand", "args": dock_args}
                dock_result = await self.executor.execute("dock_ligand", dock_args)
                if "task_id" in dock_result:
                    task_id = dock_result["task_id"]
                    yield {"type": "task_submitted", "tool": "dock_ligand", "task_id": task_id}
                    elapsed = 0
                    while elapsed < 7200:
                        if getattr(self, "_cancelled", False):
                            break
                        await asyncio.sleep(15)
                        elapsed += 15
                        dock_result = await self.executor.execute("poll_task_status", {"task_id": task_id})
                        yield {"type": "task_progress", "tool": "dock_ligand", "task_id": task_id,
                               "status": dock_result.get("status"), "progress": dock_result.get("progress", 0),
                               "message": dock_result.get("progress_msg", "")}
                        if dock_result.get("status") in ("completed", "failed"):
                            break
                yield {"type": "tool_result", "tool": "dock_ligand",
                       "success": dock_result.get("status") == "completed", "summary": str(dock_result)[:300]}
                yield {"type": "answer", "content": f"Docking {'completed' if dock_result.get('status')=='completed' else 'failed'}."}
                return

            # ── Pocket detection: call fpocket after fetch_pdb ──
            if route_name == "pocket" and "fetch_pdb" in all_results:
                pdb_r = all_results["fetch_pdb"]
                pocket_args = {"input_pdb": pdb_r.get("output_pdb", ""), "job_name": f"pocket_{pdb_r.get('pdb_id','')}"}
                yield {"type": "tool_call", "tool": "fpocket_detect_pockets", "args": pocket_args}
                result = await self.executor.execute("fpocket_detect_pockets", pocket_args)
                if "task_id" in result:
                    task_id = result["task_id"]
                    yield {"type": "task_submitted", "tool": "fpocket_detect_pockets", "task_id": task_id}
                    elapsed = 0
                    while elapsed < 600:
                        await asyncio.sleep(10); elapsed += 10
                        result = await self.executor.execute("poll_task_status", {"task_id": task_id})
                        yield {"type": "task_progress", "tool": "fpocket_detect_pockets", "task_id": task_id,
                               "status": result.get("status"), "progress": result.get("progress", 0), "message": result.get("progress_msg", "")}
                        if result.get("status") in ("completed", "failed"): break
                yield {"type": "tool_result", "tool": "fpocket_detect_pockets",
                       "success": result.get("status") == "completed", "summary": str(result)[:300]}
                yield {"type": "answer", "content": f"Pocket detection {'completed' if result.get('status')=='completed' else 'failed'}."}
                return

            # ── Epitope prediction: call discotope3 after fetch_pdb ──
            if route_name == "epitope" and "fetch_pdb" in all_results:
                pdb_r = all_results["fetch_pdb"]
                dt3_args = {"input_pdb": pdb_r.get("output_pdb", ""), "job_name": f"epitope_{pdb_r.get('pdb_id','')}"}
                yield {"type": "tool_call", "tool": "discotope3_predict", "args": dt3_args}
                result = await self.executor.execute("discotope3_predict", dt3_args)
                if "task_id" in result:
                    task_id = result["task_id"]
                    yield {"type": "task_submitted", "tool": "discotope3_predict", "task_id": task_id}
                    elapsed = 0
                    while elapsed < 1200:
                        await asyncio.sleep(15); elapsed += 15
                        result = await self.executor.execute("poll_task_status", {"task_id": task_id})
                        yield {"type": "task_progress", "tool": "discotope3_predict", "task_id": task_id,
                               "status": result.get("status"), "progress": result.get("progress", 0), "message": result.get("progress_msg", "")}
                        if result.get("status") in ("completed", "failed"): break
                yield {"type": "tool_result", "tool": "discotope3_predict",
                       "success": result.get("status") == "completed", "summary": str(result)[:300]}
                yield {"type": "answer", "content": f"Epitope prediction {'completed' if result.get('status')=='completed' else 'failed'}."}
                return

            # ── MD simulation: call gromacs after fetch_pdb ──
            if route_name == "md_simulation" and "fetch_pdb" in all_results:
                pdb_r = all_results["fetch_pdb"]
                md_args = {"input_pdb": pdb_r.get("output_pdb", ""), "job_name": f"md_{pdb_r.get('pdb_id','')}", "sim_time_ns": 10.0}
                yield {"type": "tool_call", "tool": "gromacs_md_simulation", "args": md_args}
                result = await self.executor.execute("gromacs_md_simulation", md_args)
                if "task_id" in result:
                    task_id = result["task_id"]
                    yield {"type": "task_submitted", "tool": "gromacs_md_simulation", "task_id": task_id}
                    elapsed = 0
                    while elapsed < 7200:
                        if getattr(self, "_cancelled", False): break
                        await asyncio.sleep(30); elapsed += 30
                        result = await self.executor.execute("poll_task_status", {"task_id": task_id})
                        yield {"type": "task_progress", "tool": "gromacs_md_simulation", "task_id": task_id,
                               "status": result.get("status"), "progress": result.get("progress", 0), "message": result.get("progress_msg", "")}
                        if result.get("status") in ("completed", "failed"): break
                yield {"type": "tool_result", "tool": "gromacs_md_simulation",
                       "success": result.get("status") == "completed", "summary": str(result)[:300]}
                yield {"type": "answer", "content": f"MD simulation {'completed' if result.get('status')=='completed' else 'failed'}."}
                return

            # ── ADMET: call chemprop after fetch_molecule ──
            if route_name == "admet" and "fetch_molecule" in all_results:
                mol_r = all_results["fetch_molecule"]
                smiles = mol_r.get("smiles", "")
                if smiles:
                    admet_args = {"smiles": [smiles], "job_name": f"admet_{mol_r.get('cid','')}"}
                    yield {"type": "tool_call", "tool": "chemprop_predict", "args": admet_args}
                    result = await self.executor.execute("chemprop_predict", admet_args)
                    if "task_id" in result:
                        task_id = result["task_id"]
                        yield {"type": "task_submitted", "tool": "chemprop_predict", "task_id": task_id}
                        elapsed = 0
                        while elapsed < 300:
                            await asyncio.sleep(5); elapsed += 5
                            result = await self.executor.execute("poll_task_status", {"task_id": task_id})
                            yield {"type": "task_progress", "tool": "chemprop_predict", "task_id": task_id,
                                   "status": result.get("status"), "progress": result.get("progress", 0), "message": result.get("progress_msg", "")}
                            if result.get("status") in ("completed", "failed"): break
                    yield {"type": "tool_result", "tool": "chemprop_predict",
                           "success": result.get("status") == "completed", "summary": str(result)[:300]}
                    yield {"type": "answer", "content": f"ADMET prediction {'completed' if result.get('status')=='completed' else 'failed'}."}
                    return

            # Other routes (binder/drug_discovery pipeline, literature, download): tools already executed
            yield {"type": "answer", "content": f"Task submitted. Check progress in the left panel."}
            return

        skip_skills = self.is_simple_message(user_message)

        _SIMPLE_PROMPT = "You are the AI assistant of the OIH computational biology platform. You help users with protein structure prediction, drug design, molecular docking, and other computational biology tasks. Reply concisely. Do not reveal server IPs, GPU models, or internal paths."
        if skip_skills:
            dynamic_system = _SIMPLE_PROMPT
            yield {"type": "status", "content": "Quick reply mode..."}
        else:
            dynamic_system, detected_skills = build_dynamic_system_prompt(
                QWEN_SYSTEM_PROMPT, user_message
            )
            if detected_skills:
                yield {"type": "status", "content": f"Loaded knowledge: {', '.join(detected_skills)}"}

        max_iterations = 20
        iteration = 0
        tool_fail_count: Dict[str, int] = {}

        thinking_budget = 0  # thinking mode disabled for speed

        yield {"type": "thinking", "content": "LLM is reasoning..."}

        while iteration < max_iterations:
            iteration += 1

            MAX_HISTORY_CHARS = 48000
            history_window = self.conversation_history[-10:]
            total_chars = sum(len(str(m.get("content", ""))) for m in history_window)
            while total_chars > MAX_HISTORY_CHARS and len(history_window) > 1:
                dropped = history_window.pop(0)
                total_chars -= len(str(dropped.get("content", "")))

            llm_response = await self._llm_backend.chat(
                messages=history_window,
                tools=None if skip_skills else ALL_TOOLS,
                system_prompt=dynamic_system,
                max_tokens=1024 if skip_skills else 8192,
                temperature=0.1,
            )

            self.conversation_history.append(self._llm_backend.build_history_message(llm_response))

            # No tool calls → final answer
            if not llm_response.has_tool_calls:
                raw = (llm_response.content or "") or (llm_response.reasoning or "") or ""
                clean = re.sub(r"<[^>]+>", "", raw).strip()
                yield {"type": "answer", "content": clean}
                return

            # Execute tool calls
            for tc in llm_response.tool_calls:
                fn_name = tc.name
                fn_args = tc.arguments

                yield {"type": "tool_call", "tool": fn_name, "args": fn_args}

                if tool_fail_count.get(fn_name, 0) >= 2:
                    skip_msg = f"Tool {fn_name} has failed {tool_fail_count[fn_name]} times consecutively, skipping."
                    yield {"type": "tool_skip", "tool": fn_name, "reason": skip_msg}
                    self.conversation_history.append(
                        self._llm_backend.build_tool_result_message(
                            tc.id, fn_name, json.dumps({"error": skip_msg, "skipped": True})))
                    continue

                result = await self.executor.execute(fn_name, fn_args)

                if "task_id" in result and fn_name != "poll_task_status":
                    task_id = result["task_id"]
                    yield {"type": "task_submitted", "tool": fn_name, "task_id": task_id}
                    # Poll with progress events
                    elapsed = 0
                    poll_interval = 15
                    while elapsed < 7200:
                        if getattr(self, "_cancelled", False):
                            result = {"error": "cancelled by reset"}
                            break
                        await asyncio.sleep(poll_interval)
                        elapsed += poll_interval
                        result = await self.executor.execute("poll_task_status", {"task_id": task_id})
                        status = result.get("status")
                        progress = result.get("progress", 0)
                        msg = result.get("progress_msg", "")
                        yield {"type": "task_progress", "tool": fn_name, "task_id": task_id,
                               "status": status, "progress": progress, "message": msg}
                        if status in ("completed", "failed"):
                            break
                    else:
                        result = {"error": f"Task {task_id} timed out after 7200s"}

                # Emit result summary
                is_error = bool(result.get("error"))
                summary = str(result)[:300]
                yield {"type": "tool_result", "tool": fn_name, "success": not is_error, "summary": summary}

                if is_error:
                    tool_fail_count[fn_name] = tool_fail_count.get(fn_name, 0) + 1
                else:
                    tool_fail_count[fn_name] = 0

                result_str = json.dumps(result, ensure_ascii=False)
                if len(result_str) > 4000:
                    result_str = result_str[:4000] + f"\n... [truncated, {len(result_str)} chars total]"
                self.conversation_history.append(
                    self._llm_backend.build_tool_result_message(tc.id, fn_name, result_str))

            yield {"type": "thinking", "content": f"LLM analyzing results (round {iteration+1})..."}

        yield {"type": "answer", "content": "Agent exceeded max iterations. Please check task status manually."}

    def reset(self):
        self.conversation_history = []
        self._cancelled = True


# ─── FastAPI integration (add to main.py) ─────────────────────────────────────

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

agent_router = APIRouter()

class AgentChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    pdb_content: Optional[str] = None
    fasta_sequence: Optional[str] = None
    smiles: Optional[str] = None
    filename: Optional[str] = None
    llm_provider: Optional[str] = None   # "local" | "anthropic" | "openai" | "deepseek"
    llm_model: Optional[str] = None      # override model for this request

# ─── PTM Detection & Tool Input Generation (standalone functions) ────────────

import os
import tempfile

# CCD codes → human-readable PTM type
_PTM_CCD_MAP = {
    # Glycosylation
    "NAG": "Glycosylation(N-GlcNAc)", "FUC": "Glycosylation(Fucose)", "MAN": "Glycosylation(Mannose)",
    "BMA": "Glycosylation(β-Mannose)", "GAL": "Glycosylation(Galactose)", "SIA": "Glycosylation(Sialic acid)",
    # Phosphorylation
    "SEP": "Phosphorylation(pSer)", "TPO": "Phosphorylation(pThr)", "PTR": "Phosphorylation(pTyr)",
    # Others
    "MLY": "Methylation(Lys)", "M3L": "Trimethylation(Lys)", "ALY": "Acetylation(Lys)",
    "CSO": "Oxidation(Cys)", "OCS": "Oxidation(Cys)",
}

def detect_ptm(pdb_content: str) -> Dict[str, Any]:
    """
    Parse PDB content to detect PTMs, disulfide bonds, and chain/sequence info.
    Returns structured dict — does NOT modify any existing agent state.
    """
    ptms = []          # [{"type": "Glycosylation", "ccd": "NAG", "chain": "A", "resseq": 297}, ...]
    disulfides = []    # [{"chain1": "A", "res1": 42, "chain2": "A", "res2": 98}, ...]
    chains = {}        # {"A": "MKTIIALSYIFCLVFA...", ...}
    het_residues = []  # non-PTM HETATM residues

    lines = pdb_content.splitlines()

    for line in lines:
        # ── SSBOND records ──
        if line.startswith("SSBOND"):
            try:
                chain1 = line[15].strip()
                res1 = int(line[17:21].strip())
                chain2 = line[29].strip()
                res2 = int(line[31:35].strip())
                disulfides.append({"chain1": chain1, "res1": res1, "chain2": chain2, "res2": res2})
            except (IndexError, ValueError):
                pass

        # ── HETATM → PTM detection ──
        if line.startswith("HETATM"):
            resname = line[17:20].strip()
            chain = line[21].strip()
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue

            if resname == "HOH":
                continue

            if resname in _PTM_CCD_MAP:
                entry = {"type": _PTM_CCD_MAP[resname], "ccd": resname,
                         "chain": chain, "resseq": resseq}
                if entry not in ptms:
                    ptms.append(entry)
            else:
                entry = {"resname": resname, "chain": chain, "resseq": resseq}
                if entry not in het_residues:
                    het_residues.append(entry)

        # ── ATOM → extract sequence per chain ──
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            chain = line[21].strip()
            resname = line[17:20].strip()
            # 3-letter → 1-letter
            aa_map = {
                "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
                "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
                "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
                "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
            }
            aa = aa_map.get(resname, "X")
            chains.setdefault(chain, "")
            chains[chain] += aa

    return {
        "ptms": ptms,
        "disulfides": disulfides,
        "chains": chains,
        "het_residues": het_residues,
    }


def generate_tool_inputs(
    pdb_content: Optional[str],
    fasta_sequence: Optional[str],
    smiles: Optional[str],
    filename: Optional[str],
    session_id: str,
) -> str:
    """
    Given uploaded data, detect PTMs and generate pre-built tool input files.
    Returns a context string to prepend to the user message.
    Does NOT modify any existing agent state.
    """
    upload_dir = f"/tmp/oih_uploads/{session_id}"
    os.makedirs(upload_dir, exist_ok=True)

    context_parts = []
    file_label = filename or "uploaded_file"

    # ── Save PDB to disk ──
    if pdb_content:
        pdb_path = os.path.join(upload_dir, filename or "upload.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_content)
        context_parts.append(f"[User uploaded file: {file_label}]")
        context_parts.append(f"[PDB file saved: {pdb_path}]")

        # ── PTM detection ──
        ptm_result = detect_ptm(pdb_content)
        ptms = ptm_result["ptms"]
        disulfides = ptm_result["disulfides"]
        chains = ptm_result["chains"]

        # PTM summary line
        ptm_summary_parts = []
        if ptms:
            # Group by type
            from collections import Counter
            type_counts = Counter(p["type"] for p in ptms)
            for ptype, cnt in type_counts.items():
                positions = [f"{p['chain']}{p['resseq']}" for p in ptms if p["type"] == ptype]
                ptm_summary_parts.append(f"{ptype}×{cnt}({','.join(positions)})")
        if disulfides:
            ptm_summary_parts.append(f"Disulfide bonds×{len(disulfides)}")

        if ptm_summary_parts:
            context_parts.append(f"[PTM detection: {', '.join(ptm_summary_parts)}]")
        else:
            context_parts.append("[PTM detection: no modifications]")

        # ── Generate AF3 input JSON ──
        af3_chains = []
        for chain_id, seq in chains.items():
            chain_entry = {"type": "protein", "sequence": seq, "count": 1}
            # Add PTM modifications for this chain
            chain_ptms = [p for p in ptms if p["chain"] == chain_id and p["ccd"] in _PTM_CCD_MAP]
            if chain_ptms:
                chain_entry["modifications"] = [
                    {"ptmType": p["ccd"], "ptmPosition": p["resseq"]}
                    for p in chain_ptms
                ]
            af3_chains.append(chain_entry)

        # Add glycan ligands as separate CCD ligand entries
        glycan_ccds = [p["ccd"] for p in ptms if "Glycosylation" in p["type"]]
        for ccd in set(glycan_ccds):
            af3_chains.append({"type": "ligand", "ccdCodes": [ccd], "count": glycan_ccds.count(ccd)})

        af3_input = {
            "name": file_label.replace(".pdb", ""),
            "modelSeeds": [1, 2, 3],
            "sequences": af3_chains,
        }
        af3_path = os.path.join(upload_dir, "af3_input.json")
        with open(af3_path, "w") as f:
            json.dump(af3_input, f, indent=2)

        # ── Generate GROMACS PTM notes ──
        gromacs_notes = {
            "pdb_path": pdb_path,
            "force_field": "charmm36" if any("Phosphorylation" in p["type"] for p in ptms) else "amber99sb-ildn",
            "disulfide_pairs": [[d["res1"], d["res2"]] for d in disulfides],
            "unsupported_ptms": [p["type"] for p in ptms if "Glycosylation" in p["type"]],
            "note": "Glycosylated residues require GLYCAM force field, not supported by GROMACS. Use AF3 prediction only." if any("Glycosylation" in p["type"] for p in ptms) else "",
        }
        gromacs_path = os.path.join(upload_dir, "gromacs_ptm_notes.json")
        with open(gromacs_path, "w") as f:
            json.dump(gromacs_notes, f, indent=2, ensure_ascii=False)

        # ── Generate ADC input if Cys sites detected ──
        cys_positions = []
        for chain_id, seq in chains.items():
            for i, aa in enumerate(seq):
                if aa == "C":
                    cys_positions.append({"chain": chain_id, "position": i + 1})
        if cys_positions:
            adc_input = {
                "pdb_path": pdb_path,
                "candidate_conjugation_sites": [f"C{c['position']}" for c in cys_positions[:10]],
                "note": "Must confirm SASA>40A^2 with FreeSASA before selecting conjugation sites",
            }
            adc_path = os.path.join(upload_dir, "adc_input.json")
            with open(adc_path, "w") as f:
                json.dump(adc_input, f, indent=2, ensure_ascii=False)

        # ── Context: generated files ──
        context_parts.append("[Generated tool inputs:]")
        context_parts.append(f"- AF3: {af3_path}")
        context_parts.append(f"- GROMACS: {gromacs_path}")
        if cys_positions:
            context_parts.append(f"- ADC: {os.path.join(upload_dir, 'adc_input.json')}")
        context_parts.append("[LLM can use the above paths directly, no need to rebuild inputs]")

    # ── FASTA sequence ──
    if fasta_sequence:
        fasta_path = os.path.join(upload_dir, "upload.fasta")
        with open(fasta_path, "w") as f:
            if not fasta_sequence.startswith(">"):
                f.write(f">uploaded_sequence\n{fasta_sequence}\n")
            else:
                f.write(fasta_sequence)
        context_parts.append(f"[User uploaded FASTA sequence: {fasta_path}]")
        context_parts.append("[No structure file available. Recommend using AlphaFold3 to predict structure first.]")

    # ── SMILES ──
    if smiles:
        context_parts.append(f"[User uploaded SMILES: {smiles}]")
        context_parts.append("[Recommend predicting ADMET properties with Chemprop first, then use for docking.]")

    if not context_parts:
        return ""

    return "\n".join(context_parts)


# Simple session store (use Redis in production)
_sessions: Dict[str, QwenBioAgent] = {}


def _get_or_create_agent(session_id: str, req: AgentChatRequest) -> QwenBioAgent:
    """Get existing agent or create new one, optionally with a different LLM backend."""
    if session_id not in _sessions:
        _sessions[session_id] = QwenBioAgent()

    agent = _sessions[session_id]

    # Switch LLM backend if requested
    if req.llm_provider and req.llm_provider != agent._llm_backend.provider:
        backend = create_backend(
            provider=req.llm_provider,
            api_key=_cfg.LLM_API_KEY,
            model=req.llm_model or _cfg.LLM_MODEL,
            base_url=_cfg.LLM_BASE_URL,
            vllm_url=_cfg.QWEN_BASE_URL,
            vllm_model=_cfg.QWEN_MODEL,
        )
        agent._llm_backend = backend
        print(f"[LLM] Session {session_id}: switched to {req.llm_provider} ({getattr(backend, 'model', '')})")

    return agent


@agent_router.post("/chat")
async def agent_chat(req: AgentChatRequest):
    """
    Send a natural language message to the bio-computing agent.
    The agent will plan and execute multi-step workflows using the tool chain.

    Example messages:
    - "Please predict the structure of this sequence: MKTIIALSYIFCLVFA..."
    - "Design 50 protein binders targeting the ACE2 receptor (target.pdb)"  
    - "Dock aspirin (CC(=O)Oc1ccccc1C(=O)O) into the PDB 1Z0R binding site"
    - "Run a complete drug discovery pipeline for EGFR with erlotinib"
    """
    session_id = req.session_id or "default"
    agent = _get_or_create_agent(session_id, req)

    # ── Attachment processing: prepend context if uploads present ──
    user_message = req.message
    if req.pdb_content or req.fasta_sequence or req.smiles:
        context = generate_tool_inputs(
            pdb_content=req.pdb_content,
            fasta_sequence=req.fasta_sequence,
            smiles=req.smiles,
            filename=req.filename,
            session_id=session_id,
        )
        if context:
            user_message = context + "\n\n" + req.message
            print(f"[Upload] Enhanced message with attachment context ({len(context)} chars)")

    response = await agent.chat(user_message)
    return {
        "response": response,
        "session_id": session_id,
        "turns": len(agent.conversation_history),
    }

@agent_router.post("/chat/stream")
async def agent_chat_stream(req: AgentChatRequest):
    """SSE streaming version — emits real-time events for each reasoning step."""
    session_id = req.session_id or "default"
    agent = _get_or_create_agent(session_id, req)

    user_message = req.message
    if req.pdb_content or req.fasta_sequence or req.smiles:
        context = generate_tool_inputs(
            pdb_content=req.pdb_content, fasta_sequence=req.fasta_sequence,
            smiles=req.smiles, filename=req.filename, session_id=session_id,
        )
        if context:
            user_message = context + "\n\n" + req.message

    async def event_generator():
        try:
            async for event in agent.chat_stream(user_message):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'turns': len(agent.conversation_history)})}\n\n"
        except Exception as e:
            import traceback
            err_detail = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"
            print(f"[SSE ERROR] {err_detail}")
            yield f"data: {json.dumps({'type': 'error', 'content': err_detail})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@agent_router.delete("/chat/{session_id}")
async def reset_session(session_id: str):
    if session_id in _sessions:
        _sessions[session_id].reset()
    return {"reset": True}


# ─── CLI for testing ──────────────────────────────────────────────────────────

async def main():
    agent = QwenBioAgent()
    print("OIH Bio-Computing Agent (Qwen3-14B)")
    print("Commands: 'quit', 'reset'")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("Session reset.")
            continue

        response = await agent.chat(user_input)
        print(f"\nAgent: {response}")


if __name__ == "__main__":
    asyncio.run(main())
