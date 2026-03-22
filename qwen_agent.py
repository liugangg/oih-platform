"""
Qwen3-14B Agent Integration
将 Qwen 连接到 OIH 工具链的完整示例

部署方式：
  GPU0 → vLLM serving Qwen3-14B
  GPU1 → OIH API Gateway (this FastAPI service)
  
Usage:
  python qwen_agent.py
  或通过 HTTP POST /api/v1/agent/chat 调用
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
    def __init__(self):
        self.executor = OIHToolExecutor()
        self.conversation_history: List[Dict] = []

    SIMPLE_PATTERNS = [
        "你好", "hello", "hi", "嗨", "早", "晚安", "谢谢", "thanks",
        "ok", "好的", "收到", "再见", "bye", "nice", "👍", "哈哈",
    ]
    BIO_KEYWORDS = [
        "protein", "pdb", "smiles", "fasta", "药", "蛋白", "抗体", "设计",
        "预测", "分析", "结构", "对接", "序列", "靶点", "分子", "nanobody",
        "alphafold", "gromacs", "docking", "md", "adc", "ligand",
    ]

    @staticmethod
    def is_simple_message(msg: str) -> bool:
        msg_lower = msg.lower().strip()
        if len(msg_lower) < 20:
            if any(p in msg_lower for p in QwenBioAgent.SIMPLE_PATTERNS):
                return True
            if not any(k in msg_lower for k in QwenBioAgent.BIO_KEYWORDS):
                return True
        return False

    async def chat(self, user_message: str) -> str:
        """Single turn: user message → agent response with tool use"""
        self._cancelled = False
        self.conversation_history.append({"role": "user", "content": user_message})

        # ── Simple message fast path: skip skills, thinking_budget=0 ──
        skip_skills = self.is_simple_message(user_message)

        if skip_skills:
            dynamic_system = QWEN_SYSTEM_PROMPT
            print(f"[Simple] Skipping skills injection")
        else:
            dynamic_system, detected_skills = build_dynamic_system_prompt(
                QWEN_SYSTEM_PROMPT, user_message
            )
            if detected_skills:
                print(f"[Skills] 注入: {detected_skills}")

        max_iterations = 20   # prevent infinite loops
        iteration = 0
        tool_fail_count: Dict[str, int] = {}   # track consecutive failures per tool

        # 动态thinking budget
        def _get_thinking_budget(msg: str) -> int:
            if skip_skills:
                return 0
            m = msg.lower()
            if any(k in m for k in ["pipeline", "流程", "然后", "接着", "再用", "最后用", "完整"]):
                return 4096
            if any(k in m for k in ["rfdiffusion", "alphafold", "gromacs", "autodock", "bindcraft", "proteinmpnn", "diffdock"]):
                return 2048
            if any(k in m for k in ["dock", "predict", "design", "simulate", "fold", "embed", "对接", "预测", "设计", "模拟"]):
                return 1024
            return 2048
        thinking_budget = _get_thinking_budget(user_message)
        print(f"[Thinking] budget={thinking_budget} tokens")

        while iteration < max_iterations:
            iteration += 1

            # Call Qwen
            async with httpx.AsyncClient(timeout=60 + thinking_budget * 0.1) as client:  # 动态timeout
                payload = {
                    "model": QWEN_MODEL,
                    "messages": [
                        {"role": "system", "content": dynamic_system}
                    ] + self.conversation_history[-10:],
                    "tools": ALL_TOOLS,
                    "tool_choice": "auto",
                    "max_tokens": 8192,
                    "temperature": 0.1,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_budget": thinking_budget},
                }
                resp = await client.post(f"{QWEN_BASE_URL}/chat/completions", json=payload)
                resp.raise_for_status()

            response_data = resp.json()
            message = response_data["choices"][0]["message"]
            finish_reason = response_data["choices"][0]["finish_reason"]

            # Add assistant message to history
            self.conversation_history.append(message)

            # No tool calls → final answer
            if finish_reason == "stop" or not message.get("tool_calls"):
                raw = message.get("content", "") or ""
                import re as _re
                clean = _re.sub(r"<[^>]+>", "", raw).strip()
                return clean

            # Execute tool calls
            tool_calls = message.get("tool_calls", [])
            print(f"\n[Agent] Calling {len(tool_calls)} tool(s)...")

            for tool_call in tool_calls:
                fn_name = tool_call["function"]["name"]
                fn_args = json.loads(tool_call["function"]["arguments"])

                print(f"  → {fn_name}({list(fn_args.keys())})")

                # ── Skip tools that have failed ≥2 times consecutively ────
                if tool_fail_count.get(fn_name, 0) >= 2:
                    skip_msg = (
                        f"工具 {fn_name} 已连续失败{tool_fail_count[fn_name]}次，"
                        f"请跳过该步骤继续后续流程，不要再调用此工具。"
                    )
                    print(f"     SKIPPED (consecutive failures={tool_fail_count[fn_name]})")
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": fn_name,
                        "content": json.dumps({"error": skip_msg, "skipped": True}),
                    })
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
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": fn_name,
                    "content": result_str,
                })

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
                    "instruction": "分析上方error内容找出根本原因，修改参数后重试，不要用相同参数重试"
                }

        return {"error": f"Task {task_id} timed out after {max_wait}s"}

    def reset(self):
        self.conversation_history = []
        self._cancelled = True


# ─── FastAPI integration (add to main.py) ─────────────────────────────────────

from fastapi import APIRouter
from pydantic import BaseModel

agent_router = APIRouter()

class AgentChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    pdb_content: Optional[str] = None
    fasta_sequence: Optional[str] = None
    smiles: Optional[str] = None
    filename: Optional[str] = None

# ─── PTM Detection & Tool Input Generation (standalone functions) ────────────

import os
import tempfile

# CCD codes → human-readable PTM type
_PTM_CCD_MAP = {
    # Glycosylation
    "NAG": "糖基化(N-GlcNAc)", "FUC": "糖基化(Fucose)", "MAN": "糖基化(Mannose)",
    "BMA": "糖基化(β-Mannose)", "GAL": "糖基化(Galactose)", "SIA": "糖基化(Sialic acid)",
    # Phosphorylation
    "SEP": "磷酸化(pSer)", "TPO": "磷酸化(pThr)", "PTR": "磷酸化(pTyr)",
    # Others
    "MLY": "甲基化(Lys)", "M3L": "三甲基化(Lys)", "ALY": "乙酰化(Lys)",
    "CSO": "氧化(Cys)", "OCS": "氧化(Cys)",
}

def detect_ptm(pdb_content: str) -> Dict[str, Any]:
    """
    Parse PDB content to detect PTMs, disulfide bonds, and chain/sequence info.
    Returns structured dict — does NOT modify any existing agent state.
    """
    ptms = []          # [{"type": "糖基化", "ccd": "NAG", "chain": "A", "resseq": 297}, ...]
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
        context_parts.append(f"[用户上传文件: {file_label}]")
        context_parts.append(f"[PDB文件已保存: {pdb_path}]")

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
            ptm_summary_parts.append(f"二硫键×{len(disulfides)}")

        if ptm_summary_parts:
            context_parts.append(f"[PTM检测: {', '.join(ptm_summary_parts)}]")
        else:
            context_parts.append("[PTM检测: 无修饰]")

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
        glycan_ccds = [p["ccd"] for p in ptms if "糖基化" in p["type"]]
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
            "force_field": "charmm36" if any("磷酸化" in p["type"] for p in ptms) else "amber99sb-ildn",
            "disulfide_pairs": [[d["res1"], d["res2"]] for d in disulfides],
            "unsupported_ptms": [p["type"] for p in ptms if "糖基化" in p["type"]],
            "note": "糖基化残基需要GLYCAM力场，GROMACS不支持，建议仅用AF3预测" if any("糖基化" in p["type"] for p in ptms) else "",
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
                "note": "需要先用freesasa确认SASA>40Å²再选择偶联位点",
            }
            adc_path = os.path.join(upload_dir, "adc_input.json")
            with open(adc_path, "w") as f:
                json.dump(adc_input, f, indent=2, ensure_ascii=False)

        # ── Context: generated files ──
        context_parts.append("[已生成工具输入:]")
        context_parts.append(f"- AF3: {af3_path}")
        context_parts.append(f"- GROMACS: {gromacs_path}")
        if cys_positions:
            context_parts.append(f"- ADC: {os.path.join(upload_dir, 'adc_input.json')}")
        context_parts.append("[Qwen直接使用以上路径，无需重新构建输入]")

    # ── FASTA sequence ──
    if fasta_sequence:
        fasta_path = os.path.join(upload_dir, "upload.fasta")
        with open(fasta_path, "w") as f:
            if not fasta_sequence.startswith(">"):
                f.write(f">uploaded_sequence\n{fasta_sequence}\n")
            else:
                f.write(fasta_sequence)
        context_parts.append(f"[用户上传FASTA序列: {fasta_path}]")
        context_parts.append("[无结构文件，建议先用alphafold3预测结构]")

    # ── SMILES ──
    if smiles:
        context_parts.append(f"[用户上传SMILES: {smiles}]")
        context_parts.append("[建议先用chemprop预测ADMET属性，再用于对接]")

    if not context_parts:
        return ""

    return "\n".join(context_parts)


# Simple session store (use Redis in production)
_sessions: Dict[str, QwenBioAgent] = {}

@agent_router.post("/chat")
async def agent_chat(req: AgentChatRequest):
    """
    Send a natural language message to Qwen3-14B bio-computing agent.
    The agent will plan and execute multi-step workflows using the tool chain.
    
    Example messages:
    - "Please predict the structure of this sequence: MKTIIALSYIFCLVFA..."
    - "Design 50 protein binders targeting the ACE2 receptor (target.pdb)"  
    - "Dock aspirin (CC(=O)Oc1ccccc1C(=O)O) into the PDB 1Z0R binding site"
    - "Run a complete drug discovery pipeline for EGFR with erlotinib"
    """
    session_id = req.session_id or "default"
    if session_id not in _sessions:
        _sessions[session_id] = QwenBioAgent()

    agent = _sessions[session_id]

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
