# OIH — Open Intelligence Hub

An autonomous LLM-agent platform for computational binder design and conjugation-aware prioritization of antibody–drug conjugates.

![Architecture](docs/images/architecture.png)

## Overview

OIH orchestrates **32 computational biology tools** across **15 Docker containers** using a large language model agent. Given a target protein name, the platform autonomously executes the complete binder design and in silico ADC modelling pipeline — from hotspot identification through RFdiffusion backbone generation, ProteinMPNN sequence design, AlphaFold 3 validation, to final ADC conjugation.

## Architecture

![Tool Ecosystem](docs/images/tool_grid.png)

- **LLM Agent**: LLM-agnostic backend — swap between local and cloud LLMs without changing any pipeline logic
- **Tool Orchestration**: Multi-round function calling with dynamic skills injection (26 workflow documents)
- **Tier Routing**: Structure-guided (Tier 1) and prediction-guided (Tier 2) hotspot selection
- **Validation**: AlphaFold 3 + ipSAE interface quality filtering
- **ADC Assembly**: Automated conjugation site selection (FreeSASA) and linker chemistry (RDKit)

## Pipeline

![Pipeline](docs/images/pipeline.png)

## Tool Inventory

| Category | Tools |
|----------|-------|
| **Structure Prediction** | AlphaFold 3 |
| **Protein Design** | RFdiffusion, ProteinMPNN, BindCraft |
| **Binding Site Analysis** | fpocket, P2Rank, PeSTo, DiscoTope3, IgFold, ipSAE |
| **Molecular Docking** | GNINA, AutoDock-GPU, Vina-GPU, DiffDock |
| **MD Simulation** | GROMACS |
| **ML Analysis** | ESM2 (embedding + mutant scan), Chemprop (ADMET) |
| **ADC Design** | FreeSASA, Linker Selection, RDKit Conjugation |
| **Literature** | PubMed + bioRxiv RAG, Web Search |
| **Data Analysis** | Python execution (matplotlib/pandas), Report generation |

## Quick Start

### 1. Start containers

```bash
# Start vLLM (GPU 0)
docker compose -f docker-compose.vllm.yml up -d

# Start bio containers (GPU 1)
docker compose up -d

# Start FastAPI backend
python -m uvicorn main:app --host 0.0.0.0 --port 8080
```

### 2. Open the dashboard

Navigate to `http://<server-ip>:8080/` in your browser.

### 3. Try it

Type natural language commands like:
- "Design a binder targeting HER2"
- "Dock CD36 with palmitic acid"
- "Predict the structure of TP53 with AlphaFold3"

## LLM Backend Configuration

The platform is **LLM-agnostic**: switching the LLM backend does not affect any pipeline logic — skills, tool definitions, and RAG all communicate through standard text and function-calling protocols.

### Option A: Environment variables

```bash
# Default: local Qwen (data stays on your server)
LLM_PROVIDER=local python -m uvicorn main:app --port 8080

# Switch to Anthropic Claude
LLM_PROVIDER=anthropic LLM_API_KEY=sk-ant-xxx LLM_MODEL=claude-sonnet-4-20250514 \
  python -m uvicorn main:app --port 8080

# Switch to OpenAI GPT-4
LLM_PROVIDER=openai LLM_API_KEY=sk-xxx LLM_MODEL=gpt-4o \
  python -m uvicorn main:app --port 8080
```

### Option B: `.env` file

```env
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-api03-xxxxx
LLM_MODEL=claude-sonnet-4-20250514
```

### Option C: Docker Compose

```yaml
environment:
  - LLM_PROVIDER=anthropic
  - LLM_API_KEY=${ANTHROPIC_API_KEY}
```

### In code

```python
from core.llm_backend import get_backend

llm = get_backend(provider="anthropic", api_key="sk-ant-xxx", model="claude-sonnet-4-20250514")
response = await llm.chat(messages, tools=tools)
```

## API Routes

| Endpoint | Module | Function |
|----------|--------|----------|
| `/api/v1/structure` | structure_prediction.py | AlphaFold 3 |
| `/api/v1/design` | protein_design.py | RFdiffusion, ProteinMPNN, BindCraft |
| `/api/v1/pocket` | pocket_analysis.py | fpocket, P2Rank |
| `/api/v1/docking` | molecular_docking.py | GNINA, AutoDock-GPU, Vina, DiffDock |
| `/api/v1/md` | md_simulation.py | GROMACS |
| `/api/v1/ml` | ml_tools.py | ESM2, Chemprop |
| `/api/v1/adc` | adc.py | Linker selection, RDKit conjugation |
| `/api/v1/pipeline` | pipeline.py | End-to-end pipelines |
| `/api/v1/tasks` | tasks.py | Async task management |
| `/api/v1/agent/chat` | qwen_agent.py | LLM agent chat (SSE streaming) |

## Hardware Requirements

- **GPU 0**: 24GB+ VRAM — LLM inference (vLLM)
- **GPU 1**: 24GB+ VRAM (45GB recommended) — All bio computation tools
- **CPU**: 16+ cores recommended
- **RAM**: 64GB+ recommended

## Key Results

Across five oncology-relevant targets:

| Target | ipTM | ipSAE | In silico conjugation |
|--------|------|-------|----------------------|
| Nectin-4 | 0.87 | 0.68 | Modelled |
| HER2 | 0.85 | 0.53 | Modelled |
| EGFR | 0.52 | 0.19 | Modelled |
| CD36 | 0.58 | 0.056 | — |
| TROP2 | 0.22 | 0.000 | — |

## Citation

If you use OIH in your research, please cite:
> Liu, G. et al. An autonomous LLM-agent platform for computational binder design and conjugation-aware prioritization of antibody–drug conjugates. *Nature Communications* (under review, 2026).

## License

MIT License
