# Open Intelligence Hub (OIH)

An LLM-agent platform for computational biology and drug discovery. Researchers submit natural language requests, and a Qwen3-14B agent autonomously plans and executes full computational workflows using 30 integrated tools across 15 GPU-accelerated containers.

## Architecture

```
User (natural language) → Qwen3-14B Agent → Tool Orchestration → Results
                              ↑                    ↓
                        Skills / RAG          15 Docker containers
                                              (GPU-accelerated)
```

**Key design principle:** No hardcoded pipelines. The agent reads tool descriptions (including upstream/downstream relationships), domain knowledge from skill files, and literature from RAG — then plans and executes dynamically.

## Tool Stack (30 tools, 15 containers)

### Structure Prediction
| Tool | Container | Description |
|------|-----------|-------------|
| AlphaFold3 | oih-alphafold3 | Protein/complex structure prediction |
| ESM2 | oih-esm | Protein language model (embeddings, PPL scoring, mutant scanning) |
| IgFold | oih-igfold | Antibody/nanobody structure prediction |

### Protein Design
| Tool | Container | Description |
|------|-----------|-------------|
| RFdiffusion | oih-rfdiffusion | De novo protein backbone design (diffusion model) |
| ProteinMPNN | oih-proteinmpnn | Sequence design for protein backbones |
| BindCraft | oih-bindcraft | End-to-end binder design |

### Molecular Docking
| Tool | Container | Description |
|------|-----------|-------------|
| DiffDock | oih-diffdock | Deep learning blind docking |
| GNINA | oih-gnina | CNN-scored molecular docking |
| AutoDock-GPU | oih-autodock-gpu | GPU-accelerated AutoDock |
| Vina-GPU | oih-vina-gpu | GPU-accelerated AutoDock Vina |

### Analysis
| Tool | Container | Description |
|------|-----------|-------------|
| fpocket | oih-fpocket | Binding pocket detection |
| P2Rank | oih-p2rank | ML-based binding site prediction |
| FreeSASA | (host) | Solvent-accessible surface area |
| GROMACS | oih-gromacs | GPU-accelerated molecular dynamics |

### Immunology
| Tool | Container | Description |
|------|-----------|-------------|
| DiscoTope3 | oih-discotope3 | B-cell epitope prediction |
| extract_interface | (host) | Antibody-antigen interface extraction |

### Chemistry
| Tool | Container | Description |
|------|-----------|-------------|
| ChemProp | oih-chemprop | Molecular property prediction (ADMET) |
| RDKit conjugate | (host) | ADC conjugation chemistry |

## Pipelines

### Pocket-Guided Binder Design (16 steps)
Fully automated: PDB ID → pocket detection → 6D scoring → binder design → AF3 validation → ADC assembly.

Features a **Target Tier System**:
- **Tier 1**: Known antibody-antigen complex exists → extract interface residues (most reliable)
- **Tier 2**: Homologous structure from literature → RAG-guided
- **Tier 3**: Novel target → DiscoTope3 + IEDB + RAG computational prediction

### Drug Discovery Pipeline
Target sequence + ligand SMILES → complete drug discovery workflow.

### Binder Design Pipeline
PDB + known hotspot residues → binder + ADC design (7 steps).

## RAG Sources

| Source | Type | Description |
|--------|------|-------------|
| PubMed | Real-time | Biomedical literature search |
| bioRxiv | Real-time | Preprint search |
| IEDB | Database | Immune epitope database |
| SAbDab | Database | Structural antibody database |
| LocalDB | Vector DB | 86+ curated documents |

## Setup

### Prerequisites
- 2× NVIDIA RTX 4090 (or equivalent)
  - GPU 0: Qwen3-14B via vLLM
  - GPU 1: All compute tools
- Docker with NVIDIA Container Toolkit
- Python 3.10+

### Configuration
```bash
# Set server host (default: localhost)
export OIH_SERVER_HOST=your.server.ip

# Start containers
docker compose up -d

# Start API
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `OIH_SERVER_HOST` | `localhost` | Server hostname/IP |
| `NVIDIA_VISIBLE_DEVICES` | `1` | GPU for compute containers |

## Project Structure

```
oih-api/
├── main.py                 # FastAPI application entry point
├── qwen_agent.py           # Qwen3-14B tool-calling agent
├── skills_loader.py        # Keyword → skill document injection
├── routers/                # API route handlers
│   ├── pipeline.py         # Multi-step pipeline orchestration
│   ├── structure_prediction.py
│   ├── protein_design.py
│   ├── molecular_docking.py
│   ├── immunology.py
│   ├── pocket_analysis.py
│   ├── adc.py
│   └── ml_tools.py
├── schemas/models.py       # Pydantic request/response models
├── core/
│   ├── config.py           # Settings (env-based)
│   ├── task_manager.py     # Async task queue (CPU/GPU routing)
│   └── docker_client.py    # Container execution layer
├── tool_definitions/
│   └── qwen_tools.py       # OpenAI-format tool definitions for Qwen
├── skills/                 # Domain knowledge documents
│   ├── DISCOTOPE3_WORKFLOW.md
│   ├── IGFOLD_WORKFLOW.md
│   ├── TOOL_SCOPE_RULES.md
│   └── ...
├── docs/                   # Session logs
├── data/                   # Runtime data (gitignored)
└── docker-compose.yml      # Container definitions
```

## Key Design Decisions

1. **No hardcoded pipelines** — Qwen plans dynamically based on tool descriptions and domain knowledge
2. **Target Tier System** — Automatically classifies targets for optimal hotspot identification strategy
3. **Binder type awareness** — IgFold is only used for antibody/nanobody sequences (not de novo binders)
4. **6D pocket scoring** — Multi-criteria composite: P2Rank + SASA + conservation + RAG + electrostatics + epitope
5. **Known epitope override** — Literature-validated epitopes bypass computational scoring

## License

Proprietary — Open Intelligence Hub
