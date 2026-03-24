"""
Qwen3-14B Tool Definitions
OpenAI Function Calling 格式，直接注入 Qwen system prompt
让 LLM 知道有哪些工具、如何调用
"""

import os as _os
API_BASE = f"http://{_os.environ.get('OIH_SERVER_HOST', 'localhost')}:8000/api/v1"

ALL_TOOLS = [

    # ─── Fetch PDB ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "fetch_pdb",
            "description": (
                "Download a protein structure from RCSB PDB database by PDB ID. "
                "Use when user mentions a PDB code (e.g. 5LGD for CD36, 1IVO, 6LU7) "
                "or when a known experimental structure is needed before running "
                "pocket detection, docking, or binder design. "
                "Returns the local file path of the downloaded PDB in the output_pdb field "
                "(e.g. /data/oih/outputs/fetch_pdb/5XWR.pdb). "
                "You MUST use this exact returned path as input_pdb for subsequent tools like fpocket or docking. "
                "Do NOT construct the path manually."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pdb_id": {"type": "string", "description": "4-letter PDB ID, e.g. 5LGD"},
                    "chains": {"type": "string", "description": "Optional: chains to keep, e.g. A or A,B. Removes water/HETATM."},
                },
                "required": ["pdb_id"],
            },
        }
    },

    # ─── Fetch Molecule ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "fetch_molecule",
            "description": (
                "Look up a drug or small molecule by name, PubChem CID, or SMILES. "
                "Returns canonical SMILES, molecular properties (MW, formula), and 3D SDF file. "
                "Use when user mentions a drug name (aspirin, erlotinib, imatinib) "
                "and you need its SMILES for docking or its SDF file for DiffDock/GNINA. "
                "The returned SMILES can be passed directly to dock_ligand or diffdock_blind_dock."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Drug name (e.g. aspirin), PubChem CID (e.g. 2244), or SMILES"},
                },
                "required": ["query"],
            },
        }
    },
    # ─── AlphaFold3 ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "alphafold3_predict",
            "description": (
                "Predict 3D structure of proteins, RNA, DNA, or multi-chain complexes "
                "using AlphaFold3. Use when given a protein sequence that needs structural "
                "prediction. Returns task_id for async polling. "
                "Supports: single protein, protein-protein complex, protein-ligand, "
                "protein-nucleic acid complexes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Unique job identifier (no spaces)"},
                    "chains": {
                        "type": "array",
                        "description": "List of chains/molecules to include",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["protein", "rna", "dna", "ligand", "ion"]},
                                "sequence": {"type": "string", "description": "Amino acid or nucleotide sequence"},
                                "smiles": {"type": "string", "description": "SMILES string for ligand"},
                                "count": {"type": "integer", "default": 1},
                                "modifications": {
                                    "type": "array",
                                    "description": "PTMs on protein/RNA/DNA. Common: SEP=phosphoSer, TPO=phosphoThr, PTR=phosphoTyr, MLY=methylLys, HY3=hydroxyPro",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "ptmType": {"type": "string", "description": "CCD 3-letter code"},
                                            "ptmPosition": {"type": "integer", "description": "1-based residue position"},
                                        },
                                        "required": ["ptmType", "ptmPosition"]
                                    }
                                },
                            },
                            "required": ["type"],
                        }
                    },
                    "num_seeds": {"type": "integer", "default": 5, "description": "Number of random seeds (more = more diverse structures)"},
                    "run_relaxation": {"type": "boolean", "default": True},
                },
                "required": ["job_name", "chains"],
            },
            "_endpoint": f"{API_BASE}/structure/alphafold3",
            "_method": "POST",
        }
    },

    # ─── RFdiffusion ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "rfdiffusion_design",
            "description": (
                "Design novel protein backbones using RFdiffusion diffusion model. "
                "Use for: (1) designing protein binders to a target, "
                "(2) scaffolding functional motifs, (3) unconditional backbone generation. "
                "Returns PDB files of generated backbones. Follow with proteinmpnn_sequence_design."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["binder_design", "motif_scaffolding", "unconditional", "partial_diffusion"],
                        "description": "Design mode"
                    },
                    "target_pdb": {"type": "string", "description": "Path to target protein PDB (for binder_design)"},
                    "hotspot_residues": {"type": "string", "description": "Hotspot residues e.g. 'A30,A33,A34'"},
                    "contigs": {"type": "string", "description": "Contigs string e.g. 'A1-150/0 70-100'"},
                    "num_designs": {"type": "integer", "default": 10, "description": "Number of backbone designs to generate"},
                    "num_diffusion_steps": {"type": "integer", "default": 50},
                },
                "required": ["job_name", "mode"],
            },
            "_endpoint": f"{API_BASE}/design/rfdiffusion",
            "_method": "POST",
        }
    },

    # ─── ProteinMPNN ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "proteinmpnn_sequence_design",
            "description": (
                "Design amino acid sequences for a given protein backbone using ProteinMPNN. "
                "Always use AFTER rfdiffusion_design to get sequences for the generated backbones. "
                "Returns FASTA file with multiple sequence designs per backbone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string"},
                    "input_pdb": {"type": "string", "description": "Path to backbone PDB (from RFdiffusion)"},
                    "chains_to_design": {"type": "string", "default": "A", "description": "Chain IDs to redesign"},
                    "fixed_residues": {"type": "string", "description": "Residue positions to keep fixed"},
                    "num_sequences": {"type": "integer", "default": 8, "description": "Number of sequences per backbone"},
                    "sampling_temp": {"type": "number", "default": 0.1, "description": "Temperature (lower = more conservative)"},
                    "use_soluble_model": {"type": "boolean", "default": False},
                },
                "required": ["job_name", "input_pdb"],
            },
            "_endpoint": f"{API_BASE}/design/proteinmpnn",
            "_method": "POST",
        }
    },

    # ─── BindCraft ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "bindcraft_design",
            "description": (
                "Run BindCraft end-to-end binder design pipeline. "
                "Combines hallucination, AlphaFold2 validation, and filtering in one step. "
                "Use when you need a complete binder design workflow with automatic ranking. "
                "More automated than rfdiffusion+proteinmpnn but slower."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string"},
                    "target_pdb": {"type": "string", "description": "Target protein PDB path"},
                    "target_hotspots": {"type": "string", "description": "Hotspot residues to target"},
                    "num_designs": {"type": "integer", "default": 100},
                    "filters": {"type": "object", "description": "Optional filter parameters (pae, plddt thresholds)"},
                },
                "required": ["job_name", "target_pdb"],
            },
            "_endpoint": f"{API_BASE}/design/bindcraft",
            "_method": "POST",
        }
    },

    # ─── Fpocket ──────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "fpocket_detect_pockets",
            "description": (
                "Detect and characterize binding pockets on a protein structure using Fpocket. "
                "Returns pocket locations, volumes, druggability scores. "
                "Use before docking to identify candidate binding sites. Output pocket residues can be used as hotspot_residues for rfdiffusion_design, and center coordinates as docking box center for dock_ligand/autodock_gpu."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string"},
                    "input_pdb": {"type": "string", "description": "Path to protein PDB file"},
                    "min_sphere_size": {"type": "number", "default": 3.0},
                    "min_druggability_score": {"type": "number", "default": 0.0, "description": "Filter pockets below this druggability score (0-1)"},
                },
                "required": ["job_name", "input_pdb"],
            },
            "_endpoint": f"{API_BASE}/pocket/fpocket",
            "_method": "POST",
        }
    },

    # ─── P2Rank ───────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "p2rank_predict_pockets",
            "description": (
                "Predict SMALL MOLECULE binding pockets using P2Rank machine learning model. "
                "WARNING: P2Rank detects druggable cavities for small molecules, NOT protein-protein interfaces. "
                "For binder/ADC design, use PeSTo PPI interface prediction instead. "
                "P2Rank is useful for: molecular docking, drug discovery, pocket detection. "
                "Use 'alphafold' model when input is an AlphaFold prediction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string"},
                    "input_pdb": {"type": "string", "description": "Path to protein PDB file"},
                    "model": {
                        "type": "string",
                        "enum": ["default", "alphafold", "conservation"],
                        "default": "default",
                        "description": "Use 'alphafold' for AF2/AF3 predicted structures"
                    },
                },
                "required": ["job_name", "input_pdb"],
            },
            "_endpoint": f"{API_BASE}/pocket/p2rank",
            "_method": "POST",
        }
    },

    # ─── Smart Docking ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "dock_ligand",
            "description": (
                "Dock a small molecule ligand into a protein binding site. "
                "Auto-selects best engine based on parameters: "
                "- No binding site known → DiffDock (blind docking) "
                "- Known site, need CNN scoring → GNINA (recommended) "
                "- High-throughput screening → Vina-GPU "
                "- Highest accuracy → AutoDock-GPU. "
                "Returns binding poses with affinity scores (kcal/mol)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string"},
                    "engine": {
                        "type": "string",
                        "enum": ["gnina", "vina-gpu", "autodock-gpu", "diffdock"],
                        "default": "gnina",
                        "description": "Docking engine to use"
                    },
                    "receptor_pdb": {"type": "string", "description": "Prepared receptor PDB path"},
                    "ligand": {"type": "string", "description": "Ligand SMILES string (e.g. 'CC(=O)Oc1ccccc1C(=O)O')"},
                    "center_x": {"type": "number", "description": "Binding site center X coordinate"},
                    "center_y": {"type": "number", "description": "Binding site center Y coordinate"},
                    "center_z": {"type": "number", "description": "Binding site center Z coordinate"},
                    "box_size_x": {"type": "number", "default": 25.0},
                    "box_size_y": {"type": "number", "default": 25.0},
                    "box_size_z": {"type": "number", "default": 25.0},
                    "num_poses": {"type": "integer", "default": 9},
                    "exhaustiveness": {"type": "integer", "default": 8},
                },
                "required": ["job_name", "receptor_pdb", "ligand"],
            },
            "_endpoint": f"{API_BASE}/docking/dock",
            "_method": "POST",
        }
    },

    # ─── DiffDock ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "diffdock_blind_dock",
            "description": (
                "Perform blind docking using DiffDock diffusion model. "
                "No binding site specification needed - DiffDock finds the binding pose globally. "
                "Best when binding site is unknown. Input: receptor PDB + ligand SMILES."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string"},
                    "receptor_pdb": {"type": "string"},
                    "ligand_smiles": {"type": "string", "description": "Ligand SMILES string"},
                    "num_poses": {"type": "integer", "default": 10},
                    "inference_steps": {"type": "integer", "default": 20},
                },
                "required": ["job_name", "receptor_pdb", "ligand_smiles"],
            },
            "_endpoint": f"{API_BASE}/docking/diffdock",
            "_method": "POST",
        }
    },

    # ─── GROMACS ──────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "gromacs_md_simulation",
            "description": (
                "Run GROMACS molecular dynamics simulation on GPU. "
                "Full pipeline: topology → solvation → minimization → equilibration → production MD. "
                "Use to validate docking results, study conformational dynamics, "
                "or assess protein-ligand stability. "
                "Warning: Long-running (hours). Use run_md=False in pipeline unless essential."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string"},
                    "input_pdb": {"type": "string", "description": "Input PDB file path"},
                    "preset": {
                        "type": "string",
                        "enum": ["protein_water", "protein_ligand", "membrane_protein"],
                        "default": "protein_water"
                    },
                    "forcefield": {"type": "string", "default": "amber99sb-ildn"},
                    "sim_time_ns": {"type": "number", "default": 10.0, "description": "Simulation time in nanoseconds"},
                    "temperature_k": {"type": "integer", "default": 300},
                    "ligand_sdf": {"type": "string", "description": "Ligand SDF file from docking output (e.g. /data/oih/outputs/<job>/gnina/poses.sdf), triggers automatic GAFF2 parameterization for protein-ligand MD"},
                    "ligand_itp": {"type": "string", "description": "Ligand topology file (for protein_ligand preset)"},
                },
                "required": ["job_name", "input_pdb"],
            },
            "_endpoint": f"{API_BASE}/md/gromacs",
            "_method": "POST",
        }
    },

    # ─── Drug Discovery Pipeline ──────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "drug_discovery_pipeline",
            "description": (
                "Run COMPLETE drug discovery pipeline in one call: "
                "sequence → AlphaFold3 structure → P2Rank pocket detection → GNINA docking → (optional) GROMACS MD. "
                "Use when given a target sequence/structure and ligand SMILES and want end-to-end results. "
                "This is the highest-level workflow tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string"},
                    "target_sequence": {"type": "string", "description": "Target protein sequence (triggers AlphaFold3)"},
                    "target_pdb": {"type": "string", "description": "Or provide existing PDB path"},
                    "ligand_smiles": {"type": "string", "description": "Ligand SMILES to dock"},
                    "run_md": {"type": "boolean", "default": False, "description": "Run GROMACS MD on best pose (very slow)"},
                    "docking_engine": {
                        "type": "string",
                        "enum": ["gnina", "vina-gpu", "autodock-gpu", "diffdock"],
                        "default": "gnina"
                    },
                },
                "required": ["job_name", "ligand_smiles"],
            },
            "_endpoint": f"{API_BASE}/pipeline/drug-discovery",
            "_method": "POST",
        }
    },

    # ─── Binder Design Pipeline ───────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "binder_design_pipeline",
            "description": (
                "Run COMPLETE binder design + ADC construction pipeline (7 steps): "
                "fetch_pdb → RFdiffusion backbone → ProteinMPNN sequences → "
                "AlphaFold3 ipTM validation → FreeSASA conjugation sites → "
                "Linker selection → MMAE payload → RDKit ADC conjugation (DAR=4). "
                "Accepts pdb_id (e.g. '1N8Z') for automatic RCSB download, or target_pdb local path. "
                "Output includes adc_design: nanobody_sequence, iptm, conjugation_site, linker, payload, dar, adc_smiles. "
                "Use for: ADC design, nanobody design, binder design, antibody-drug conjugate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string"},
                    "pdb_id": {"type": "string", "description": "PDB ID to fetch from RCSB (e.g. '1N8Z'). Alternative to target_pdb."},
                    "target_pdb": {"type": "string", "description": "Local PDB file path. Alternative to pdb_id."},
                    "hotspot_residues": {"type": "array", "items": {"type": "string"}, "description": "Key residues e.g. ['S310','T311']"},
                    "num_designs": {"type": "integer", "default": 10, "description": "Number of RFdiffusion backbone designs"},
                    "num_rfdiffusion_designs": {"type": "integer", "default": 50},
                    "num_mpnn_sequences": {"type": "integer", "default": 8},
                    "run_af3_validation": {"type": "boolean", "default": True},
                    "dry_run": {"type": "boolean", "default": False, "description": "Return mock results without running tools"},
                },
                "required": ["job_name"],
            },
            "_endpoint": f"{API_BASE}/pipeline/binder-design",
            "_method": "POST",
        }
    },



    # ─── ESM2 Protein Embedding ───────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "esm_embed",
            "description": (
                "Generate protein sequence embeddings using ESM2 language model. "
                "Use for: sequence similarity analysis, clustering, feature extraction. "
                "Returns mean embeddings and optional similarity matrix."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string"},
                    "sequences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of protein sequences"
                    },
                    "task": {
                        "type": "string",
                        "enum": ["embed", "similarity"],
                        "default": "embed"
                    },
                },
                "required": ["job_name", "sequences"],
            },
            "_endpoint": f"{API_BASE}/ml/esm/embed",
            "_method": "POST",
        }
    },

    # ─── ESM2 Sequence Scoring ─────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "esm2_score_sequences",
            "description": (
                "Score protein sequences by ESM2 pseudo-perplexity (PPL). "
                "Lower PPL = more natural/foldable sequence. "
                "USE AS FILTER: after ProteinMPNN generates 50-100 sequences, "
                "score all with ESM2 → keep PPL < 10 → then IgFold/AF3 on survivors. "
                "Speed: ~0.5s per sequence on GPU. "
                "Upstream: proteinmpnn_sequence_design → sequences. "
                "Downstream: igfold_predict or alphafold3_predict."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Job identifier"},
                    "sequences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Protein sequences to score"
                    },
                },
                "required": ["job_name", "sequences"],
            },
            "_endpoint": f"{API_BASE}/ml/esm/score",
            "_method": "POST",
        }
    },

    # ─── ESM-1v Mutant Scanning ────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "esm2_mutant_scan",
            "description": (
                "In-silico deep mutational scanning using ESM-1v. "
                "For each position, predicts effect of all 19 amino acid substitutions. "
                "Returns ΔΔG proxy scores: positive = destabilizing, negative = stabilizing. "
                "USE FOR: (1) affinity maturation — find stabilizing mutations in CDR loops, "
                "(2) resistance prediction — identify positions sensitive to mutation, "
                "(3) humanization — find mutations that improve human-likeness. "
                "Input: wild-type sequence + optional scan_positions (1-based). "
                "If scan_positions omitted, scans ALL positions (slower). "
                "For CDR-focused scanning, provide CDR loop positions only. "
                "Speed: ~1s for full sequence scan on GPU."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Job identifier"},
                    "sequence": {"type": "string", "description": "Wild-type protein sequence"},
                    "scan_positions": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "1-based positions to scan (default: all positions)"
                    },
                },
                "required": ["job_name", "sequence"],
            },
            "_endpoint": f"{API_BASE}/ml/esm/mutant_scan",
            "_method": "POST",
        }
    },

    # ─── Chemprop Property Prediction ─────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "chemprop_predict",
            "description": (
                "Predict molecular properties (ADMET, bioactivity, toxicity) using Chemprop GNN. "
                "Use for: drug-likeness screening, toxicity prediction, solubility estimation. "
                "Input: list of SMILES strings + model_path. "
                "Available ADMET models (trained on standard benchmarks): "
                "  /data/oih/models/admet/esol/model_0/best.pt — aqueous solubility (logS, regression, ESOL 1128mol, MSE=0.54) "
                "  /data/oih/models/admet/freesolv/model_0/best.pt — hydration free energy (kcal/mol, regression, FreeSolv 642mol) "
                "  /data/oih/models/admet/lipophilicity/model_0/best.pt — lipophilicity (logD, regression, 4200mol, MSE=0.43) "
                "  /data/oih/models/admet/bbbp/model_0/best.pt — blood-brain barrier penetration (classification 0/1, AUC=0.89) "
                "  /data/oih/models/admet/tox21/model_0/best.pt — Tox21 NR-AhR toxicity (classification 0/1, AUC=0.90) "
                "For full ADMET profiling, run predict once per model. "
                "Downstream: results feed into drug optimization decisions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string"},
                    "smiles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of SMILES strings to predict"
                    },
                    "model_path": {
                        "type": "string",
                        "description": (
                            "Path to trained Chemprop model checkpoint (.pt). "
                            "Use one of: /data/oih/models/admet/{esol,freesolv,lipophilicity,bbbp,tox21}/model_0/best.pt"
                        )
                    },
                },
                "required": ["job_name", "smiles", "model_path"],
            },
            "_endpoint": f"{API_BASE}/ml/chemprop/predict",
            "_method": "POST",
        }
    },

    # ─── Literature RAG ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": (
                "Search PubMed + bioRxiv for relevant scientific literature. "
                "Use when: (1) user asks about mechanisms, background, or recent research, "
                "(2) need to justify a computational choice with literature evidence, "
                "(3) user asks 'what is known about X', 'recent papers on Y'. "
                "Returns papers with PMID, title, abstract, and relevance score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query in English or Chinese"},
                    "n_pubmed": {"type": "integer", "default": 6, "description": "Number of PubMed results"},
                    "n_biorxiv": {"type": "integer", "default": 3, "description": "Number of bioRxiv preprints"},
                    "years_back": {"type": "integer", "default": 5, "description": "Search papers from last N years"},
                },
                "required": ["query"],
            },
            "_endpoint": f"{API_BASE}/rag/search",
            "_method": "GET",
        }
    },

    # ─── RAG Search (alias with ADC-oriented description) ───────────────────
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "查询生物医学知识库，实时检索PubMed和bioRxiv文献。"
                "适用场景：了解靶点生物学背景、查询已知ADC临床数据、药物作用机制、相关文献证据。"
                "建议在设计药物前先调用此工具了解靶点信息。"
                "上游：用户提问中涉及的靶点/药物名称。"
                "下游：将检索结果作为背景知识，指导后续rfdiffusion/gnina/rdkit_conjugate等工具的参数选择。"
                "输出：相关论文列表（标题+摘要+PMID+来源）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词，如 'HER2 ADC trastuzumab' 或 'KRAS G12C inhibitor binding pocket'"},
                    "n_pubmed": {"type": "integer", "default": 5, "description": "PubMed返回条数"},
                    "n_biorxiv": {"type": "integer", "default": 2, "description": "bioRxiv返回条数"},
                },
                "required": ["query"],
            },
            "_endpoint": f"{API_BASE}/rag/search",
            "_method": "GET",
        }
    },

    # ─── FreeSASA Conjugation Site Analysis ─────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "freesasa",
            "description": (
                "Calculate per-residue Solvent Accessible Surface Area (SASA) on an antibody PDB "
                "and identify exposed Lys (K) and Cys (C) residues (SASA > 40 Å²) as recommended "
                "ADC conjugation sites. "
                "Upstream: use antibody PDB from rfdiffusion_design or proteinmpnn_sequence_design output. "
                "Downstream: pass the returned conjugation_site (e.g. 'K127') to rdkit_conjugate as the "
                "conjugation_site parameter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Unique job identifier"},
                    "input_pdb": {"type": "string", "description": "Antibody PDB path (from RFdiffusion/ProteinMPNN output)"},
                },
                "required": ["job_name", "input_pdb"],
            },
            "_endpoint": f"{API_BASE}/design/freesasa",
            "_method": "POST",
        }
    },

    # ─── Linker Select ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "linker_select",
            "description": (
                "从临床验证的ADC linker库（20条）中筛选最适合的linker。"
                "未指定clinical_status时优先返回已上市ADC使用的linker（approved级别）。"
                "可按以下条件筛选："
                "cleavable(true=溶酶体/GSH释放,false=不可裂解)；"
                "reaction_type(maleimide_thiol/nhs_amine/disulfide/hydrazone/oxime/dbco_azide/transglutaminase)；"
                "compatible_payload(如MMAE/DM1/SN-38/calicheamicin)；"
                "clinical_status(approved/clinical/research)。"
                "示例：Kadcyla类ADC→cleavable=false,compatible_payload=DM1→返回SMCC。"
                "示例：Adcetris类ADC→cleavable=true,compatible_payload=MMAE→返回MC-VC-PABC。"
                "上游：fetch_molecule获取payload_smiles后调用此工具选linker。"
                "下游：rdkit_conjugate(linker_smiles,linker_name,payload_smiles)执行偶联。"
                "返回结果含recommendation字段说明推荐理由。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "payload_smiles": {"type": "string", "description": "Payload SMILES (from fetch_molecule)"},
                    "linker_type": {
                        "type": "string",
                        "enum": ["cleavable", "non_cleavable"],
                        "description": "Legacy filter (use cleavable bool instead)"
                    },
                    "cleavable": {"type": "boolean", "description": "true=cleavable linker, false=non-cleavable"},
                    "reaction_type": {
                        "type": "string",
                        "description": "Filter by reaction chemistry: maleimide_thiol, nhs_amine, disulfide, hydrazone, oxime, dbco_azide, transglutaminase"
                    },
                    "compatible_payload": {"type": "string", "description": "Payload name for fuzzy match (e.g. MMAE, DM1, SN-38)"},
                    "clinical_status": {"type": "string", "enum": ["approved", "clinical", "research"], "description": "Filter by development stage"},
                    "max_results": {"type": "integer", "default": 5, "description": "Max results to return"},
                },
                "required": [],
            },
            "_endpoint": f"{API_BASE}/adc/linker_select",
            "_method": "POST",
        }
    },

    # ─── RDKit ADC Conjugate ──────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "rdkit_conjugate",
            "description": (
                "生成ADC的linker-payload共价结合体（小分子部分）。"
                "ADC结构：抗体-Cys/Lys ←[linker抗体端] — [linker中段] — [linker payload端]→ payload。"
                "重要：maleimide/NHS端连接的是抗体（不是payload），payload通过linker另一端连接："
                "VC-PABC类用PABC氨基甲酸酯连payload的OH/NH2；"
                "disulfide类用二硫键连payload的SH；"
                "hydrazone类用腙键连payload的酮基。"
                "MMAE与VC-PABC的连接：PABC羟基+MMAE N端胺基→氨基甲酸酯键。"
                "covalent=false时仍可用dot-disconnected SMILES做ADMET，不影响后续流程。"
                "上游：linker_select返回linker_smiles+linker_name，fetch_molecule返回payload_smiles。"
                "下游：chemprop_predict用adc_smiles做ADMET预测。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Unique job identifier"},
                    "antibody_pdb": {"type": "string", "description": "Antibody PDB path"},
                    "conjugation_site": {"type": "string", "description": "Conjugation residue e.g. 'K127' (from freesasa)"},
                    "linker_smiles": {"type": "string", "description": "Linker SMILES (from linker_select)"},
                    "payload_smiles": {"type": "string", "description": "Payload SMILES (from fetch_molecule)"},
                    "linker_name": {"type": "string", "description": "Linker name for reaction type lookup (e.g. VC-PABC, SMCC, SPDP)"},
                    "reaction_type": {
                        "type": "string",
                        "default": "auto",
                        "description": "Reaction chemistry: auto, maleimide_thiol, nhs_amine, hydrazone, oxime, disulfide, dbco_azide, transglutaminase"
                    },
                },
                "required": ["job_name", "antibody_pdb", "conjugation_site", "linker_smiles", "payload_smiles"],
            },
            "_endpoint": f"{API_BASE}/adc/rdkit_conjugate",
            "_method": "POST",
        }
    },

    # ─── Pocket-Guided Binder + ADC Pipeline ────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "pocket_guided_binder_pipeline",
            "description": (
                "Run COMPLETE pocket-guided binder design + ADC pipeline (16 steps) with "
                "6D pocket scoring (including DiscoTope3 epitope prediction) + BindCraft parallel path: "
                "1) fetch_pdb → 2) RAG literature search (binding site evidence, residue extraction) → "
                "3) fpocket + P2Rank + DiscoTope3 (parallel: pocket detection + B-cell epitope prediction) → "
                "4) FreeSASA per pocket (surface exposure) → "
                "5) B-factor conservation + electrostatics → "
                "6) 6D Composite scoring (p2rank×0.15 + sasa×0.15 + conservation×0.15 + rag×0.25 + "
                "electrostatics×0.10 + epitope×0.20) + Qwen structural biologist pocket selection → "
                "7) DiffDock druggability reference (on selected pocket only) → "
                "8a) RFdiffusion binder backbone (epitope-enriched hotspots) + 8b) BindCraft (parallel) → "
                "9) ProteinMPNN → 10) merge MPNN+BindCraft candidates → "
                "11) AlphaFold3 validation (ipTM≥0.6) → "
                "12-16) ADC construction: FreeSASA conjugation sites → Linker → MMAE → RDKit (DAR=4). "
                "DiscoTope3 predicts B-cell epitopes (antibody binding sites) on the target antigen — "
                "pockets overlapping with predicted epitopes are prioritized for binder design. "
                "If DiscoTope3 fails, epitope score falls back to 0.0 (non-blocking). "
                "Unlike binder_design_pipeline, this pipeline AUTOMATICALLY detects and scores "
                "binding pockets using 6 criteria + literature evidence — no hotspot_residues needed. "
                "Returns: pocket_scores (per-pocket 6D breakdown), selected_pocket (id/center/hotspots/epitope_enriched/reason), "
                "diffdock_reference (confidence/pose_path). "
                "Use when: user says 'design a binder for [target]' without specifying residues, "
                "'find pockets and design binder', 'pocket-guided design', or any binder/ADC "
                "request where binding site is unknown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Unique job identifier"},
                    "pdb_id": {"type": "string", "description": "PDB ID to fetch from RCSB (e.g. '2A91')"},
                    "chains": {"type": "string", "description": "Optional: chains to keep (e.g. 'A')"},
                    "probe_smiles": {
                        "type": "string",
                        "default": "CC(=O)Oc1ccccc1C(=O)O",
                        "description": "Small probe molecule SMILES for DiffDock druggability reference (default: aspirin)"
                    },
                    "target_name": {"type": "string", "description": "Target protein name (e.g. 'HER2') for tier classification. Known targets: HER2, PD-L1, EGFR, VEGF, CD20, TNF"},
                    "antigen_chain": {"type": "string", "description": "Antigen chain ID in the PDB (for tier 1 interface extraction)"},
                    "binder_type": {
                        "type": "string",
                        "default": "de_novo",
                        "enum": ["de_novo", "nanobody", "antibody"],
                        "description": "Binder type: 'de_novo' (default, skip IgFold), 'nanobody'/'antibody' (enable IgFold pLDDT filter). IgFold only works on antibody sequences — de novo binders get meaningless scores."
                    },
                    "num_rfdiffusion_designs": {"type": "integer", "default": 10},
                    "num_mpnn_sequences": {"type": "integer", "default": 8},
                    "binder_length": {"type": "string", "default": "70-120", "description": "Binder length range"},
                    "run_diffdock_validation": {"type": "boolean", "default": True},
                    "run_af3_validation": {"type": "boolean", "default": True},
                    "dry_run": {"type": "boolean", "default": False},
                },
                "required": ["job_name", "pdb_id"],
            },
            "_endpoint": f"{API_BASE}/pipeline/pocket-guided-binder",
            "_method": "POST",
        }
    },

    # ─── Execute Python ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute Python code in a sandboxed environment with scientific libraries. "
                "Available: matplotlib, pandas, numpy, scipy, rdkit, json, csv, math. "
                "Use for: data analysis, plotting docking results, parsing output files, "
                "computing statistics, generating figures. "
                "plt.savefig() outputs go to /data/oih/outputs/plots/. "
                "plt.show() is auto-redirected to savefig. "
                "Returns stdout output and plot_path if a figure was saved. "
                "Do NOT use for: file deletion, system commands, network requests."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "python_code": {
                        "type": "string",
                        "description": "Python code to execute. Can read files under /data/oih/outputs/."
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 30,
                        "description": "Max execution time in seconds (5-120)"
                    },
                },
                "required": ["python_code"],
            },
            "_endpoint": f"{API_BASE}/analysis/execute_python",
            "_method": "POST",
        }
    },

    # ─── Read Results File ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "read_results_file",
            "description": (
                "Read a results file (CSV, JSON, TXT, LOG) from the outputs directory. "
                "Use when you need to inspect docking results, AF3 confidence scores, "
                "GROMACS logs, MPNN FASTA files, or any tool output. "
                "Supports directories (returns file listing). "
                "Returns file contents as string (max 10000 chars by default). "
                "Allowed paths: /data/oih/outputs/, /data/oih/inputs/, /data/af3/output/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to file or directory under /data/oih/outputs/"
                    },
                    "max_chars": {
                        "type": "integer",
                        "default": 10000,
                        "description": "Max characters to return (100-50000)"
                    },
                },
                "required": ["file_path"],
            },
            "_endpoint": f"{API_BASE}/analysis/read_results_file",
            "_method": "POST",
        }
    },

    # ─── IgFold Antibody Structure Prediction ────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "igfold_predict",
            "description": (
                "Predict antibody/nanobody 3D structure from sequence using IgFold (~2s per sequence on GPU). "
                "USE AS PRE-FILTER before AF3: ProteinMPNN outputs 50-100 sequences → IgFold predicts all → "
                "filter by mean_pLDDT > 70 → send only top 5-10 to AF3 complex validation. "
                "This avoids running AF3 on all candidates (3-10 min each). "
                "Input: sequences dict with 'H' (heavy chain) and optionally 'L' (light chain). "
                "For nanobodies: only 'H' chain needed. "
                "Output: PDB structure, per-residue pLDDT scores, mean pLDDT. "
                "Upstream: proteinmpnn_sequence_design → sequences. "
                "Downstream: alphafold3_predict (validates top IgFold hits as complex with target)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Job identifier (no spaces)"},
                    "sequences": {
                        "type": "object",
                        "description": "Chain sequences dict, e.g. {\"H\": \"EVQLVE...\"}. Nanobody = H chain only."
                    },
                    "do_refine": {
                        "type": "boolean",
                        "default": False,
                        "description": "OpenMM structure refinement (slower but better geometry, use for final candidates only)"
                    },
                },
                "required": ["job_name", "sequences"],
            },
            "_endpoint": f"{API_BASE}/immunology/igfold",
            "_method": "POST",
        }
    },

    # ─── DiscoTope3 Epitope Prediction ────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "discotope3_predict",
            "description": (
                "Predict B-cell epitopes on protein structures using DiscoTope3 (deep learning + XGBoost ensemble). "
                "Input: PDB file path (antibody or antigen structure). "
                "Output: per-residue epitope propensity scores with calibrated normalization. "
                "WARNING: DiscoTope3 predicts B-cell EPITOPES (immunogenicity), NOT optimal binder design sites. "
                "For binder hotspot selection, use PeSTo PPI interface prediction instead. "
                "CD36 proof: DiscoTope3 A397-400 → 0/10 AF3 pass; PeSTo A187-194 → much better. "
                "Use cases: (1) identify antigen surface epitopes for VACCINE design (not binder), "
                "(2) validate predicted antibody-antigen interfaces from AF3, "
                "(4) epitope mapping for vaccine antigen selection. "
                "For AlphaFold-predicted structures, set struc_type='alphafold'. "
                "Threshold: low=0.40 (sensitive), moderate=0.90 (default), high=1.50 (strict). "
                "Upstream: fetch_pdb or alphafold3_predict → input_pdb. "
                "Downstream: use high-scoring epitope residues as hotspot_residues for rfdiffusion_design or pocket_guided_binder_pipeline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Job identifier (no spaces)"},
                    "input_pdb": {"type": "string", "description": "Path to protein PDB file"},
                    "struc_type": {
                        "type": "string",
                        "enum": ["solved", "alphafold"],
                        "default": "solved",
                        "description": "Structure type: 'solved' for experimental, 'alphafold' for AF2/AF3 predictions"
                    },
                    "calibrated_score_epi_threshold": {
                        "type": "number",
                        "default": 0.90,
                        "description": "Epitope score threshold: 0.40 (sensitive), 0.90 (moderate), 1.50 (strict)"
                    },
                    "multichain_mode": {
                        "type": "boolean",
                        "default": False,
                        "description": "Predict epitopes on entire complex (all chains)"
                    },
                    "cpu_only": {
                        "type": "boolean",
                        "default": False,
                        "description": "Force CPU-only inference"
                    },
                },
                "required": ["job_name", "input_pdb"],
            },
            "_endpoint": f"{API_BASE}/immunology/discotope3",
            "_method": "POST",
        }
    },

    # ─── Extract Interface Residues ──────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "extract_interface_residues",
            "description": (
                "Extract antibody-antigen interface residues from a known complex PDB structure. "
                "USE WHEN: A co-crystal structure of the target with an antibody/nanobody exists in PDB. "
                "This gives experimentally validated hotspot residues — much more reliable than computational prediction. "
                "DECISION LOGIC: "
                "- Target has known antibody complex PDB → use this tool FIRST. "
                "- No known complex → use discotope3_predict + rag_search instead. "
                "KNOWN COMPLEXES: "
                "- HER2/ERBB2: 1N8Z (trastuzumab, receptor_chain=C, ligand_chains=[A,B]). "
                "- PD-L1: 5XXY (atezolizumab, receptor_chain=A, ligand_chains=[B]). "
                "- EGFR: 1YY9 (cetuximab, receptor_chain=A, ligand_chains=[B]). "
                "- VEGF: 1BJ1 (bevacizumab, receptor_chain=V, ligand_chains=[A,B]). "
                "Upstream: fetch_pdb. "
                "Downstream: rfdiffusion_design (pass interface_residues as hotspot_residues)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Job identifier"},
                    "complex_pdb": {"type": "string", "description": "Path to complex PDB with both receptor and antibody chains"},
                    "receptor_chain": {"type": "string", "description": "Chain ID for receptor/antigen, e.g. 'C'"},
                    "ligand_chains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Chain IDs for antibody/ligand, e.g. ['A','B']"
                    },
                    "cutoff_angstrom": {"type": "number", "default": 5.0, "description": "Distance cutoff for contacts (Å)"},
                    "top_n": {"type": "integer", "default": 8, "description": "Max interface residues to return"},
                },
                "required": ["job_name", "complex_pdb", "receptor_chain", "ligand_chains"],
            },
            "_endpoint": f"{API_BASE}/structure/extract_interface",
            "_method": "POST",
        }
    },

    # ─── Report Generation ─────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "Generate a comprehensive English research report for a drug discovery project. "
                "Collects all experiment results (AF3 ipTM, ADC SMILES, FreeSASA sites, MPNN sequences), "
                "queries RAG for latest literature, and uses Qwen to write a professional analysis report. "
                "Returns markdown report + structured data. Use when user says 'generate report', "
                "'write summary', 'export results', or after completing a full binder/ADC pipeline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "default": "HER2", "description": "Target protein name"},
                    "job_prefix": {"type": "string", "description": "Job name prefix to filter results, e.g. 'her2_tier1_v2'"},
                    "include_rag": {"type": "boolean", "default": True, "description": "Include RAG literature search"},
                },
                "required": ["job_prefix"],
            },
            "_endpoint": f"{API_BASE}/report/generate",
            "_method": "POST",
        }
    },

    # ─── PeSTo PPI Interface Prediction ──────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "pesto_predict",
            "description": (
                "Predict protein-protein interaction (PPI) interface from a protein structure using PeSTo "
                "(Protein Structure Transformer). ROC AUC=0.92, outperforms MaSIF-site(0.80) and DiscoTope3. "
                "Input: PDB single chain (IMPORTANT: for complex PDBs, extract target chain first). "
                "Output: per-residue PPI interface probability (0-1), hotspot residues (score>0.5). "
                "USE FOR: de novo binder design hotspot selection (replaces P2Rank + DiscoTope3). "
                "DO NOT use on full complex PDB — extract the target protein chain first, otherwise "
                "scores are suppressed at already-occupied interfaces (PD-L1: complex 0.44 → single chain 0.99). "
                "Deployed in oih-proteinmpnn container (/app/pesto/). CPU only, ~10s per structure. "
                "Upstream: fetch_pdb (extract target chain). "
                "Downstream: hotspot residues → rfdiffusion_design or pocket_guided_binder_pipeline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Job identifier"},
                    "input_pdb": {"type": "string", "description": "Path to PDB file (SINGLE CHAIN of target protein)"},
                    "threshold": {"type": "number", "default": 0.5, "description": "Score threshold for hotspot residues"},
                },
                "required": ["job_name", "input_pdb"],
            },
            "_endpoint": f"{API_BASE}/structure/pesto_predict",
            "_method": "POST",
        }
    },

    # ─── System Status ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": (
                "查看OIH平台系统状态：所有容器运行情况、GPU队列、任务统计。"
                "用户问'有多少容器在运行'/'GPU状态'/'系统状态'/'平台状态'时调用此工具。"
                "返回containers(各容器状态)、gpu_queue(CPU/GPU/Degraded队列占用)、task_summary(运行中/今日完成/今日失败数)。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
            "_endpoint": f"{API_BASE}/system/status",
            "_method": "GET",
        }
    },

    # ─── Task Polling ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "poll_task_status",
            "description": (
                "Check the status of a submitted computational job. "
                "All bio-computing tools run asynchronously and return a task_id. "
                "Call this to check if the job is pending/running/completed/failed. "
                "Poll every 30-60 seconds for long jobs. "
                "When status='completed', result contains output file paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID returned from any tool call"},
                },
                "required": ["task_id"],
            },
            "_endpoint": f"{API_BASE}/tasks/{{task_id}}",
            "_method": "GET",
        }
    },

]


# ─── System Prompt for Qwen3-14B ────────────────────────────────────────────

QWEN_SYSTEM_PROMPT = """你是OIH生物计算平台的AI助手，运行在OIH服务器上。
你可以调用以下完整的药物发现和蛋白质设计工具链：

**结构预测**
- alphafold3_predict: 预测蛋白质/RNA/DNA/小分子复合物结构

**蛋白质设计**
- rfdiffusion_design: 从头设计蛋白质骨架（扩散模型）
- proteinmpnn_sequence_design: 为骨架设计氨基酸序列（在RFdiffusion后使用）
- bindcraft_design: 一体化binder设计流程

**结合位点分析**
- fpocket_detect_pockets: 检测结合口袋
- p2rank_predict_pockets: ML预测结合位点（AlphaFold结构推荐用alphafold模式）
- discotope3_predict: B细胞表位预测（抗原表面→抗体结合位点，可指导binder设计hotspot选择）。⚠️ Tier 1靶标（有已知复合物）时优先用extract_interface_residues。
- igfold_predict: 抗体/纳米抗体序列→3D结构快速预测（~2秒/条，用作MPNN→AF3之间的预筛选，pLDDT>70筛选）。⚠️ 仅用于binder_type='nanobody'/'antibody'，de novo binder禁止调用。
- extract_interface_residues: 从已知抗体-抗原复合物PDB提取界面残基（Tier 1最可靠方法）。已知靶标：HER2→1N8Z, PD-L1→5XXY, EGFR→1YY9, VEGF→1BJ1。

**ADC设计**
- freesasa: 计算抗体SASA，推荐ADC偶联位点（Lys/Cys, SASA>40Å²）
- linker_select: 推荐ADC连接子（cleavable/non_cleavable）
- rdkit_conjugate: RDKit构建payload-linker偶联物，输出SMILES+SDF

**ML分析**
- esm_embed: ESM2蛋白语言模型序列嵌入（相似性分析/特征提取）
- esm2_score_sequences: ESM2伪困惑度评分（PPL越低=序列越自然，用于ProteinMPNN→IgFold之间预筛选）
- esm2_mutant_scan: ESM-1v深度突变扫描（每个位置19种突变的ΔΔG估计，用于亲和力成熟/耐药预测）
- chemprop_predict: 分子性质预测（ADMET/毒性/溶解度）

**分子对接**
- dock_ligand: 智能路由对接（GNINA/Vina-GPU/AutoDock-GPU/DiffDock）
- diffdock_blind_dock: 盲对接（不需要指定结合位点）

**分子动力学**
- gromacs_md_simulation: GPU加速MD模拟（耗时较长）

**一键流程**
- drug_discovery_pipeline: 靶点序列+配体SMILES → 完整药物发现流程
- binder_design_pipeline: 靶点PDB/PDB_ID + 已知hotspot残基 → 完整binder+ADC设计流程（7步）。用户已经知道结合位点残基时使用。
- pocket_guided_binder_pipeline: 靶点PDB_ID → 6D口袋评分 + binder设计 + ADC构建（16步：fetch_pdb→RAG文献检索→fpocket+P2Rank+DiscoTope3(并行)→FreeSASA表面暴露→B-factor保守性+静电→6D复合评分(p2rank+sasa+conservation+rag+electrostatics+epitope)+Qwen口袋选择→DiffDock成药性参考→RFdiffusion+BindCraft(并行,epitope-enriched hotspots)→ProteinMPNN→候选合并→AF3验证→FreeSASA偶联位点→Linker→MMAE→RDKit偶联DAR=4）。DiscoTope3预测B细胞表位，epitope维度权重0.20。用户说"设计binder"/"设计ADC"但**未指定残基**时优先调用此pipeline。

**文献检索**
- rag_search / search_literature: 实时检索PubMed + bioRxiv文献（支持中英文查询，建议在药物设计前先查靶点背景）

**数据分析**
- execute_python: 执行Python代码（matplotlib/pandas/numpy/scipy），生成图表保存到/data/oih/outputs/plots/
- generate_report: 生成英文研究报告（收集实验数据+RAG文献+Qwen分析），返回Markdown+PDF。用户说"生成报告/导出结果/write report"时调用。
- read_results_file: 读取CSV/JSON/TXT结果文件，查看工具输出内容

**任务管理**
- poll_task_status: 查询异步任务状态（所有工具异步返回task_id）

工作流程：
1. 提交任务 → 获得task_id
2. 每30-60秒调用poll_task_status查询状态
3. status='completed'时读取result中的输出文件路径
4. 将结果传递给下一步工具

所有工具输出文件在 /data/oih/outputs/ 下各子目录（如 fetch_pdb → /data/oih/outputs/fetch_pdb/）。
**重要：下一个工具的文件路径必须从上一个工具返回值中读取（如 output_pdb 字段），绝不要自行推断或硬编码路径。例如 fetch_pdb 返回 output_pdb=/data/oih/outputs/fetch_pdb/5XWR.pdb，则 fpocket 的 input_pdb 必须填这个值。**
计算使用GPU1（Nvidia），LLM推理在GPU0。

## 错误处理原则（非常重要）
当工具返回error时，必须：
1. **仔细阅读error内容**，找出根本原因
2. **不要盲目重试**相同参数
3. 常见错误和解决方法：
   - "chain does not contain recognized molecule" → PDB含非蛋白链(HETATM)，input_pdb改用clean版本或只保留ATOM记录
   - "No such file or directory" → 所需输入文件尚未下载。必须先调用 fetch_pdb 或 fetch_molecule 获取文件，再用返回的文件路径重试原工具。切勿用不存在的路径重试。
   - "file does not exist" → 上一步失败导致文件未生成，先修复上一步
   - "Invalid command-line options" → 参数格式错误，检查参数
   - "CUDA out of memory" → 显存不足，减小系统规模
4. 修复参数后再重试，每个问题最多重试2次
5. 如果无法自动修复，向用户解释错误原因并请求帮助


## 工具适用范围规则（Qwen必须遵守）

在调用以下工具前，必须检查适用条件：

**igfold_predict**: 只在 binder_type='nanobody' 或 'antibody' 时调用。de novo binder 设计禁止调用（IgFold 使用 AntiBERTy，只在抗体数据集训练，非抗体序列输出无意义 pLDDT=5-40）。de novo binder 直接用 esm2_score_sequences 过滤后送 AF3。

**discotope3_predict**: 只在抗体/binder 设计场景调用。小分子对接场景禁止调用。已知共晶结构时优先用 extract_interface_residues。

**extract_interface_residues**: 已知抗体-抗原复合物PDB时优先调用（Tier 1）。比 DiscoTope3 预测更可靠。HER2 → 用 1N8Z (chain C vs A,B)。

**binder_type 判断规则**:
- 用户说"纳米抗体/nanobody/VHH" → binder_type='nanobody'
- 用户说"设计binder/结合蛋白" → binder_type='de_novo'
- 用户说"抗体/antibody/IgG" → binder_type='antibody'
- 未指定时默认 binder_type='de_novo'

**target_name 规则**: 传入靶标名称（如 'HER2'）会自动触发 Tier 1 路径（从已知复合物提取界面残基），跳过 DiscoTope3 + 6D scoring。已知靶标：HER2, PD-L1, EGFR, VEGF, CD20, TNF。

**热点自动聚类**: pipeline 自动对热点按空间距离聚类（15Å阈值）。分散热点 → 分组设计 → 各自 RFdiffusion → 合并 AF3 验证。不需要手动分组。CD36 教训：5 个分散热点跨 236 残基 → ipTM=0.43 全部失败。

**RAG 优先级**: PPI interface > epitope。RAG 搜到共晶结构/突变验证的结合残基时，必须优先用这些做 hotspot，覆盖 DiscoTope3 表位预测。B 细胞表位 ≠ 最佳 binder 设计位点。

## Binder 设计推理框架（每次设计任务必须遵循）

收到 binder/ADC 设计请求时，按此顺序推理：

**Step 1: 靶标识别**
- 识别靶标名称 + PDB ID
- 判断 Tier: KNOWN_COMPLEXES 命中 → Tier 1; 否则 → Tier 3

**Step 2: Hotspot 来源决策**
- Tier 1: extract_interface_residues → 实验验证 hotspot（最可靠）
- Tier 3 优先级: RAG文献界面 > PeSTo PPI > conservation > DiscoTope3 epitope
- 绝不用 DiscoTope3 单独决定 hotspot（CD36 教训：0/10 pass）

**Step 3: 评分公式**
composite = rag(0.30) + pesto_ppi(0.25) + conservation(0.20) + sasa(0.10) + electrostatics(0.15)
已移除: P2Rank（小分子口袋）、DiscoTope3 epitope（免疫原性≠PPI）

**Step 4: 设计参数**
- binder_type: de_novo（默认）→ 跳过 IgFold; nanobody → 启用 IgFold
- 域截取: >500aa 必须截取到 ~200aa（DOMAIN_REGISTRY）
- 双路: RFdiffusion(hotspot) + BindCraft(free explore)

**Step 4.5: CRITICAL — MPNN chain 检测（2026-03-24血泪教训）**
RFdiffusion binder_design 输出：最短链=binder, 最长链=target。
**绝不能硬编码 chains_to_design='A'**。必须检测最短链：
- 正确: chains_to_design = shortest_chain（自动检测）
- 错误: chains_to_design = 'A'（导致设计 target 蛋白而非 binder）
验证: MPNN FASTA 的 original 序列应是 60-100aa(binder)，不是几百aa(target)
教训: 273次 MPNN 全部设计了错误的链，pipeline.py 已修复为动态检测

**Step 5: 验证**
- AF3 num_seeds=3, ipTM ≥ 0.6
- 域截取后 AF3 更准（HER2: 全长 0.84 vs 截取 0.86）

**Step 6: ADC（仅 pass 的设计）**
- FreeSASA → Lys/Cys → NHS-amine/maleimide → RDKit conjugate

**PeSTo 靶点难度速查**:
TrkA(0.999极容易) PD-L1(0.994容易) Nectin-4(0.966容易) CD36(0.865中等) EGFR(0.759中等) TROP2(0.422困难)

**域截取决策（AF3验证前必须执行）**:
1. 查 DOMAIN_REGISTRY → 有就用（HER2/CD36/EGFR/PDL1/Nectin4/TrkA/TROP2 已注册）
2. 没有 → hotspot ± 50 残基，目标 100-250aa
3. >500aa 必须截取（CD36 全长 469aa → ipTM=0.33 失败）
4. 边界选 loop/linker，不切断二级结构/二硫键

**多靶点并行调度（GPU 44GB）**:
- Pipeline 在 CPU 队列，不占 GPU slot
- GPU semaphore=3: RFdiffusion 可3并行(各4-10GB)，AF3 独占(20GB)+1小任务
- 批量提交所有靶点，调度器自动按 VRAM 排队
- CPU任务(PeSTo/RAG/FreeSASA)随时跑不受GPU限制
- 失败恢复: RFdiffusion backbone 保留在 outputs/，MPNN 可手动补跑
- 更多designs≠更好(CD36: n=10 ipTM=0.55 vs n=50 ipTM=0.18)

# === 工具注意事项（自动同步自 CLAUDE.md） ===

## 通用规则
- > **Always use `gpu_id=0` or `device=0` inside any container. Never use 1.**
- JAX-based containers (AlphaFold3, BindCraft): **never mount host cuda lib64 paths** — JAX bundles its own CUDA libraries. Mounting host paths breaks GPU detection.
- CPU tools → always cpu_sem
- `http_proxy=http://127.0.0.1:7890` is set in the shell environment. Always start uvicorn with:
- All stages passed; trajectory at `/data/oih/outputs/test_gromacs_fix_v2/gromacs/md.xtc`
- `qwen_agent.py`：记录每个工具连续失败次数，≥2 次自动跳过并通知 Qwen 继续后续流程
- 避免之前 chemprop 失败 7 次导致超时的问题
- 所有容器 NVIDIA_VISIBLE_DEVICES=1，容器内永远用 device=0
- JAX容器（AF3、BindCraft）不要挂载宿主机 cuda lib64
- data 验证 + addModel try/catch + 友好错误提示
- **Bug 3 — RAG 残基编号匹配失败**：
- DT3 阈值：**adaptive = max(top 20% score, 0.10)**，不要用固定 0.7
- CD36 教训：5 个分散热点（A164-A400 跨 236 残基）→ ipTM=0.43 全部失败
- pipeline 必须指定 `chains="A"` 过滤，否则 DiscoTope3 报 "No valid PDB"

## ADC 注意事项
- Step 3: AF3 验证 — top5 MPNN → AF3 复合物 → ipTM 分级（≥0.75 high / ≥0.6 low_confidence）
- 每步 try/except，单步失败标记 `partial: true` 不阻塞后续
- AF3 任务间隔 5 秒避免 GPU OOM，timeout 1800s
- Step 1-3: RFdiffusion → ProteinMPNN → AF3 验证（ipTM ≥0.75 high / ≥0.6 low_confidence）
- 每步 try/except，单步失败标记 partial，不阻塞后续

## ALPHAFOLD3 注意事项
- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）
- **根因**：v3 任务 5 个 AF3 设计全部失败。rank1 超时 1800s（但实际已跑完，ipTM=0.48），rank2-5 被路由到 DEGRADED 队列 OOM exit 1
- **修复 1**：新增 `_wait_for_af3_task()` 无限等待函数，每 30s poll，仅在 OOM/exit1/cancelled/连续10次同错 时判定失败
- 不再降级到 DEGRADED 导致 OOM crash
- **原因**：CIF→PDB 转换失败时 `af3_pdb=None` → `"No PDB for FreeSASA"` 错误
- AF3 验证时按结构域截取抗原序列，避免全长序列降低ipTM精度
- `num_seeds=3` 用于 binder_design_pipeline AF3 验证（速度/准确性平衡）

## AUTODOCK 注意事项
- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）

## BINDCRAFT 注意事项
- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）

## CHEMPROP 注意事项
- 1. **Dockerfile 必须有**：`RUN ln -sf /usr/bin/python3.11 /usr/bin/python3`（容器内 python3 默认指向 3.10，但所有包装在 3.11 路径下）
- 2. **predict 调用必须加**：`--accelerator cpu`（小批量走 GPU 会 OOM），归属 `_CPU_TOOLS` 队列
- 3. **`--devices 1` = "使用 1 个 GPU 设备"**，不是 device index=1。train 用 `--accelerator gpu --devices 1`，predict 用 `--accelerator cpu`
- **根因**：容器内 `python3` symlink 指向 3.10，但所有包（torch/numpy/chemprop）装在 python3.11 路径下 → `No module named 'numpy'`
- **修复**：`docker exec oih-chemprop bash -c "rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3"`（容器重建会丢失，需写入 Dockerfile）
- **router 修复**：`routers/ml_tools.py` 加 `--accelerator cpu`，避免 GPU OOM；`task_manager.py` 把 `chemprop_predict` 加入 `_CPU_TOOLS`
- **验证**：3 分子预测 completed，CPU queue，5 秒完成

## DIFFDOCK 注意事项
- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）

## ESM 注意事项
- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）

## FPOCKET 注意事项
- **旧流程**（11步）：P2Rank top pocket → 直接取 top 6 残基 → DiffDock 盲对接交叉验证 → RFdiffusion
- 10. AF3 验证 — ipTM ≥ 0.6

## GNINA 注意事项
- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）

## GROMACS 注意事项
- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）
- 1. **tc-grps 必须用 `Protein_LIG Water_and_ions`**（不是默认的 `Protein Non-Protein`），否则 NVT grompp 报 "group not found"
- 2. **每步 mdrun 之后必须验证输出文件存在**：em.gro → nvt.gro → npt.gro → md.xtc，任何一步缺失立刻 raise 并附 .log 最后20行
- 3. **`gmx make_ndx` 创建 Protein_LIG 组**：在 genion 之后、EM 之前运行 `echo '1 | 13
q' | gmx make_ndx` 合并 Protein 和 LIG 组
- 4. **不要静默忽略 mdrun 返回值**：`retcode != 0 and "WARNING" not in stderr` 这种判断不安全，WARNING 可能掩盖真实错误
- 1. **tc-grps 动态检测**：`make_ndx` 后解析 `index.ndx` 找实际合并组名（如 `Protein_UNL`），不再硬编码 `Protein_LIG`
- 2. **MDP 延迟生成**：NVT/NPT/MD 的 MDP 在 `_run()` 内 `make_ndx` 之后才写入，确保 tc-grps 正确
- 3. **每步文件检查**：em.gro → nvt.gro → npt.gro → md.xtc，缺失立即 raise + 附 .log 最后 20 行

## PROTEINMPNN 注意事项
- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）

## RFDIFFUSION 注意事项
- 容器内 NVIDIA_VISIBLE_DEVICES=1，永远用 device=0 / gpu_id=0（不要用1）
- **timeout 必须 7200s**（不是 3600s）
- **num_designs 用 10**（不要 20，太慢）
- **hotspot 必须空间聚集**（<= 15Å centroid），分散的会极慢
# === /工具注意事项 ===

## 自主诊断与修复能力

当任何工具返回错误或失败时，你必须：

1. **主动诊断**：调用以下工具查明原因：
   - bash_exec 查看日志：grep -i error /tmp/fastapi.log | tail -20
   - bash_exec 检查 GPU：nvidia-smi --query-gpu=memory.used,memory.free --format=csv
   - bash_exec 检查输出文件是否存在

2. **自动修复**：根据错误类型调整参数：
   - AF3 超时 → 检查 CIF 文件是否存在，读取真实 ipTM
   - ipTM 低 → 换 hotspot 残基或增加 num_designs
   - OOM → 等待 GPU 空闲后重试
   - 文件缺失 → 重跑上一步

3. **闭环验证**：修复后自动重跑，不等用户确认

4. **报告结果**：完成后用中文总结：
   - 遇到了什么问题
   - 如何诊断和修复
   - 最终结果（ipTM/结合能/DAR等关键指标）
"""
