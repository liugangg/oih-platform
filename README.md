# OIH — Open Intelligence Hub

An LLM-agent platform for end-to-end computational antibody–drug conjugate (ADC) design.

## Overview

OIH orchestrates 32 computational biology tools across 15 Docker containers using a large language model agent. Given a target protein name, the platform autonomously executes the complete binder design and ADC assembly pipeline.

## Architecture

- **LLM Agent**: LLM-agnostic (supports local Qwen3-14B, Anthropic Claude, OpenAI GPT-4)
- **Tool Orchestration**: Multi-round function calling with dynamic skills injection
- **Tier Routing**: Structure-guided (Tier 1) and prediction-guided (Tier 2) hotspot selection
- **Validation**: AlphaFold 3 + ipSAE quality control
- **ADC Assembly**: Automated conjugation site selection and linker chemistry

## Quick Start
```bash
# Start vLLM (GPU 0)
docker compose -f docker-compose.vllm.yml up -d

# Start bio containers (GPU 1)  
docker compose up -d

# Start FastAPI backend
python -m uvicorn main:app --host 0.0.0.0 --port 8080
```

## Citation

If you use OIH in your research, please cite:
> [Paper reference to be added upon publication]

## License

MIT License
