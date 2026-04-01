
import os

# LLM Backend Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")  # local | anthropic | openai
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")  # leave empty for provider default
