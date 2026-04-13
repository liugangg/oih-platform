"""
LLM Backend Abstraction Layer
Supports local (vLLM) and remote (Anthropic, OpenAI) inference endpoints.
The orchestration layer (skills, tools, RAG) is LLM-agnostic.
"""

import os
import json
import httpx
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Abstract base class for LLM inference backends."""
    
    @abstractmethod
    async def chat(self, messages: List[Dict], tools: Optional[List] = None, 
                   temperature: float = 0.7, max_tokens: int = 4096) -> Dict:
        """Send messages and return response with optional tool calls."""
        pass
    
    @abstractmethod
    def format_tool_calls(self, response: Dict) -> List[Dict]:
        """Extract tool calls from response in unified format."""
        pass


class LocalVLLM(LLMBackend):
    """Local vLLM backend (Qwen3-14B-AWQ or compatible)."""
    
    def __init__(self, base_url: str = "http://localhost:8002/v1",
                 model: str = "Qwen/Qwen3-14B-AWQ"):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=300)
    
    async def chat(self, messages, tools=None, temperature=0.7, max_tokens=4096):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        resp = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()
    
    def format_tool_calls(self, response):
        choices = response.get("choices", [])
        if not choices:
            return []
        msg = choices[0].get("message", {})
        tool_calls = msg.get("tool_calls", [])
        return [{
            "id": tc.get("id"),
            "name": tc["function"]["name"],
            "arguments": json.loads(tc["function"]["arguments"]),
        } for tc in tool_calls]


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""
    
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.client = httpx.AsyncClient(timeout=300)
    
    async def chat(self, messages, tools=None, temperature=0.7, max_tokens=4096):
        # Convert OpenAI format messages to Anthropic format
        system_msg = ""
        anthro_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                anthro_messages.append({"role": m["role"], "content": m["content"]})
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": anthro_messages,
        }
        if system_msg:
            payload["system"] = system_msg
        if tools:
            payload["tools"] = self._convert_tools(tools)
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        resp = await self.client.post(
            "https://api.anthropic.com/v1/messages", 
            json=payload, headers=headers
        )
        resp.raise_for_status()
        return resp.json()
    
    def _convert_tools(self, openai_tools):
        """Convert OpenAI function-calling format to Anthropic tool_use format."""
        anthro_tools = []
        for t in openai_tools:
            if t.get("type") == "function":
                func = t["function"]
                anthro_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
        return anthro_tools
    
    def format_tool_calls(self, response):
        tool_calls = []
        for block in response.get("content", []):
            if block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": block["input"],
                })
        return tool_calls


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible API backend (GPT-4, DeepSeek, etc)."""
    
    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=300)
    
    async def chat(self, messages, tools=None, temperature=0.7, max_tokens=4096):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload, headers=headers
        )
        resp.raise_for_status()
        return resp.json()
    
    def format_tool_calls(self, response):
        choices = response.get("choices", [])
        if not choices:
            return []
        msg = choices[0].get("message", {})
        tool_calls = msg.get("tool_calls", [])
        return [{
            "id": tc.get("id"),
            "name": tc["function"]["name"],
            "arguments": json.loads(tc["function"]["arguments"]),
        } for tc in tool_calls]


def get_backend(provider: str = "local", **kwargs) -> LLMBackend:
    """Factory function to create LLM backend.

    Accepts a *universal* kwargs set from callers (provider, api_key, model,
    base_url, vllm_url, vllm_model) and translates them to each backend's
    actual __init__ signature, dropping unknown / None values.

    Args:
        provider: "local" | "anthropic" | "openai"
        **kwargs: any of api_key, model, base_url, vllm_url, vllm_model

    Returns:
        LLMBackend instance with `.provider` attribute set.
    """
    backends = {
        "local": LocalVLLM,
        "anthropic": AnthropicBackend,
        "openai": OpenAIBackend,
    }
    if provider not in backends:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(backends.keys())}")

    # Translate universal kwargs → per-backend signature
    if provider == "local":
        instance_kwargs = {
            "base_url": kwargs.get("vllm_url") or kwargs.get("base_url"),
            "model":    kwargs.get("vllm_model") or kwargs.get("model"),
        }
    elif provider == "anthropic":
        instance_kwargs = {
            "api_key": kwargs.get("api_key"),
            "model":   kwargs.get("model"),
        }
    elif provider == "openai":
        instance_kwargs = {
            "api_key":  kwargs.get("api_key"),
            "base_url": kwargs.get("base_url"),
            "model":    kwargs.get("model"),
        }

    # Drop None so backend defaults apply
    instance_kwargs = {k: v for k, v in instance_kwargs.items() if v is not None}

    instance = backends[provider](**instance_kwargs)
    instance.provider = provider  # qwen_agent expects backend.provider
    return instance
