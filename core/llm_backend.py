"""
LLM Backend Abstraction Layer
Supports local (vLLM) and remote (Anthropic, OpenAI) inference endpoints.
The orchestration layer (skills, tools, RAG) is LLM-agnostic.
"""

import os
import json
import logging
import httpx
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ── Unified response object ──────────────────────────────────────────────────

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Unified response from any LLM backend."""
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Dict = field(default_factory=dict)
    usage: Dict = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ── Base class ───────────────────────────────────────────────────────────────

class LLMBackend(ABC):
    """Abstract base class for LLM inference backends."""

    @abstractmethod
    async def chat(self, messages: List[Dict], tools: Optional[List] = None,
                   system_prompt: Optional[str] = None,
                   temperature: float = 0.7, max_tokens: int = 4096) -> LLMResponse:
        """Send messages and return a unified LLMResponse."""
        pass

    def build_history_message(self, response: LLMResponse) -> Dict:
        """Convert LLMResponse to a conversation history entry."""
        msg: Dict[str, Any] = {"role": "assistant"}
        if response.content:
            msg["content"] = response.content
        else:
            msg["content"] = ""
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in response.tool_calls
            ]
        return msg

    def build_tool_result_message(self, tool_call_id: str, name: str, content: str) -> Dict:
        """Build a tool result message for conversation history."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        }

    # Legacy helper (kept for backward compat)
    def format_tool_calls(self, response: Dict) -> List[Dict]:
        return []


# ── Helpers ──────────────────────────────────────────────────────────────────

def _prepend_system(messages: List[Dict], system_prompt: Optional[str]) -> List[Dict]:
    """Prepend system_prompt as a system message if provided."""
    if not system_prompt:
        return messages
    # Replace existing system message or prepend
    if messages and messages[0].get("role") == "system":
        return [{"role": "system", "content": system_prompt}] + messages[1:]
    return [{"role": "system", "content": system_prompt}] + messages


def _parse_openai_response(data: Dict) -> LLMResponse:
    """Parse OpenAI-format response (used by vLLM and OpenAI)."""
    choices = data.get("choices", [])
    if not choices:
        return LLMResponse(raw=data, usage=data.get("usage", {}))

    msg = choices[0].get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or None

    tool_calls = []
    for tc in msg.get("tool_calls", []):
        args_str = tc["function"]["arguments"]
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            args = {"_raw": args_str}
        tool_calls.append(ToolCall(
            id=tc.get("id", ""),
            name=tc["function"]["name"],
            arguments=args,
        ))

    return LLMResponse(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        raw=data,
        usage=data.get("usage", {}),
    )


# ── Backend implementations ──────────────────────────────────────────────────

class LocalVLLM(LLMBackend):
    """Local vLLM backend (Qwen3-14B-AWQ or compatible)."""

    def __init__(self, base_url: str = "http://localhost:8002/v1",
                 model: str = "Qwen/Qwen3-14B-AWQ"):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=300)

    async def chat(self, messages, tools=None, system_prompt=None,
                   temperature=0.7, max_tokens=4096):
        messages = _prepend_system(messages, system_prompt)
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
        return _parse_openai_response(resp.json())


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.client = httpx.AsyncClient(timeout=300)

    async def chat(self, messages, tools=None, system_prompt=None,
                   temperature=0.7, max_tokens=4096):
        # Extract system from messages or use system_prompt
        system_msg = system_prompt or ""
        anthro_messages = []
        for m in messages:
            if m["role"] == "system":
                if not system_msg:
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
        data = resp.json()

        # Parse Anthropic response format
        content = ""
        tool_calls = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block["input"],
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            raw=data,
            usage=data.get("usage", {}),
        )

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

    def build_tool_result_message(self, tool_call_id: str, name: str, content: str) -> Dict:
        """Anthropic uses a different tool_result format."""
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }
            ],
        }


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible API backend (GPT-4, DeepSeek, etc)."""

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=300)

    async def chat(self, messages, tools=None, system_prompt=None,
                   temperature=0.7, max_tokens=4096):
        messages = _prepend_system(messages, system_prompt)
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
        return _parse_openai_response(resp.json())


# ── Factory ──────────────────────────────────────────────────────────────────

def get_backend(provider: str = "local", **kwargs) -> LLMBackend:
    """Factory function to create LLM backend.

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

    # Translate universal kwargs -> per-backend signature
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
    instance.provider = provider
    return instance
