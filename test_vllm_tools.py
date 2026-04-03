import json, requests, sys
sys.path.insert(0, '/data/oih/oih-api')
from tool_definitions.qwen_tools import ALL_TOOLS, QWEN_SYSTEM_PROMPT

VLLM_URL = "http://localhost:8002/v1/chat/completions"

def call_qwen(user_msg, test_name=""):
    print(f"\n{'='*50}\nTest: {test_name}\n{'='*50}")
    resp = requests.post(VLLM_URL, json={
        "model": "Qwen3-14B",
        "messages": [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        "tools": ALL_TOOLS,
        "tool_choice": "auto",
        "max_tokens": 2048,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    })
    result = resp.json()
    choice = result["choices"][0]
    message = choice["message"]
    print(f"finish_reason: {choice['finish_reason']}")

    if message.get("tool_calls"):
        print(f"✅ Triggered {len(message['tool_calls'])} tool(s):")
        for tc in message["tool_calls"]:
            try:
                args = json.loads(tc["function"]["arguments"])
                print(f"  🔧 {tc['function']['name']}")
                print(f"     {json.dumps(args, ensure_ascii=False, indent=4)}")
            except Exception as e:
                print(f"  🔧 {tc['function']['name']} argument parse failed: {e}")
                print(f"     raw: {tc['function']['arguments']}")
    else:
        content = message.get("content") or message.get("reasoning_content") or str(message)
        print(f"❌ No tool_calls\n   {content[:300]}")

call_qwen("Detect pockets on /data/oih/inputs/egfr.pdb", "single-tool-fpocket")
call_qwen("Dock aspirin CC(=O)Oc1ccccc1C(=O)O to /data/oih/inputs/cox2.pdb with GNINA", "single-tool-docking")
call_qwen("Check status of task task_abc123", "single-tool-poll")
call_qwen("I have EGFR protein sequence MRPSGTAGAALLALLAALCPAS, do a complete drug discovery analysis with aspirin", "multi-tool-pipeline")
