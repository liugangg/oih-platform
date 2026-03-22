import json, requests, sys
sys.path.insert(0, '/data/oih/oih-api')
from tool_definitions.qwen_tools import ALL_TOOLS, QWEN_SYSTEM_PROMPT

VLLM_URL = "http://localhost:8002/v1/chat/completions"

def call_qwen(user_msg, test_name=""):
    print(f"\n{'='*50}\n测试: {test_name}\n{'='*50}")
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
        print(f"✅ 触发 {len(message['tool_calls'])} 个工具:")
        for tc in message["tool_calls"]:
            try:
                args = json.loads(tc["function"]["arguments"])
                print(f"  🔧 {tc['function']['name']}")
                print(f"     {json.dumps(args, ensure_ascii=False, indent=4)}")
            except Exception as e:
                print(f"  🔧 {tc['function']['name']} 参数解析失败: {e}")
                print(f"     raw: {tc['function']['arguments']}")
    else:
        content = message.get("content") or message.get("reasoning_content") or str(message)
        print(f"❌ 无tool_calls\n   {content[:300]}")

call_qwen("对 /data/oih/inputs/egfr.pdb 进行口袋检测", "单工具-fpocket")
call_qwen("用GNINA把阿司匹林CC(=O)Oc1ccccc1C(=O)O对接到/data/oih/inputs/cox2.pdb", "单工具-对接")
call_qwen("查询任务 task_abc123 的状态", "单工具-poll")
call_qwen("我有EGFR蛋白序列MRPSGTAGAALLALLAALCPAS，用阿司匹林做完整药物发现分析", "多工具-pipeline")
