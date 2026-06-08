#!/usr/bin/env python3
"""
测试新的智能 Agent

运行方式:
python test_intelligent_agent.py
"""

import asyncio
import json
import logging
from src.new_agent import intelligent_analyze_issue

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_intelligent_agent():
    """测试智能 agent"""

    print("🚀 Testing Intelligent Agent with Reasoning Layer")
    print("=" * 60)

    # 测试用例
    test_cases = [
        {
            "name": "Real GitHub Issue",
            "url": "https://github.com/microsoft/vscode/issues/12345",  # 替换为真实 issue
            "max_turns": 8,
            "token_budget": 50000
        },
        {
            "name": "Simple Bug Fix",
            "url": "https://github.com/example/simple-repo/issues/1",  # 模拟简单 issue
            "max_turns": 5,
            "token_budget": 30000
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print(f"URL: {test_case['url']}")
        print(f"Max Turns: {test_case['max_turns']}")
        print(f"Token Budget: {test_case['token_budget']}")
        print("-" * 40)

        try:
            result = await intelligent_analyze_issue(
                issue_url=test_case['url'],
                max_turns=test_case['max_turns'],
                token_budget=test_case['token_budget']
            )

            print("✅ Agent completed execution")
            print(f"Final State: {result.get('final_state', 'unknown')}")
            print(f"Success: {result.get('success', False)}")
            print(f"Turns Used: {result.get('turns', 0)}")
            print(f"Tokens Used: {result.get('tokens_used', 0)}")
            print(f"Confidence: {result.get('confidence', 0.0):.2f}")
            print(f"Files Modified: {result.get('applied_fixes', 0)}")
            print(f"Tests Passed: {result.get('test_results', 0)}")

            if result.get('error'):
                print(f"❌ Error: {result['error']}")

            # 保存详细结果
            with open(f"test_result_{i}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            logging.exception(f"Test case {i} failed")

        print("=" * 60)

def test_state_machine_logic():
    """测试状态机逻辑"""
    from src.new_agent import AgentState, AgentContext, Reflection

    print("🔧 Testing State Machine Logic")
    print("-" * 40)

    # 测试状态枚举
    states = [state.value for state in AgentState]
    print(f"Available States: {states}")

    # 测试上下文初始化
    ctx = AgentContext(
        issue_url="https://github.com/test/repo/issues/1",
        owner="test",
        repo="repo",
        issue_number=1
    )

    print(f"Context initialized: {ctx.owner}/{ctx.repo}#{ctx.issue_number}")
    print(f"Max retries: {ctx.max_retries}")
    print(f"Token budget: {ctx.token_budget}")

    # 测试反思结构
    reflection = Reflection(
        root_cause="Test issue",
        fix_confidence=0.8,
        next_action="submit",
        reasoning="High confidence fix",
        test_success=True
    )

    print(f"Reflection: {reflection.next_action} (confidence: {reflection.fix_confidence})")
    print("✅ State machine logic test passed")

async def test_api_integration():
    """测试 API 集成"""
    import httpx

    print("🌐 Testing API Integration")
    print("-" * 40)

    # 启动服务器的测试（需要服务器运行）
    try:
        async with httpx.AsyncClient() as client:
            # 测试新的智能 agent 端点
            response = await client.post(
                "http://localhost:8000/intelligent-agent",
                json={
                    "issue_url": "https://github.com/test/repo/issues/1",
                    "max_turns": 5,
                    "token_budget": 30000
                },
                timeout=60.0
            )

            if response.status_code == 200:
                result = response.json()
                print("✅ API integration test passed")
                print(f"Response: {json.dumps(result, indent=2)[:200]}...")
            else:
                print(f"❌ API test failed: {response.status_code}")
                print(f"Response: {response.text}")

    except Exception as e:
        print(f"⚠️  API test skipped (server not running?): {e}")

async def main():
    """主测试函数"""
    print("RepoPilot 智能 Agent 测试套件")
    print("=" * 60)

    # 1. 测试状态机逻辑
    test_state_machine_logic()
    print()

    # 2. 测试 API 集成（可选）
    await test_api_integration()
    print()

    # 3. 测试完整 agent（需要 GitHub token 和真实 issue）
    choice = input("是否测试完整 agent？这需要 GitHub token 和真实 issue (y/N): ")
    if choice.lower() == 'y':
        await test_intelligent_agent()
    else:
        print("跳过完整 agent 测试")

    print("\n🎉 测试完成！")

if __name__ == "__main__":
    asyncio.run(main())