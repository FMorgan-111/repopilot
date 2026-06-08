import asyncio, json, sys, os
sys.path.insert(0, '.')
os.environ.setdefault('LLM_MODEL', 'deepseek-v4-flash')
os.environ['REPOPILOT_DISABLE_PARALLEL'] = '1'
os.environ['REPOPILOT_DISABLE_CACHE'] = '1'
from src.new_agent import agent_v2

async def main():
    r = await agent_v2(
        'https://github.com/theskumar/python-dotenv/issues/657',
        max_retries=1, token_budget=20000
    )
    with open('examples/traces/case_1.json', 'w') as f:
        json.dump(r, f, indent=2, default=str)
    phase = r.get('final_phase', '?')
    files = len(r.get('relevant_files', []))
    err = r.get('error', 'none')
    print(f'DONE: phase={phase} files={files} error={err}')

asyncio.run(main())
