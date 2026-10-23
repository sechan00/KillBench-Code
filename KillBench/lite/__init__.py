"""KillBench/lite: lightweight Kill Switch benchmark that reuses

- Lite scenarios (PII / NEWS / HACK websites in `AutoGuard_ver2/data/website/`)
- Lite attack prompts (`AutoGuard_ver2/data/data.json`)
- Lite defense prompts (`AutoGuard_ver2/results/prompt_library_*.json`)

but swaps the agent stack from test.py's LangChain `create_agent` +
`bind_tools(required)` to KillBench's agentlab `GenericAgent` +
BrowserGym + Playwright pipeline.

Motivation: forced-tool-call (`tool_choice='required'`) collides with the
intrinsic alignment of reasoning models like Qwen3.6 — the model produces
content=='' and tool_calls=[] (empty turn). Heavy's GenericAgent uses
ReAct-style prompting (Thought/Action format) instead, sidestepping the
conflict and giving us `<think>` blocks for CSR extraction as a side
benefit.
"""
