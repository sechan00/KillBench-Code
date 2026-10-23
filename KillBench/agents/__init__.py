"""Agent variants for Kill Switch evaluation.

Three agents are supported:
- "default":    plain Claude via agentlab GenericAgent (anthropic_chat_adapter)
- "hardened":   same, but with adversarial system-prompt extra_instructions
                injected via GenericPromptFlags.extra_instructions
- "browseruse": browser-use Agent run in its own Playwright session
"""
