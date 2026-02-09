"""
Governor WebUI: Presentation layer for Agent Governor.

Serves a combined chat + governor panel at the root URL, and exposes an
OpenAI-compatible API for external clients. Underneath, it:
1. Routes requests through ChatBridge to Anthropic, Ollama, or Claude Code CLI
2. Applies governor hooks based on context mode (fiction, code, nonfiction)
3. Maintains isolated governor contexts per user/project
"""

__version__ = "0.3.0"
