"""Harbor's Claude Code agent with an install step that works in emulated (amd64-on-arm64) task containers.

The stock install runs Claude Code's bootstrap installer, which segfaults under QEMU user emulation. The native
binary itself runs fine, so this copies it straight from the platform npm package.
"""

from harbor.agents.installed.claude_code import ClaudeCode

CLAUDE_CODE_VERSION = '2.1.325'


class LocalClaudeCode(ClaudeCode):
    async def install(self, environment) -> None:
        if await self._installed_claude_satisfies_version(environment):
            return
        await self.ensure_system_dependencies(environment, ('curl', 'bash', 'procps'))
        await self.exec_as_agent(environment, command=(
            'set -euo pipefail; '
            'case "$(uname -m)" in x86_64) p=linux-x64;; aarch64|arm64) p=linux-arm64;; '
            '*) echo "unsupported arch"; exit 1;; esac; '
            '[ -f /etc/alpine-release ] && p="$p-musl"; '
            f'v={CLAUDE_CODE_VERSION}; d=$(mktemp -d); '
            'curl -fsSL "https://registry.npmjs.org/@anthropic-ai/claude-code-$p/-/claude-code-$p-$v.tgz" | tar xz -C "$d"; '
            'mkdir -p "$HOME/.local/bin"; install -m 755 "$d/package/claude" "$HOME/.local/bin/claude"; rm -rf "$d"; '
            'echo \'export PATH="$HOME/.local/bin:$PATH"\' >> ~/.bashrc; '
            'export PATH="$HOME/.local/bin:$PATH"; claude --version'))
