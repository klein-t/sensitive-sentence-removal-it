"""Build an isolated Codex CLI invocation for one annotation packet."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "apps",
    "plugins",
    "remote_plugin",
    "memories",
    "multi_agent",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "view_image",
    "image_generation",
    "hooks",
    "goals",
    "skill_search",
    "tool_suggest",
    "workspace_dependencies",
    "shell_snapshot",
    "unbounded_connection_retries",
)


def build_codex_argv(
    packet_dir: Path, output_path: Path, schema_path: Path | None = None
) -> list[str]:
    """Return argv for a fresh Sol session; the caller supplies the prompt on stdin.

    ``packet_dir`` must contain only the assigned worker's material. The CLI
    writes its final response to ``output_path`` outside that readable root.
    The caller must keep JSON events and stderr private because they can echo
    the prompt. A separate ``codex sandbox`` canary tests the same profile.
    """
    packet_dir = packet_dir.resolve(strict=True)
    output_path = output_path.absolute()
    if not packet_dir.is_dir():
        raise ValueError("packet_dir must be a directory")
    if output_path.resolve(strict=False).is_relative_to(packet_dir):
        raise ValueError("output_path must be outside the worker root")
    if schema_path is not None:
        schema_path = schema_path.resolve(strict=True)
        if not schema_path.is_file() or not schema_path.is_relative_to(packet_dir):
            raise ValueError("schema_path must be a file inside the worker root")

    argv = [
        "codex",
        "--no-daemon",
        "-a",
        "never",
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--strict-config",
        "--skip-git-repo-check",
        "--json",
        "-C",
        str(packet_dir),
        "-m",
        "gpt-6-sol",
        "-o",
        str(output_path),
        "-c",
        'default_permissions="annotation"',
        "-c",
        'permissions.annotation.filesystem={":root"="deny",":minimal"="read",":workspace_roots"={"."="read"},":tmpdir"="deny",":slash_tmp"="deny"}',
        "-c",
        "permissions.annotation.network.enabled=false",
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        'history.persistence="none"',
        "-c",
        'web_search="disabled"',
        "-c",
        'model_reasoning_effort="medium"',
    ]
    for feature in _DISABLED_FEATURES:
        argv.extend(("--disable", feature))
    if schema_path is not None:
        argv.extend(("--output-schema", str(schema_path)))
    argv.append("-")
    return argv
