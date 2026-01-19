"""
SPEEDB04T Project Handler

Git workflows and shell command execution.
"""

import subprocess
import os
from typing import Dict, Any, List

# =============================================================================
# UTILITIES
# =============================================================================

def run_command(cmd: str) -> dict:
    """Execute a shell command."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stderr": "Timeout after 5 minutes", "returncode": -1}
    except Exception as e:
        return {"success": False, "stderr": str(e), "returncode": -1}


def apply_file_changes(files: List[dict], base_path: str = "/workspace/project") -> List[dict]:
    """Apply file changes to workspace."""
    results = []
    for file in files:
        path = os.path.join(base_path, file.get("path", ""))
        action = file.get("action", "create")
        content = file.get("content", "")
        
        try:
            if action == "delete":
                if os.path.exists(path):
                    os.remove(path)
                results.append({"path": path, "action": "deleted", "success": True})
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(content)
                results.append({"path": path, "action": action, "success": True})
        except Exception as e:
            results.append({"path": path, "action": action, "success": False, "error": str(e)})
    
    return results

# =============================================================================
# HANDLER
# =============================================================================

async def handle(input_data: dict) -> dict:
    """
    SPEEDB04T project handler.
    
    Actions:
    - command: Execute shell command
    - git_workflow: Clone → edit → commit → push
    """
    action = input_data.get("action", "command")
    
    # Simple command execution
    if action == "command":
        cmd = input_data.get("command", "echo 'No command provided'")
        return run_command(cmd)
    
    # Git workflow
    if action == "git_workflow":
        workflow = input_data.get("git_workflow", {})
        repo_url = workflow.get("repo_url", "")
        branch = workflow.get("branch", "main")
        files = workflow.get("files", [])
        commit_message = workflow.get("commit_message", "Self-development update")
        
        results = []
        
        # Clone
        clone_result = run_command(f"cd /workspace && rm -rf project && git clone -b {branch} {repo_url} project")
        results.append({"step": "clone", **clone_result})
        
        if not clone_result["success"]:
            clone_result = run_command(f"cd /workspace && git clone {repo_url} project && cd project && git checkout -B {branch}")
            results.append({"step": "clone_fallback", **clone_result})
        
        # Apply files
        if files:
            file_results = apply_file_changes(files)
            results.append({"step": "apply_files", "files": file_results})
        
        # Commit and push
        commit_result = run_command(f'cd /workspace/project && git add -A && git commit -m "{commit_message}"')
        results.append({"step": "commit", **commit_result})
        
        push_result = run_command(f"cd /workspace/project && git push origin {branch}")
        results.append({"step": "push", **push_result})
        
        return {
            "success": push_result["success"],
            "steps": results,
            "message": "Git workflow completed" if push_result["success"] else "Git workflow failed"
        }
    
    return {"error": f"Unknown action: {action}"}
