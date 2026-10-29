"""The composite Action installs its own code and passes untrusted inputs as data."""

import subprocess
from pathlib import Path

import yaml


def test_action_shell_syntax_and_reproducible_source():
    action = yaml.safe_load((Path(__file__).parents[1] / "action.yml").read_text())
    steps = action["runs"]["steps"]
    for step in steps:
        if step.get("shell") == "bash":
            subprocess.run(["bash", "-n"], input=step["run"], text=True, check=True)
            assert "${{" not in step["run"]
    install = next(s for s in steps if "ACTION_SOURCE" in s.get("env", {}))
    assert 'pip install --quiet "$ACTION_SOURCE"' in install["run"]
    scan = steps[-1]
    assert "${{ github.event.pull_request.base.sha }}" == scan["env"]["BASE_SHA"]
    assert 'promptscan "${args[@]}" -- "$PATH_INPUT"' in scan["run"]
    assert "--changed FETCH_HEAD" in scan["run"]
    assert action["inputs"]["fail-on"]["default"] == "injection"
