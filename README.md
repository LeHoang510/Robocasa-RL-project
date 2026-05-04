# Project Setup

## Installation

```bash
# 1. Create venv and install project dependencies
uv sync

# 2. Install robosuite and robocasa as editable installs
#    --config-settings editable_mode=compat is required to avoid a namespace
#    package conflict: running from the project root puts '' (CWD) on sys.path,
#    which causes Python to find the bare robocasa/ and robosuite/ directories
#    as namespace packages before the editable install can intercept. The compat
#    mode writes the source path directly into a .pth file instead of using a
#    meta path finder, so PathFinder resolves the real package correctly.
uv pip install -e ./robosuite --config-settings editable_mode=compat
uv pip install -e ./robocasa --config-settings editable_mode=compat
```

> If you ever reinstall either package without `--config-settings editable_mode=compat`, the namespace package conflict will reappear.
