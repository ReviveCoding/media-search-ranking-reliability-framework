from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> None:
    command = [sys.executable, "-m", "pip", *args]
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    root = Path.cwd()
    run("install", "--upgrade", "pip")
    requirements = root / "requirements.txt"
    if requirements.is_file():
        run("install", "-r", str(requirements))

    install_editable = (root / "setup.py").is_file() or (root / "setup.cfg").is_file()
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        install_editable = install_editable or bool(
            re.search(r"(?m)^\[project\]\s*$", text)
        )

    if install_editable:
        run("install", "-e", ".")
    elif not requirements.is_file():
        run("install", "pytest")


if __name__ == "__main__":
    main()
