"""conftest."""
import subprocess
import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_ccsniff_available():
    try:
        subprocess.run(["npx", "--yes", "ccsniff@latest", "--list-sessions"],
                       capture_output=True, text=True, timeout=60, check=False)
    except Exception:
        pass
