"""Pytest configuration -- fail loud when required extras are absent.

New contributors should see a clear message, not 15 cryptic failures
that look like flakiness. This hook runs before collection and prints
actionable guidance when test dependencies are missing.
"""

import importlib


def pytest_configure(config):
    """Check for required test dependencies and warn clearly if absent."""
    missing = []

    # Core test dependency
    _check("pytest_asyncio", "pytest-asyncio", missing)

    # Embeddings extra (optional for runtime, required for full test suite)
    if not _can_import("numpy"):
        missing.append("embeddings extra: pip install 'throttle-pro[embeddings]'")
    elif not _can_import("onnxruntime"):
        missing.append("onnxruntime: pip install 'onnxruntime>=1.16.0'")

    # Config extra
    if not _can_import("yaml"):
        missing.append("config extra: pip install 'throttle-pro[config]'")

    if missing:
        border = "=" * 60
        print("\n" + border)
        print("WARNING: MISSING TEST DEPENDENCIES")
        print(border)
        for m in missing:
            print(f"  [MISSING] {m}")
        print()
        print("  Quick fix (all extras):")
        print("    pip install -e .[embeddings,config]")
        print("    pip install pytest-asyncio")
        print()
        print("  Tests requiring these will be SKIPPED, not failed.")
        print(border + "\n")


def _can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _check(module_name, pip_name, missing):
    if not _can_import(module_name):
        missing.append(f"{pip_name}: pip install {pip_name}")
