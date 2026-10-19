"""Record what produced a result: code version, environment and run settings. Benchmark scripts
store this next to their numbers so every saved figure can be traced to a commit."""
import datetime
import os
import platform
import subprocess
import sys
from importlib import metadata

PACKAGES = ["palm-bulk", "numpy", "scipy", "pandas", "jax", "jaxlib", "numpyro", "anndata", "decoupler", "torch"]


def _git(*args):
    root = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def provenance(**settings):
    """Dict with git commit (and whether the working tree had uncommitted changes), package
    versions, Python, OS, JAX backend and precision, time, command line, plus any `settings`."""
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    try:
        import jax
        backend = dict(platform=jax.default_backend(),
                       precision="float64" if jax.config.jax_enable_x64 else "float32")
    except Exception:
        backend = {}
    return dict(git_commit=commit or None, git_dirty=bool(status) if status is not None else None,
                versions=versions, python=sys.version.split()[0], os=platform.platform(), **backend,
                time=datetime.datetime.now().isoformat(timespec="seconds"), argv=sys.argv, settings=settings)
