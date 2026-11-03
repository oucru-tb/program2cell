"""Deprecated alias: the package was renamed to program2cell (2026-11-03). `import palm_bulk` and
`import palm_bulk.core.unmix` etc. return the program2cell modules themselves (not copies), so old scripts and
notebooks keep working. Use `import program2cell`."""
import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import program2cell as _new

_OLD, _NEW = "palm_bulk", "program2cell"


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.startswith(_OLD + "."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(_NEW + spec.name[len(_OLD):])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _Alias) for f in sys.meta_path):
    sys.meta_path.insert(0, _Alias())
warnings.warn("palm_bulk was renamed to program2cell; use `import program2cell`", DeprecationWarning, stacklevel=2)
sys.modules[_OLD] = _new
