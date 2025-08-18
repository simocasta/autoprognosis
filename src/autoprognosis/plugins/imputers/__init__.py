# stdlib
import glob
from os.path import basename, dirname, isfile, join

# third party
try:  # pragma: no cover - optional dependency
    import random
    import numpy as np
    import hyperimpute.utils.distributions as _hi_distr

    def _safe_enable_reproducible_results(seed: int = 0) -> None:
        """Patch hyperimpute's seeding to avoid CUDA initialisation errors.

        The original implementation calls ``torch.manual_seed`` which may
        trigger a CUDA context initialisation even in environments without a
        proper GPU setup.  This can lead to crashes such as
        ``torch.AcceleratorError: CUDA error: device-side assert triggered``
        during plugin construction.  To keep the behaviour deterministic while
        remaining robust, we seed Python's ``random`` and ``numpy`` modules and
        attempt to seed ``torch`` only if it is available.  Any error raised by
        ``torch`` (for example when no GPU is present) is silently ignored.
        """

        random.seed(seed)
        np.random.seed(seed)
        try:  # pragma: no cover - torch may be missing
            import torch

            try:
                torch.manual_seed(seed)
            except Exception:
                # If torch fails (e.g. due to a broken GPU setup), proceed
                # without seeding to avoid crashing the pipeline.
                pass
        except Exception:
            pass

    _hi_distr.enable_reproducible_results = _safe_enable_reproducible_results
except Exception:  # pragma: no cover - hyperimpute not installed
    pass

# autoprognosis absolute
from autoprognosis.plugins.core.base_plugin import PluginLoader

# autoprognosis relative
from .base import ImputerPlugin  # noqa: F401,E402

plugins = glob.glob(join(dirname(__file__), "plugin*.py"))


class Imputers(PluginLoader):
    def __init__(self) -> None:
        super().__init__(plugins, ImputerPlugin)


__all__ = [basename(f)[:-3] for f in plugins if isfile(f)] + [
    "Imputers",
    "ImputerPlugin",
]
