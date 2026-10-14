"""Folding engines behind one protocol. `cross_check: auto` is resolved here."""

from __future__ import annotations

from chlide.engines.af3 import AF3Folder
from chlide.engines.base import EngineError, Folder, FoldResult, LigandSpec
from chlide.engines.boltz import BoltzFolder

__all__ = ['AF3Folder', 'BoltzFolder', 'Folder', 'FoldResult', 'LigandSpec',
           'EngineError', 'get_folder', 'other_engine']

_REGISTRY = {'af3': AF3Folder, 'boltz': BoltzFolder}


def get_folder(name: str, **kw):
    """Instantiate an engine by the name a campaign YAML uses."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise EngineError(
            f'unknown engine {name!r}; known: {", ".join(sorted(_REGISTRY))}') from None
    return cls(**kw) if kw else cls()


def other_engine(name: str) -> str:
    """What `cross_check: auto` means: whichever engine did NOT lead.

    A campaign cannot grade itself. On v8's top 25 this is what caught the joint-best design
    by BUNs -- AF3 rated its ligand 0.951, Boltz 0.506.
    """
    if name not in _REGISTRY:
        raise EngineError(f'unknown engine {name!r}')
    return 'boltz' if name == 'af3' else 'af3'
