"""chlide -- command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from chlide.config import ConfigError, load_campaign


def _validate(paths: list[Path]) -> int:
    bad = 0
    for p in paths:
        try:
            cfg = load_campaign(p)
        except (ConfigError, ValidationError) as e:
            bad += 1
            print(f'FAIL  {p}')
            for line in str(e).splitlines():
                print(f'      {line}')
        else:
            print(f'ok    {cfg.summary()}')
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='chlide')
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('validate', help='parse and check campaign YAMLs')
    v.add_argument('paths', nargs='+', type=Path)
    a = ap.parse_args(argv)
    if a.cmd == 'validate':
        return _validate(a.paths)
    ap.error(f'unknown command {a.cmd}')
    return 2


if __name__ == '__main__':
    sys.exit(main())
