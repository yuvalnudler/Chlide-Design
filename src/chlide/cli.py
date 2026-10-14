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
        # A leading underscore marks a FRAGMENT, not a campaign: _base.yaml is deep-merged
        # into every campaign and is not a valid config on its own.
        if p.name.startswith('_'):
            print(f'skip  {p.name} (fragment, merged into every campaign)')
            continue
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



def _plan(args) -> int:
    """Show exactly what a campaign WOULD run, touching nothing.

    This is the dry run that has to exist before `chlide run` does: on a shared box with four
    GPUs, the first thing anyone should be able to do with a new campaign is read its plan.
    """
    from chlide.loop import nise_params
    from chlide.stages import summarise, terminal_stages

    cfg = load_campaign(args.campaign)
    run = Path(args.out).resolve()
    params = nise_params(cfg, run)

    print(cfg.summary())
    print(f'  run directory : {run}' + ('' if run.exists() else '   (does not exist yet)'))
    print(f'  devices       : {", ".join(cfg.engine.devices)}')
    print(f'  reserve       : {cfg.compute.reserve_cpus} CPUs left for other users')
    print(f'  per iteration : {cfg.loop.per_iteration:,} designs '
          f'({cfg.loop.per_backbone} x {cfg.loop.top_backbones} backbones)')
    print(f'  total budget  : up to {cfg.loop.iterations * cfg.loop.per_iteration:,} designs')
    c = cfg.loop.convergence
    print(f'  convergence   : {c.metric}, {c.window}-iteration mean, eps {c.eps}, '
          f'patience {c.patience}, min {c.min_iterations}')

    print('\n  key parameters')
    for k in ('ligand_3lc', 'fixed_identity_residue_indices', 'rmsd_use_chirality',
              'burial_mask_alpha_hull_alpha', 'relax_r0', 'coordination_window',
              'num_iterations', 'sequences_sampled_per_backbone'):
        print(f'    {k:38s} {params[k]!r}')

    print('\n  terminal pipeline')
    for i, s in enumerate(terminal_stages(cfg.engine.fold), 1):
        flags = ' '.join(f for f, on in (('GPU', s.needs_gpu), ('optional', s.optional)) if on)
        print(f'    {i:2d}. {s.name:20s} tools/{s.tool}.py{"   [" + flags + "]" if flags else ""}')

    from chlide.config import resolve_input
    CHLIDES = '/home/data/yuvaln/main_pipeline/input/chlides_fixed.mmcif'
    found = {
        'ligand source': resolve_input(run, cfg.ligand.source, 'CHLIDES_MMCIF', CHLIDES),
        'scaffold': resolve_input(run, cfg.scaffold.pdb),
    }
    print('\n  inputs')
    for label, path in found.items():
        where = str(path) if path else 'NOT FOUND'
        print(f'    {label:14s} {where}')
    missing = [k for k, v in found.items() if v is None]
    if missing:
        print(f'\n  NOT READY: {", ".join(missing)}')
        return 1
    print('\n  ready.')
    return 0


def _finish(args) -> int:
    """List the terminal pipeline for a run directory. Execution is not wired yet."""
    from chlide.stages import terminal_stages
    run = Path(args.run_dir).resolve()
    drivers = list(run.glob('run_nise_*.py'))
    eng = 'af3' if any('from af3_adapter import' in d.read_text() for d in drivers) else 'boltz'
    print(f'{run.name}: {eng}-led')
    for i, s in enumerate(terminal_stages(eng), 1):
        print(f'  {i:2d}. {s.name:20s} tools/{s.tool}.py')
    print('\nNOT WIRED: run tools/finish_at_iter.sh for now. The order above is the tested one '
          'and is a faithful transcription of that script.')
    return 0



def _run(args) -> int:
    """Launch a campaign. The run directory is created here and nothing is copied into it."""
    from chlide.loop import run_campaign
    cfg = load_campaign(args.campaign)
    return run_campaign(cfg, args.out, campaign_file=args.campaign,
                        debug=args.debug, allow_foreign_gpu=args.allow_foreign_gpu,
                        dry_run=args.dry_run)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='chlide')
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('validate', help='parse and check campaign YAMLs')
    v.add_argument('paths', nargs='+', type=Path)

    p = sub.add_parser('plan', help='show what a campaign would run, touching nothing')
    p.add_argument('campaign', type=Path)
    p.add_argument('--out', type=Path, required=True, help='run directory (a CLI argument, '
                                                           'never a property of the config)')

    f = sub.add_parser('finish', help='the terminal pipeline for a run directory')
    f.add_argument('run_dir', type=Path)

    r = sub.add_parser('run', help='launch a campaign from a YAML')
    r.add_argument('campaign', type=Path)
    r.add_argument('--out', type=Path, required=True, help='run directory')
    r.add_argument('--debug', action='store_true', help='smoke test: 1 sequence per device')
    r.add_argument('--dry-run', action='store_true', help='resolve everything, launch nothing')
    r.add_argument('--allow-foreign-gpu', action='store_true',
                   help='proceed although another user holds a requested card')

    a = ap.parse_args(argv)
    if a.cmd == 'validate':
        return _validate(a.paths)
    if a.cmd == 'plan':
        return _plan(a)
    if a.cmd == 'finish':
        return _finish(a)
    if a.cmd == 'run':
        return _run(a)
    ap.error(f'unknown command {a.cmd}')
    return 2


if __name__ == '__main__':
    sys.exit(main())
