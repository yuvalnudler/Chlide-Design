"""The terminal pipeline: everything that runs once the loop has stopped.

This is `tools/finish_at_iter.sh` as data. The order below is the corrected one, and the
comments say what each position is defending against.

The two GPU scoring stages answer different questions and are NOT interchangeable:

  dimer       ALWAYS AlphaFold3. It is an aggregation proxy over two copies of the design,
              and AF3 is the reference for that regardless of what folded the campaign.
  fold check  ENGINE-DEPENDENT. A campaign optimises its own folder's confidence, so that
              folder cannot also be the check. Running af3_monomer on an AF3-led campaign
              grades the exam with the model that sat it.
"""

from __future__ import annotations

from chlide.engines import other_engine
from chlide.stages.base import Stage, validate_order

# The engine writes here whichever engine it is: af3_adapter.to_boltz_layout puts AF3
# output into a boltz_results_* tree on purpose, so the path name is not evidence.
_PRED = 'input/sampled_backbones/iter_*/boltz_results_boltz_inputs/predictions/*/*_model_0.pdb'

# Written by the LOOP, before any of this runs.
PREEXISTING = {
    'input/sampling_dataframes/*.pkl',
    'input/sampled_backbones/iter_*/backbone_queue.txt',
    _PRED,
}

_FINAL_REPORT = Stage(
    name='final_iteration_report',
    tool='nise_telegram_report',
    args=('--once', '--no-harvest'),
    # Runs FIRST, before the driver and reporter are stopped. The per-iteration reporter polls
    # on an interval and the harvest kills it moments later, so the last iteration's album was
    # simply never sent -- v10 banked 25 iterations and reported 24.
    reads=('input/sampling_dataframes/*.pkl',),
    writes=('logs/plots',),
    optional=True,
)

_RELAX = Stage(
    name='relax_pool',
    tool='relax_pool',
    reads=(_PRED,),
    writes=('results/relaxed_pool/index.csv',),
    # Must not race the driver's own in-loop relax for the pool lock -- the driver is stopped first.
)

_HARVEST = Stage(
    name='harvest',
    tool='harvest_global_top',
    args=('--top', '50'),
    reads=('input/sampling_dataframes/*.pkl',),
    writes=('results/global_top/metrics.csv',),
)

_CASCADE = Stage(
    name='cascade',
    tool='polizzi_cascade',
    args=('--from-metrics', '--pre-relaxed'),
    reads=('results/global_top/metrics.csv', 'results/relaxed_pool/index.csv'),
    writes=('results/cascade/cascade_metrics.csv',),
    # CPU only. GPUs are free from the moment the driver stops until the dimer stage.
)

_DIMER = Stage(
    name='dimer_pae',
    tool='af3_interface_pae',
    args=('--top', '25'),
    reads=('results/cascade/cascade_metrics.csv',),
    writes=('results/af3_dimer/interface_pae.csv',),
    needs_gpu=True,
    optional=True,
)

# final_shortlist produces the top*.pdb and shortlist.csv that the scoring stages read.
# It used to run AFTER them. See stages/base.py.
_SHORTLIST = Stage(
    name='shortlist',
    tool='final_shortlist',
    args=('--top', '25'),
    reads=('results/cascade/cascade_metrics.csv',),
    writes=('results/final/shortlist.csv', 'results/final/top*.pdb', 'results/final/README.md'),
)

_FOLD_CHECK = Stage(
    name='fold_check',
    tool='crossfold',                      # replaced per campaign; see terminal_stages()
    args=('--top', '25', '--mode', 'complex'),
    reads=('results/final/shortlist.csv', 'results/final/top*.pdb'),
    writes=('results/final/crossfold_complex.csv',),
    needs_gpu=True,
    optional=True,
    engine_dependent=True,
)

_BURIAL = Stage(
    name='burial_rescore',
    tool='burial_rescore',
    reads=('results/final/shortlist.csv',),
    writes=('results/final/shortlist_burial.csv',),
    optional=True,
)

# The rebuild is the whole reason the order matters: it is what lets the fold-check and burial
# columns enter the RANKING. Drop it and those measures are computed and then ignored.
_REBUILD = Stage(
    name='shortlist_rebuild',
    tool='final_shortlist',
    args=('--top', '25'),
    reads=('results/final/shortlist_burial.csv', 'results/cascade/cascade_metrics.csv'),
    writes=('results/final/shortlist.csv',),
)

_SASA = Stage(
    name='sasa',
    tool='sasa_rank',
    args=('--top', '25'),
    reads=('results/final/shortlist.csv', 'results/final/top*.pdb'),
    writes=('results/final/sasa/sasa_ranking.csv',),
    optional=True,
)

_ORDER_SET = Stage(
    name='order_set',
    tool='order_set',
    args=('--n', '8'),
    # Runs BEFORE the report and the figures: both render the order section only when the
    # ORDER directory exists, which is why v8's had to be assembled by hand.
    reads=('results/final/shortlist.csv', 'results/final/top*.pdb'),
    writes=('results/final/ORDER',),
    optional=True,
)

_REPORT = Stage(
    name='report',
    tool='make_report',
    reads=('results/final/shortlist.csv',),
    writes=('results/figures/report.html',),
    optional=True,
)

# Runs regardless, which is exactly why it must never be read as proof the run succeeded.
_FIGURES = Stage(
    name='figures',
    tool='paper_figures',
    args=('--telegram',),
    reads=('results/cascade/cascade_metrics.csv',),
    writes=('results/figures/1_self_consistency.png',),
    optional=True,
)

ORDER = [_FINAL_REPORT, _RELAX, _HARVEST, _CASCADE, _DIMER, _SHORTLIST, _FOLD_CHECK,
         _BURIAL, _REBUILD, _SASA, _ORDER_SET, _REPORT, _FIGURES]


def terminal_stages(fold_engine: str) -> list[Stage]:
    """The pipeline for a campaign led by `fold_engine`.

    Only the fold check changes: it is always the engine that did NOT lead.
      AF3-led   -> crossfold.py  (Boltz re-folds the complex)
      Boltz-led -> af3_monomer.py (AF3 folds the single chain)
    """
    checker = other_engine(fold_engine)
    out = []
    for s in ORDER:
        if s.engine_dependent:
            if checker == 'boltz':
                s = Stage(**{**s.__dict__, 'tool': 'crossfold',
                             'args': ('--top', '25', '--mode', 'complex'),
                             'writes': ('results/final/crossfold_complex.csv',)})
            else:
                s = Stage(**{**s.__dict__, 'tool': 'af3_monomer',
                             'args': ('--top', '25'),
                             'writes': ('results/final/af3_monomer.csv',)})
        out.append(s)
    validate_order(out, PREEXISTING)
    return out
