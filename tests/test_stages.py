"""Stage ordering and truthful reporting. No GPU, no subprocess: this is about structure."""

from __future__ import annotations

import pytest

from chlide.stages.base import Outcome, OrderError, Stage, StageResult, summarise, validate_order
from chlide.stages.terminal import ORDER, PREEXISTING, terminal_stages


def _names(stages):
    return [s.name for s in stages]


# ------------------------------------------------------------------ the ordering guard

def test_shipped_order_validates_for_both_engines():
    for eng in ('af3', 'boltz'):
        terminal_stages(eng)          # raises OrderError if any input has no earlier producer


def test_the_historical_bug_is_caught():
    """af3_monomer and burial_rescore ran BEFORE final_shortlist, which produces what they read.

    On a first run that failed with "no top*.pdb" and a FileNotFoundError. On a second it
    silently used the previous run's files, so a broken order looked correct. The guard must
    fail on this, and name the later stage that actually produces the artifact.
    """
    stages = terminal_stages('af3')
    i_short = _names(stages).index('shortlist')
    i_burial = _names(stages).index('burial_rescore')
    broken = stages[:]
    broken.insert(i_short, broken.pop(i_burial))      # burial before shortlist
    with pytest.raises(OrderError, match='LATER stage|shortlist'):
        validate_order(broken, PREEXISTING)


def test_guard_rejects_an_artifact_nothing_produces():
    s = Stage(name='x', tool='t', reads=('results/invented.csv',))
    with pytest.raises(OrderError, match='nothing produces'):
        validate_order([s], PREEXISTING)


def test_preexisting_loop_artifacts_are_accepted():
    """The loop writes the sampling dataframes; the terminal pipeline must not demand a
    producer for them."""
    s = Stage(name='x', tool='t', reads=('input/sampling_dataframes/*.pkl',))
    validate_order([s], PREEXISTING)


# ------------------------------------------------------------------ the rebuild

def test_rebuild_runs_after_every_scorer_that_feeds_it():
    """The rebuild is the whole reason the order matters: it is what lets the fold-check and
    burial columns enter the RANKING. Drop it or move it early and those measures are computed
    and then ignored."""
    names = _names(terminal_stages('af3'))
    i_rebuild = names.index('shortlist_rebuild')
    for scorer in ('fold_check', 'burial_rescore'):
        assert names.index(scorer) < i_rebuild, f'{scorer} must run before the rebuild'
    for consumer in ('sasa', 'report'):
        assert names.index(consumer) > i_rebuild, f'{consumer} must see the rebuilt ranking'


def test_shortlist_is_built_twice_on_purpose():
    tools = [s.tool for s in terminal_stages('af3')]
    assert tools.count('final_shortlist') == 2


# ------------------------------------------------------------------ engine dependence

def test_fold_check_is_always_the_other_engine():
    """A campaign optimises its own folder's confidence, so that folder cannot be the check."""
    af3 = next(s for s in terminal_stages('af3') if s.engine_dependent)
    boltz = next(s for s in terminal_stages('boltz') if s.engine_dependent)
    assert af3.tool == 'crossfold'        # AF3 led -> Boltz checks
    assert boltz.tool == 'af3_monomer'    # Boltz led -> AF3 checks


def test_dimer_is_always_af3_regardless_of_the_leading_engine():
    """It is an aggregation proxy, not a fold check -- AF3 is the reference either way."""
    for eng in ('af3', 'boltz'):
        d = next(s for s in terminal_stages(eng) if s.name == 'dimer_pae')
        assert d.tool == 'af3_interface_pae'
        assert not d.engine_dependent


# ------------------------------------------------------------------ gpu + optionality

def test_gpu_stages_are_declared():
    gpu = {s.name for s in terminal_stages('af3') if s.needs_gpu}
    assert gpu == {'dimer_pae', 'fold_check'}, 'a GPU stage must say so on a shared box'


def test_scoring_stages_are_optional_but_the_spine_is_not():
    """A missing score should narrow the ranking, not abort a run that cost GPU-days.
    The stages that PRODUCE the ranking are not optional."""
    by = {s.name: s for s in terminal_stages('af3')}
    for n in ('dimer_pae', 'fold_check', 'burial_rescore', 'sasa', 'report', 'figures'):
        assert by[n].optional, f'{n} should be optional'
    for n in ('relax_pool', 'harvest', 'cascade', 'shortlist', 'shortlist_rebuild'):
        assert not by[n].optional, f'{n} must not be optional'


# ------------------------------------------------------------------ truthful reporting

def test_a_failed_stage_makes_the_summary_say_incomplete():
    """A cascade SIGKILLed at 2770/11201 still printed DONE, because the figures stage runs
    regardless and the summary was written unconditionally."""
    rs = [StageResult('cascade', Outcome.FAILED, detail='SIGKILL at 2770/11201'),
          StageResult('figures', Outcome.OK)]
    out = summarise(rs)
    assert out.startswith('INCOMPLETE')
    assert 'cascade' in out and 'SIGKILL' in out
    assert 'DONE' not in out


def test_all_ok_says_done():
    assert summarise([StageResult('a', Outcome.OK)]).startswith('DONE')


def test_skipped_is_reported_not_counted_as_success():
    """A disabled stage contributes NOTHING -- it is neither a success nor a failure, and the
    summary must name it so nobody reads a narrowed ranking as a complete one."""
    out = summarise([StageResult('a', Outcome.OK), StageResult('dimer_pae', Outcome.SKIPPED)])
    assert out.startswith('DONE')
    assert 'skipped' in out and 'dimer_pae' in out
    assert '1/2 stages ok' in out


def test_blocked_counts_as_failure():
    out = summarise([StageResult('sasa', Outcome.BLOCKED, detail='no top*.pdb')])
    assert out.startswith('INCOMPLETE') and 'no top*.pdb' in out


# ------------------------------------------------------------------ every tool exists

def test_every_stage_names_a_tool_that_exists():
    from pathlib import Path
    tools = Path(__file__).resolve().parents[1] / 'tools'
    for eng in ('af3', 'boltz'):
        for s in terminal_stages(eng):
            assert (tools / f'{s.tool}.py').is_file(), f'{s.name} -> missing tools/{s.tool}.py'


def test_order_module_matches_terminal_stages_length():
    assert len(ORDER) == len(terminal_stages('af3'))


# ------------------------------------------------------------------ drift against the shell script

def test_encoded_order_matches_finish_at_iter_sh():
    """tools/finish_at_iter.sh is still what runs a real campaign. Until `chlide finish`
    replaces it, the encoded ORDER must stay a faithful transcription of it -- otherwise the
    tested order and the executed order diverge, which is worse than having only one.

    The script invokes BOTH fold-check branches (an if/else on the leading engine); the
    encoded pipeline picks one, so that position is compared as a set.
    """
    import re
    from pathlib import Path
    sh = Path(__file__).resolve().parents[1] / 'tools' / 'finish_at_iter.sh'
    if not sh.is_file():
        pytest.skip('finish_at_iter.sh not migrated')
    seen = [m for m in re.findall(r'tools/([a-z0-9_]+)\.py', sh.read_text())]
    # collapse the two engine branches into one position
    collapsed, skip_next = [], False
    for i, t in enumerate(seen):
        if skip_next:
            skip_next = False
            continue
        if t == 'crossfold' and i + 1 < len(seen) and seen[i + 1] == 'af3_monomer':
            collapsed.append('FOLD_CHECK')
            skip_next = True
        else:
            collapsed.append(t)
    encoded = ['FOLD_CHECK' if s.engine_dependent else s.tool for s in terminal_stages('af3')]
    assert collapsed == encoded, (
        'the shell script and the encoded ORDER have drifted:\n'
        f'  shell:   {collapsed}\n  encoded: {encoded}')
