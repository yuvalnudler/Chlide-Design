"""Stage execution, provenance, event log, GPU etiquette. No GPU is ever touched."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chlide.eventlog import EventLog
from chlide.gpu import GpuError, GpuLock, foreign_processes
from chlide.provenance import file_sha256, git_state, write_manifest
from chlide.runner import StageRunner
from chlide.stages.base import Outcome, Stage

ROOT = Path(__file__).resolve().parents[1]


def _tool(tmp_path: Path, name: str, body: str) -> None:
    """Install a fake tool the runner will invoke."""
    d = ROOT / 'tools'
    (d / f'{name}.py').write_text(body)


# ------------------------------------------------------------------ outcomes

def test_disabled_stage_contributes_nothing(tmp_path):
    s = Stage(name='x', tool='sasa_rank')
    r = StageRunner(tmp_path).run_one(s, enabled=False)
    assert r.outcome is Outcome.SKIPPED and not r.produced


def test_gpu_stage_is_skipped_unless_allowed(tmp_path):
    """On a shared box a GPU stage must be opt-in, not opt-out."""
    s = Stage(name='g', tool='sasa_rank', needs_gpu=True)
    assert StageRunner(tmp_path, allow_gpu=False).run_one(s).outcome is Outcome.SKIPPED
    # with allow_gpu it gets as far as checking inputs, which is a different outcome
    assert StageRunner(tmp_path, allow_gpu=True).run_one(
        Stage(name='g', tool='sasa_rank', needs_gpu=True, reads=('nope/*.csv',))
    ).outcome is Outcome.BLOCKED


def test_absent_inputs_are_blocked_not_failed(tmp_path):
    """BLOCKED and FAILED mean different things: one never ran, the other ran and lost."""
    s = Stage(name='x', tool='sasa_rank', reads=('results/nothing.csv',))
    r = StageRunner(tmp_path).run_one(s)
    assert r.outcome is Outcome.BLOCKED and 'inputs absent' in r.detail


def test_missing_tool_fails(tmp_path):
    r = StageRunner(tmp_path).run_one(Stage(name='x', tool='no_such_tool_xyz'))
    assert r.outcome is Outcome.FAILED and 'no tools/' in r.detail


def test_exit_zero_without_the_output_is_a_failure(tmp_path):
    """A stage that returns 0 and writes nothing has failed, whatever it returned."""
    name = '_fake_silent'
    _tool(tmp_path, name, 'import sys\nsys.exit(0)\n')
    try:
        s = Stage(name='s', tool=name, writes=('results/expected.csv',))
        r = StageRunner(tmp_path).run_one(s)
        assert r.outcome is Outcome.FAILED
        assert 'did not write' in r.detail
    finally:
        (ROOT / 'tools' / f'{name}.py').unlink(missing_ok=True)


def test_a_stage_that_writes_its_output_is_ok(tmp_path):
    name = '_fake_writer'
    _tool(tmp_path, name, (
        'import sys, pathlib\n'
        'run = pathlib.Path(sys.argv[sys.argv.index("--run-dir") + 1])\n'
        '(run / "results").mkdir(parents=True, exist_ok=True)\n'
        '(run / "results" / "expected.csv").write_text("a,b\\n1,2\\n")\n'))
    try:
        s = Stage(name='s', tool=name, writes=('results/expected.csv',))
        r = StageRunner(tmp_path).run_one(s)
        assert r.outcome is Outcome.OK, r.detail
        assert r.produced == ['results/expected.csv']
    finally:
        (ROOT / 'tools' / f'{name}.py').unlink(missing_ok=True)


def test_nonzero_exit_is_reported_with_its_message(tmp_path):
    name = '_fake_boom'
    _tool(tmp_path, name,
          'import sys\nprint("boom: ligand missing", file=sys.stderr)\nsys.exit(3)\n')
    try:
        r = StageRunner(tmp_path).run_one(Stage(name='s', tool=name))
        assert r.outcome is Outcome.FAILED
        assert 'exit 3' in r.detail and 'boom' in r.detail
    finally:
        (ROOT / 'tools' / f'{name}.py').unlink(missing_ok=True)


# ------------------------------------------------------------------ halting

def test_a_spine_failure_halts_the_run(tmp_path):
    """Continuing past a failed non-optional stage builds a ranking on a missing artifact --
    the failure mode that makes a broken pipeline look like a working one."""
    name = '_fake_boom2'
    _tool(tmp_path, name, 'import sys\nsys.exit(1)\n')
    try:
        stages = [Stage(name='spine', tool=name, optional=False),
                  Stage(name='after', tool='sasa_rank')]
        res = StageRunner(tmp_path).run_all(stages)
        names = [r.stage for r in res]
        assert 'after' not in names
        assert names[-1] == '(halted)'
    finally:
        (ROOT / 'tools' / f'{name}.py').unlink(missing_ok=True)


def test_an_optional_failure_does_not_halt(tmp_path):
    """A missing score narrows the ranking; it must not abort a run that cost GPU-days."""
    name = '_fake_boom3'
    _tool(tmp_path, name, 'import sys\nsys.exit(1)\n')
    try:
        stages = [Stage(name='score', tool=name, optional=True),
                  Stage(name='after', tool='sasa_rank', reads=('nope/*',), optional=True)]
        res = StageRunner(tmp_path).run_all(stages)
        assert [r.stage for r in res] == ['score', 'after']
        assert res[0].outcome is Outcome.FAILED       # it did fail
        assert '(halted)' not in [r.stage for r in res]  # and the run continued anyway
    finally:
        (ROOT / 'tools' / f'{name}.py').unlink(missing_ok=True)


# ------------------------------------------------------------------ worker sizing

def test_workers_leave_the_reserve(tmp_path):
    """Sized from OTHER users' load. Counting our own collapsed the pool to 4 and the cascade
    to 14 hours -- the cascade was the load being measured."""
    total = os.cpu_count() or 8
    w = StageRunner(tmp_path, reserve_cpus=16).workers()
    assert 1 <= w <= max(1, total - 16)
    assert StageRunner(tmp_path, reserve_cpus=10_000).workers() == 1


# ------------------------------------------------------------------ provenance

def test_manifest_records_git_dirtiness_and_weights(tmp_path):
    """A manifest naming a commit that does not contain the code that ran is worse than none."""
    from chlide.config import load_campaign
    from chlide.loop import nise_params
    cfg = load_campaign(ROOT / 'campaigns' / 'v8-af3-coo.yaml')
    params = nise_params(cfg, tmp_path)
    out = write_manifest(tmp_path, cfg, params, campaign_file=Path('campaigns/v8-af3-coo.yaml'))
    m = json.loads(out.read_text())
    assert m['campaign'] == 'v8-af3-coo'
    assert 'dirty' in m['git'] and 'commit' in m['git']
    assert m['engine']['fold'] == 'af3' and m['engine']['cross_check'] == 'boltz'
    assert 'laser_checkpoint' in m['weights']
    assert m['params']['num_iterations'] == 25
    assert m['config']['gates']['metal_r0'] == 2.15
    json.dumps(m)       # must round-trip: sets and Paths are serialised


def test_git_state_reports_a_dirty_tree_honestly():
    g = git_state(ROOT)
    assert g['commit'] and len(g['commit']) == 40
    assert isinstance(g['dirty'], bool)


def test_sha256_of_a_known_file(tmp_path):
    f = tmp_path / 'x'
    f.write_bytes(b'abc')
    assert file_sha256(f) == (
        'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
    assert file_sha256(tmp_path / 'missing') is None


# ------------------------------------------------------------------ event log

def test_eventlog_survives_a_truncated_line(tmp_path):
    """cascade_partial.jsonl proved the pattern: a kill -9 truncates the LAST line only."""
    log = EventLog(tmp_path / 'events.jsonl', run_id='r1')
    log.emit(event='a', n=1)
    log.emit(event='b', n=2)
    with (tmp_path / 'events.jsonl').open('a') as fh:
        fh.write('{"event": "trunc"')          # killed mid-write
    rows = log.read()
    assert [r['event'] for r in rows] == ['a', 'b']
    assert all(r['run_id'] == 'r1' and 'ts' in r for r in rows)


def test_eventlog_never_raises(tmp_path):
    """A logger that can take down the run it is logging is worse than no logger."""
    log = EventLog(Path('/proc/nonexistent/events.jsonl'), run_id='r')
    log.emit(event='x')                        # must not raise
    assert log.read() == []


# ------------------------------------------------------------------ gpu etiquette

def test_lock_is_exclusive_and_reclaims_a_stale_one(tmp_path):
    with GpuLock(['cuda:0'], lock_dir=tmp_path) as held:
        assert len(held.held) == 1
        with pytest.raises(GpuError, match='locked by a live process'):
            GpuLock(['cuda:0'], lock_dir=tmp_path).acquire()
    assert not list(tmp_path.glob('*.lock'))

    # a lock from a dead pid must be reclaimed, or a reboot blocks every future run
    stale = tmp_path / 'cuda_1.lock'
    stale.write_text('999999999 ghost 2020-01-01\n')
    with GpuLock(['cuda:1'], lock_dir=tmp_path):
        assert stale.is_file()


def test_lock_reclaims_an_expired_one(tmp_path):
    stale = tmp_path / 'cuda_2.lock'
    stale.write_text(f'{os.getpid()} me 2020-01-01\n')
    os.utime(stale, (0, 0))                    # ancient
    with GpuLock(['cuda:2'], lock_dir=tmp_path, ttl=10):
        pass                                   # TTL beats liveness for an abandoned lock


def test_foreign_process_detection_excludes_us():
    me = os.environ.get('USER')
    assert all(p.user != me for p in foreign_processes())
