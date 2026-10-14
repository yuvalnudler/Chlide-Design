"""The reporting stage, exercised against the real campaigns when they are present.

Invariants, not bytes: the plan's golden-test rule is to assert on metric tables and
structure, never on rendered output, because a font or matplotlib version bump changes
bytes without changing meaning.

Skipped when the campaign archives are absent (they live outside the repo), so this is a
local guard rather than a CI gate.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V7 = Path('/home/data/yuvaln/nise_bdfp44_v7')
V8 = Path('/home/data/yuvaln/nise_bdfp44_v8')


def _render(run: Path, out: Path) -> str:
    env = {'PYTHONPATH': str(ROOT / 'src'), 'PATH': '/usr/bin:/bin', 'HOME': str(Path.home())}
    r = subprocess.run(
        [sys.executable, str(ROOT / 'tools' / 'make_report.py'),
         '--run-dir', str(run), '--out', str(out)],
        capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return out.read_text()


@pytest.fixture(scope='module')
def v8_html(tmp_path_factory):
    if not V8.is_dir():
        pytest.skip('v8 campaign not present')
    return _render(V8, tmp_path_factory.mktemp('r') / 'v8.html')


@pytest.fixture(scope='module')
def v7_html(tmp_path_factory):
    if not V7.is_dir():
        pytest.skip('v7 campaign not present')
    return _render(V7, tmp_path_factory.mktemp('r') / 'v7.html')


def _headers(html: str) -> list[str]:
    return [re.sub('<[^>]+>', '', h).strip()
            for h in re.findall(r'<th[^>]*>(.*?)</th>', html, re.S)]


# ------------------------------------------------------------------ engine awareness

def test_no_placeholder_leaks(v8_html, v7_html):
    """{ENG}/{XENG} must all be substituted -- a leak means a label went unresolved."""
    for html in (v8_html, v7_html):
        assert '{ENG}' not in html and '{XENG}' not in html
        assert '{order_html}' not in html


def test_leading_engine_is_detected_per_campaign(v8_html, v7_html):
    """v8 is AF3-led with Boltz cross-checking; v7 is the reverse. Same script, both right.

    This is the regression for the stale-label bug: figure 4 said "Boltz prediction"
    over AF3 data because the engine name was typed rather than derived.
    """
    h8, h7 = _headers(v8_html), _headers(v7_html)
    assert 'AF3 lig' in h8 and 'Boltz lig pLDDT' in h8      # AF3 leads, Boltz cross-checks
    assert 'Boltz lig' in h7                                 # Boltz leads
    assert 'Boltz lig pLDDT' not in h7                       # ...so it is not also the cross-check


# ------------------------------------------------------------------ burial columns

def test_burial_columns_present(v8_html):
    h = _headers(v8_html)
    for col in ('Ligand burial', 'Core burial', 'Apolar burial', 'SASA Mg'):
        assert col in h, f'{col} missing from the report'


def test_core_burial_exceeds_whole_ligand(v8_html):
    """The propionate is enforced exposed, so whole-ligand burial is capped BELOW core.

    If this ever inverts, either the atom sets or the exposure constraint has changed.
    """
    h = _headers(v8_html)
    li, ci = h.index('Ligand burial'), h.index('Core burial')
    body = v8_html[v8_html.find('<table id=t'):]
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', body, re.S)
    seen = 0
    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)
        if len(cells) <= ci:
            continue
        whole = re.sub('<[^>]+>', '', cells[li]).strip().rstrip('%')
        core = re.sub('<[^>]+>', '', cells[ci]).strip().rstrip('%')
        if whole in ('-', '') or core in ('-', ''):
            continue
        assert float(core) > float(whole), f'core {core} <= whole {whole}'
        seen += 1
    assert seen > 50, f'only {seen} rows carried burial'


# ------------------------------------------------------------------ order section

def test_order_section_renders_for_v8(v8_html):
    assert '<h2>Suggested order' in v8_html
    assert v8_html.count('class=pick') == 8, 'the 8 picks must be marked in the shortlist'
    assert 'why this one' in v8_html
    assert 'experimental testing' in v8_html


def test_every_pick_has_a_reason(v8_html):
    sec = re.search(r'<h2>Suggested order.*?</tbody>', v8_html, re.S).group(0)
    whys = re.findall(r'<td class=why>(.*?)</td>', sec, re.S)
    assert len(whys) == 8
    assert all(w.strip() for w in whys), 'a pick rendered with an empty rationale'


def test_order_section_absent_without_an_order_dir(v7_html):
    """v7 has no ORDER/ -- the section must not render at all, not render empty."""
    assert 'Suggested order' not in v7_html
    assert 'class=pick' not in v7_html


# ------------------------------------------------------------------ migrated tool wiring

def _load(name: str):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(name, str(ROOT / 'tools' / f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ['x']
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        sys.argv = argv
    return mod


def test_no_cross_campaign_paths_in_tools():
    """A tool must never point at a sibling campaign.

    `BOLTZ_DONOR = Path('.../nise_bdfp44_v7')` lived inside v8. That class of reference is
    how a tool rebuilt a finished campaign's ligand and a driver nearly wrote into a closed
    archive -- five incidents. Usage examples in docstrings are the only allowance.
    """
    import re
    bad = []
    for f in sorted((ROOT / 'tools').glob('*.py')):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if (re.search(r'/home/data/yuvaln/nise_bdfp44_v\d', line)
                    and not line.lstrip().startswith('#')):
                if f.name == 'config_equivalence.py':
                    continue          # usage example in the module docstring
                bad.append(f'{f.name}:{i}: {line.strip()[:70]}')
    assert not bad, 'cross-campaign paths:\n' + '\n'.join(bad)


def test_boltz_cache_resolution():
    """Explicit wins; a campaign without checkpoints must REFUSE, not silently borrow."""
    cf = _load('crossfold')
    v7 = Path('/home/data/yuvaln/nise_bdfp44_v7')
    v8 = Path('/home/data/yuvaln/nise_bdfp44_v8')
    if not (v7 / 'boltz_cache').is_dir():
        pytest.skip('campaigns not present')
    assert cf.resolve_boltz_cache(v7) == v7 / 'boltz_cache'
    assert cf.resolve_boltz_cache(v8, str(v7 / 'boltz_cache')) == v7 / 'boltz_cache'
    with pytest.raises(SystemExit) as e:
        cf.resolve_boltz_cache(v8)
    assert 'boltz_cache' in str(e.value)       # names where it looked


def test_cascade_loads_sasa_from_its_own_directory():
    """One set of SASA constants. If the cascade and sasa_rank diverge, the 142-design
    shortlist and the top-25 detail table stop agreeing and nothing says so."""
    src = (ROOT / 'tools' / 'polizzi_cascade.py').read_text()
    assert "Path(__file__).parent / 'sasa_rank.py'" in src
