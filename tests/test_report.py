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
    """A tool must never reach into a sibling campaign UNCONDITIONALLY.

    `BOLTZ_DONOR = Path('.../nise_bdfp44_v7')` lived as a module constant inside v8. That
    class of reference is how a tool rebuilt a finished campaign's ligand and a driver nearly
    wrote into a closed archive -- five incidents.

    What is still allowed, because none of it couples one campaign to another:
      * an `os.environ.get(VAR, default)` -- the default is a convenience, the var is the API
      * a docstring or comment -- usage examples have to name something
      * `replay_convergence.py`'s declared fixture corpus of finished campaigns

    Anything else is a hard failure.
    """
    import ast
    import re
    pat = re.compile(r'/home/data/yuvaln/nise_[a-z0-9_]+')
    bad = []
    for f in sorted((ROOT / 'tools').glob('*.py')):
        src = f.read_text()
        if not pat.search(src):
            continue
        if f.name == 'replay_convergence.py':
            continue                      # declared fixture corpus, documented in the file
        tree = ast.parse(src)
        docstrings = {
            id(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)}
        # map each statement to its source, so we can ask "is this guarded by environ.get?"
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if not pat.search(node.value) or id(node) in docstrings:
                continue
            stmt = _enclosing_statement(tree, node)
            seg = ast.get_source_segment(src, stmt) or ''
            if 'environ.get' in seg:
                continue                  # overridable; the env var is the real interface
            bad.append(f'{f.name}:{node.lineno}: {node.value}')
    assert not bad, (
        'unconditional cross-campaign paths (wrap in os.environ.get or justify):\n'
        + '\n'.join(bad))


def _enclosing_statement(tree, target):
    """The nearest `ast.stmt` ancestor of `target`, or the tree itself."""
    import ast

    def walk(node, current):
        if isinstance(node, ast.stmt):
            current = node
        for child in ast.iter_child_nodes(node):
            if child is target:
                return current
            found = walk(child, current)
            if found is not None:
                return found
        return None

    return walk(tree, None) or tree


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


# ------------------------------------------------------------------ coordination geometry

GEOM_HEADERS = ('Axial angle', 'Lone-pair ang', 'Mg off-imid')


def test_geometry_columns_absent_do_not_break_the_report(v8_html):
    """Backward compatibility: a campaign whose cascade predates these columns still renders.

    make_report drops any COLS key missing from the dataframe, so the only thing to assert is
    that the render succeeded (the fixture would have raised) and nothing half-rendered.
    """
    h = _headers(v8_html)
    assert 'His-Mg' in h, 'the report lost its existing coordination column'


def test_geometry_columns_render_when_present(tmp_path):
    """The other half: when the cascade HAS supplied them, they reach the HTML.

    Built from a minimal run dir rather than a campaign, so this runs without the archives.
    """
    run = tmp_path / 'run'
    (run / 'results' / 'final').mkdir(parents=True)
    (run / 'results' / 'cascade').mkdir(parents=True)
    # the funnel tiles read this; without it make_report raises IndexError on an empty funnel
    (run / 'results' / 'cascade' / 'cascade_metrics.csv').write_text(
        'name,packstat,apo_clashes,buried_charged\n'
        'iter1_chunk_0_seq_1_model_0,0.66,0,0\n'
        'iter2_chunk_0_seq_2_model_0,0.61,0,0\n')
    (run / 'results' / 'final' / 'shortlist.csv').write_text(
        '# minimal fixture\n'
        'name,iteration,buns_score_paper,buns_ligand,buns_protein,packstat,his_mg,'
        'buried_frac,buried_frac_core,'
        'axial_angle,his_lonepair_angle,mg_imidazole_oop,sequence,output\n'
        'iter1_chunk_0_seq_1_model_0,1,6,0,6,0.66,2.43,0.71,0.88,5.6,16.0,14.5,ACDE,o.pdb\n'
        'iter2_chunk_0_seq_2_model_0,2,8,0,8,0.61,2.45,0.68,0.85,9.1,31.2,28.4,FGHI,o2.pdb\n')
    html = _render(run, tmp_path / 'out.html')
    h = _headers(html)
    for col in GEOM_HEADERS:
        assert col in h, f'{col} missing from the report'


def test_geometry_is_advisory_not_a_ranking_term():
    """These measures must NOT enter `composite`. Nothing shows they predict binding, and the
    paper's Fig 1c is a warning about adopting such a term as an objective; promoting them
    would silently change which designs get ordered."""
    src = (ROOT / 'tools' / 'final_shortlist.py').read_text()
    components = src[src.index('COMPONENTS'):src.index('COMPONENTS') + 400]
    for col in ('axial_angle', 'his_lonepair_angle', 'mg_imidazole_oop'):
        assert col not in components, f'{col} leaked into the composite ranking'
