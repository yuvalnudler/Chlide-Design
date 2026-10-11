"""The Phase 1 gate: a campaign YAML must reproduce its original driver.

Skipped when the campaign directories are not present (they live outside the repo
and are not part of CI), so this is a local guard rather than a CI gate. It is the
check that keeps the YAMLs honest while the drivers still exist to compare against.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ('v7-boltz-coo.yaml', Path('/home/data/yuvaln/nise_bdfp44_v7/run_nise_bdfp44_v7.py')),
    ('v8-af3-coo.yaml', Path('/home/data/yuvaln/nise_bdfp44_v8/run_nise_bdfp44_v8.py')),
]


@pytest.mark.parametrize('campaign,driver', CASES, ids=[c for c, _ in CASES])
def test_campaign_reproduces_driver(campaign, driver):
    if not driver.is_file():
        pytest.skip(f'{driver} not present -- campaign archives are outside the repo')
    r = subprocess.run(
        [sys.executable, str(ROOT / 'tools' / 'config_equivalence.py'),
         str(ROOT / 'campaigns' / campaign), str(driver)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'EQUIVALENT' in r.stdout
