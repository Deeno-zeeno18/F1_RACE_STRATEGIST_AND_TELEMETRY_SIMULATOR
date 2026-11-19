# Code clean-up: what changed and what did not

Every self-test was re-run after the edits. Outputs are **identical** to your saved ones
(ADVISOR, DUEL, PIT_STRATEGY, SAFETY_CAR_VSC, TIRES_AND_CARS, BACKTEST --selftest, CALIBRATE --selftest),
except `LIVE_SHOW.py`, which now prints one extra block (CHECK 7, the new calibrate test).
No model constant and no calculation was changed.

## Changes

| file | change |
|---|---|
| ANALYZE_F1_DATA.py | removed the stray `from fontTools.merge.util import avg_int` (unused, and it fails on machines without fontTools) |
| LIVE_SHOW.py | docstring now says LIVE_SHOW.py; the stale CHECK 4b comment now describes what really happens (STAY by 2.28 s); **new CHECK 7**: `calibrate()` must recover a planted 1.25 wear multiplier and must refuse to fit a young tyre |
| SAFETY_CAR_VSC.py | header now says SAFETY_CAR_VSC.py and states which numbers are measured and which hand-set (it said "all placeholders"); the redundant second test in `decide_from_window` removed (best_lap <= latest, so it was implied); the `except Exception` around `stop_window` now records `stop_window_error` instead of silently returning None |
| DUEL.py | header says BUNCH_GAP is measured; unused `import math` removed |
| BACKTEST.py | unused `import os` and dead `quiet_race_laps` removed; docstring lists the tests in order 1, 2, 3 |
| CALIBRATION_LOADER.py | **new**: reads the calibration JSON files and prints each constant in the code next to what the files say, with its status (measured / tuned / hand-set). Changes nothing; shows drift at once |
| data/calibration_6races_2024.json | removed: byte-identical to calibration_2024_first6.json |

## Not changed

- The hero image `/static/hero.jpg` referenced by templates/index.html does not exist; add an image you have the rights to, or remove the reference.
- Chart.js and the fonts load from the internet; self-host them for offline demos.
- The SOFT/HARD offsets were tuned and scored on the same races (in-sample): hold out races and add confidence intervals when you next run the backtest.
- Wiring the JSON values into the constants automatically was left out on purpose: CALIBRATION_LOADER.py shows the differences first, because some constants are deliberately NOT the file's value.

## Try it
    python CALIBRATION_LOADER.py
    python LIVE_SHOW.py
