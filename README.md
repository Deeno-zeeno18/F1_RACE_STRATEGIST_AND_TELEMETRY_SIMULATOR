# F1 Race Strategy & Telemetry Simulator

A from-scratch Python model of Formula 1 race strategy: it predicts lap times, finds the best pit-stop plan,
reacts to safety cars, advises "box or stay out", and checks itself against real race data.

## What it does
| Module | Job |
|---|---|
| `TIRES_AND_CARS.py` | Lap-time model: base pace + fuel burn + tyre wear (with a sigmoid "cliff") + compound and driver offsets |
| `PIT_STRATEGY.py` | Dynamic-programming optimiser for stop laps and compounds (enforces the two-compound rule) |
| `ADVISOR.py` | Pit-now-or-stay-out advice with a safety margin and an ASK (not sure) state |
| `SAFETY_CAR_VSC.py` | Safety car / virtual safety car states and stop windows |
| `DUEL.py` | Minimax strategy duel between two cars |
| `LIVE_SHOW.py` | Replays a race lap by lap, re-calibrating tyre wear live from recent laps |
| `BACKTEST.py` | Scores the model against real races (see "How honest are the numbers?") |
| `HOLDOUT.py` | Leave-one-race-out (out-of-sample) version of the backtest |
| `CALIBRATE_FASTF1.py` | Pulls real OpenF1 data and measures pit loss, tyre wear, fuel effect, safety-car pace |
| `CALIBRATION_LOADER.py` | Compares the constants in code with the calibration JSON files |
| `app.py` + `templates/index.html` | Flask JSON API and a Chart.js dashboard |

## Install and run
```bash
pip install flask pandas numpy requests
python TIRES_AND_CARS.py        # model sanity checks
python LIVE_SHOW.py             # built-in checks, then the live replay
python app.py                   # dashboard at http://127.0.0.1:5000  (needs internet for Chart.js)
python BACKTEST.py 2024:Bahrain 2024:Spain ...   # needs internet (OpenF1)
python HOLDOUT.py --selftest    # offline check of the fair test
python HOLDOUT.py --years 2023,2024 --max 16     # the fair test on real races
```

## How honest are the numbers?
**Fair (out-of-sample) test, `HOLDOUT.py`, leave-one-race-out on 13 dry 2023 races (83 scored stints):**
- Next stop lap: within 3 laps **49%** vs **34%** for a no-hindsight baseline; mean error **5.9** vs **7.3** laps.
- 95% intervals (bootstrap over races): gain in within-3-laps share **+5 to +32 points**; error reduction **+0.6 to +2.3 laps**. The model beats the baseline out of sample, with a wide interval.
- Offsets chosen per fold were unstable (SOFT -0.6 to -0.2, HARD -0.1 to +0.2). Zero offsets score the same on timing (49%, 5.9 laps), so the timing skill comes from the wear and fuel model, not from the tuned offsets.
- **Next compound right: 26% out of sample** (49% in sample). Compound prediction is NOT reliable; the earlier 68% figure was in-sample and is withdrawn.
- 3 of the 13 races had no scorable stints (Australia, Britain, Singapore).

Earlier in-sample figures on other races (4.5 vs 6.4 laps; 52% vs 33%) are superseded by the above.
Other measured results: pit-now-or-stay is right 62% on clear cases, **worse than always saying STAY (~79%)**; replay gap error 18.6 s vs 37.2 s for a level-pace guess (4 races, not re-tested).

Other known limits: VSC values are hand-set; tyre-care and compound offsets are not stable between 2023 and 2024;
one Japan total-time record is a data anomaly; calibration JSON values are copied into code by hand
(`CALIBRATION_LOADER.py` shows any drift). 2024 races were not in the fair test.

## Data
Race data comes from the public OpenF1 API. Nothing here is official F1 data or affiliated with Formula 1.

## Author
Zelden - Mechatronics Engineering, Ashesi University.
