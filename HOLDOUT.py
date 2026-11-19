"""
HOLDOUT.py - the FAIR test of the SOFT / HARD pace offsets.

THE PROBLEM IT SOLVES: BACKTEST.py --sweep chose the two offsets (SOFT -0.3, HARD 0.0) by scoring 63 settings on
8 races, and the headline results were then measured on the same races. That is IN-SAMPLE: the offsets have already
"seen" the answers. This file repeats the choice honestly, by LEAVE-ONE-RACE-OUT cross-validation:

    for each race R:  tune the two offsets on all the OTHER races, then score the model on R only.

Pooling the scores of the held-out races gives an out-of-sample number. It also reports
  - how stable the chosen setting is from fold to fold (a plateau, or a jump?)
  - the in-sample number next to it (the gap between the two is the optimism of the old result)
  - the no-hindsight baseline on the same stints (usual stint length from the OTHER races)
  - 95% confidence intervals, from a bootstrap that resamples whole RACES (stints in one race are not independent)
  - the old placeholder (-0.7, +0.5), zero offsets, and the offsets now in TIRES_AND_CARS.py, for reference.

Run:    python HOLDOUT.py 2023:Bahrain 2023:Spain 2024:Bahrain ...     (at least 4 races; more is better)
        python HOLDOUT.py --years 2023,2024 --max 16
        python HOLDOUT.py --selftest                                    (no internet: races made by the model itself)
Everything printed is also saved in holdout_report.txt.  Expect a long run (the grid is 63 settings x every race).
"""

import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import TIRES_AND_CARS as T
import BACKTEST as B

SOFT_VALUES = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1]
HARD_VALUES = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1]
REFERENCE = {"old placeholder": (-0.7, 0.5), "zero offsets": (0.0, 0.0), "offsets now in the code": (-0.3, 0.0)}
N_BOOT = 2000


# ------------------------------------------------------------------ the grid, computed once
def build_grid(tables, soft_values, hard_values, say):
    """grid[(soft, hard)] = [TEST 1 rows of race 0, TEST 1 rows of race 1, ...]. The offsets are patched in place
    and ALWAYS restored (finally), exactly as BACKTEST.sweep does."""
    keep = dict(T.compound_pace_offset)
    grid = {}
    try:
        for sv in soft_values:
            for hv in hard_values:
                T.compound_pace_offset["SOFT"], T.compound_pace_offset["HARD"] = sv, hv
                grid[(sv, hv)] = [B.test_stop_timing(d, laps, cond) for (_, d, laps, cond) in tables]
            say(f"   SOFT {sv:+.1f} done")
    finally:
        T.compound_pace_offset.update(keep)
    return grid


# ------------------------------------------------------------------ choosing and scoring
def criterion(rows):
    """The same figure of merit as the sweep: within-3-laps share + next-compound agreement."""
    sc = B.score_stop_timing(rows)
    return None if sc is None else sc[0] + sc[2]


def best_setting(grid, race_ids):
    """Best (soft, hard) on the given races. Ties go to the first setting in grid order, like the sweep's ranking.
    Returns (setting, all settings tied for the best)."""
    scored = []
    for key, per_race in grid.items():
        c = criterion([r for i in race_ids for r in per_race[i]])
        if c is not None:
            scored.append((c, key))
    if not scored:
        return None, []
    top = max(c for c, _ in scored)
    tied = [key for c, key in scored if c >= top - 1e-12]
    return tied[0], tied


def baseline_stop(row, ref):
    """No-hindsight guess: the stint's first lap plus the usual stint length of that compound in the reference races."""
    own = ref[row["compound"]]
    return row["start"] + int(round(float(np.median(own)))) if len(own) >= 3 else None


def tag(rows, race_id, ref):
    """Copy rows, adding the race they came from and the baseline's stop lap."""
    return [dict(r, race=race_id, baseline=baseline_stop(r, ref)) for r in rows]


def summary(rows):
    """Numbers for a set of tagged rows. The baseline is scored on exactly the same stints as the model."""
    done = [r for r in rows if r["real"] is not None and r["model"] is not None]
    cn = [r for r in rows if r["next_real"] is not None and r["next_model"] is not None]
    if not done:
        return None
    err = np.array([r["model"] - r["real"] for r in done], float)
    both = [r for r in done if r["baseline"] is not None]
    out = {"n": len(done), "model_within3": float(np.mean(np.abs(err) <= 3)), "model_mae": float(np.mean(np.abs(err))),
           "compound_agree": float(np.mean([r["next_real"] == r["next_model"] for r in cn])) if cn else float("nan")}
    if both:
        m = np.array([r["model"] - r["real"] for r in both], float)
        b = np.array([r["baseline"] - r["real"] for r in both], float)
        out.update({"n_base": len(both), "same_model_within3": float(np.mean(np.abs(m) <= 3)),
                    "base_within3": float(np.mean(np.abs(b) <= 3)), "same_model_mae": float(np.mean(np.abs(m))),
                    "base_mae": float(np.mean(np.abs(b)))})
    return out


def cluster_bootstrap(rows, stat, n_boot=N_BOOT, seed=1):
    """95% interval of stat(rows) when whole RACES are resampled (stints inside one race are not independent)."""
    by_race = {}
    for r in rows:
        by_race.setdefault(r["race"], []).append(r)
    races = list(by_race)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(races), len(races))
        sample = [r for p in pick for r in by_race[races[p]]]
        v = stat(sample)
        if v == v:
            values.append(v)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (float("nan"),) * 2


def stat_gain_within3(rows):
    s = summary(rows)
    return float("nan") if not s or "base_within3" not in s else s["same_model_within3"] - s["base_within3"]


def stat_gain_mae(rows):
    s = summary(rows)
    return float("nan") if not s or "base_mae" not in s else s["base_mae"] - s["same_model_mae"]


# ------------------------------------------------------------------ the experiment
def run(tables, say, soft_values=SOFT_VALUES, hard_values=HARD_VALUES):
    n = len(tables)
    labels = [t[0] for t in tables]
    say(f"HOLD-OUT TEST: {n} races, grid of {len(soft_values) * len(hard_values)} settings, leave-one-race-out")
    if n < 4:
        say("need at least 4 races for a meaningful hold-out")
        return None
    own_lengths = [B.collect_lengths(d, cond) for (_, d, _, cond) in tables]
    refs = []
    for i in range(n):
        others = {c: [x for j in range(n) if j != i for x in own_lengths[j][c]] for c in B.COMPOUNDS}
        refs.append(others)
    grid = build_grid(tables, soft_values, hard_values, say)

    # out-of-sample: choose on the other races, score on the held-out one
    out_rows, choices = [], []
    say(f"\n{'held-out race':32}{'chosen on the others (SOFT, HARD)':>36}{'tied settings':>15}{'n stints':>10}")
    for i in range(n):
        train = [j for j in range(n) if j != i]
        key, tied = best_setting(grid, train)
        if key is None:
            say(f"{labels[i]:32}{'nothing to score':>36}")
            continue
        choices.append(key)
        held = tag(grid[key][i], i, refs[i])
        out_rows += held
        say(f"{labels[i]:32}{str((key[0], key[1])):>36}{len(tied):>15}{len(held):>10}")
    spread_soft = sorted({c[0] for c in choices}), sorted({c[1] for c in choices})
    say(f"\n   settings chosen across folds: SOFT {spread_soft[0]}, HARD {spread_soft[1]}")

    # in-sample reference: tune on ALL races, score on the same races (what the old result did)
    key_all, tied_all = best_setting(grid, list(range(n)))
    in_rows = [r for i in range(n) for r in tag(grid[key_all][i], i, refs[i])]

    def show(title, rows):
        s = summary(rows)
        if not s:
            say(f"   {title}: nothing to score")
            return
        line = (f"   {title:34} n={s['n']:3}  within 3 laps {s['model_within3']:4.0%}  mean error {s['model_mae']:4.1f} laps"
                f"  next compound right {s['compound_agree']:4.0%}")
        say(line)
        if "base_within3" in s:
            say(f"   {'   baseline, same stints (n=' + str(s['n_base']) + ')':34}       within 3 laps {s['base_within3']:4.0%}  mean error {s['base_mae']:4.1f} laps"
                f"   (model on those stints: {s['same_model_within3']:.0%}, {s['same_model_mae']:.1f})")

    say("\nRESULTS")
    show("OUT-OF-SAMPLE (leave-one-race-out)", out_rows)
    show(f"in-sample (tuned on all: {key_all})", in_rows)
    say(f"   the setting chosen on all races is tied with {len(tied_all) - 1} others: " +
        ("a plateau, so the data cannot pick a single value" if len(tied_all) > 1 else "a single winner"))
    say("\n   for reference, fixed settings scored on all races (not tuned in this run, but the code's offsets WERE tuned on these races):")
    for name, key in REFERENCE.items():
        if key in grid:
            show(name + f" {key}", [r for i in range(n) for r in tag(grid[key][i], i, refs[i])])

    result = {"races": labels, "folds": [(labels[i], list(c)) for i, c in enumerate(choices)],
              "out_of_sample": summary(out_rows), "in_sample": summary(in_rows), "in_sample_setting": list(key_all)}
    if out_rows:
        lo, hi = cluster_bootstrap(out_rows, stat_gain_within3)
        lo2, hi2 = cluster_bootstrap(out_rows, stat_gain_mae)
        say(f"\n95% INTERVALS (bootstrap over races, {N_BOOT} resamples), out-of-sample, model against the no-hindsight baseline:")
        say(f"   gain in 'within 3 laps' share: [{lo:+.0%}, {hi:+.0%}]   (positive = the model is better)")
        say(f"   reduction in mean error:       [{lo2:+.1f}, {hi2:+.1f}] laps")
        verdict = ("the model beats the baseline even out of sample" if lo > 0 and lo2 > 0
                   else "the model's advantage over the baseline is NOT clearly established out of sample")
        say(f"   -> {verdict}")
        result["gain_within3_ci"], result["gain_mae_ci"] = [lo, hi], [lo2, hi2]
    return result


# ------------------------------------------------------------------ loading real races
def load_tables(races, say):
    import CALIBRATE_FASTF1 as cf
    tables = []
    for year, name in races:
        try:
            table, label, _ = cf.load_race_openf1(year, name)
            d = cf.prepare(table)
            if d.compound.isin(["INTERMEDIATE", "WET"]).mean() > 0.03:
                say(f"--- {label}: wet race, skipped")
                continue
            tables.append((label, d, int(d.lap.max()), cf.lap_conditions(d).to_dict()))
            say(f"loaded {label}")
        except Exception as problem:
            say(f"--- {year} {name}: {type(problem).__name__}: {str(problem)[:150]}")
    return tables


# ------------------------------------------------------------------ self-test (no internet)
def fake_race(label, race_laps, entries, truth):
    """A race the MODEL generates with the TRUE offsets `truth`; entries = [(driver code, starting compound), ...]."""
    from LIVE_SHOW import RaceState
    from SAFETY_CAR_VSC import best_plan, simulate_stops
    keep = dict(T.compound_pace_offset)
    rows = []
    try:
        T.compound_pace_offset["SOFT"], T.compound_pace_offset["HARD"] = truth
        n = race_laps
        for code, start_comp in entries:
            start = RaceState(driver=B.profile_name(code), compound=start_comp, tyre_age=0, lap=0, race_laps=n,
                              compounds_used={start_comp})
            stops = best_plan(start, ["GREEN"] * n, range(0, n), False, 1.0)[1]
            times = simulate_stops(start, ["GREEN"] * n, stops, 1.0)
            stop_lap = {off + 1: comp for off, comp in stops}
            comp, age, stint = start_comp, 0, 1
            for lap in range(1, n + 1):
                if lap in stop_lap:
                    comp, age, stint = stop_lap[lap], 0, stint + 1
                rows.append({"driver": code, "lap": lap, "time_s": times[lap - 1] + 4.0, "end_s": 0.0, "compound": comp,
                             "tyre_age": age, "stint": stint, "cond": "GREEN", "pos": 1.0, "pit_in": (lap + 1) in stop_lap,
                             "pit_out": lap in stop_lap, "accurate": True, "deleted": False})
                age += 1
    finally:
        T.compound_pace_offset.update(keep)
    return (label, pd.DataFrame(rows), race_laps, {l: "GREEN" for l in range(1, race_laps + 1)})


def selftest():
    """Races generated with TRUE offsets (-0.3, 0.0). The hold-out machinery must (1) run, (2) choose settings whose
    score equals the truth's, (3) score the held-out races highly out of sample, (4) leave T.compound_pace_offset untouched."""
    print("SELF-TEST: races made by the model with true offsets (-0.3, 0.0); leave-one-race-out must recover them\n")
    truth = (-0.3, 0.0)
    layout = [("fake A", 50, [("VER", "MEDIUM"), ("NOR", "HARD")]), ("fake B", 57, [("LEC", "MEDIUM"), ("HAM", "SOFT")]),
              ("fake C", 62, [("RUS", "HARD"), ("PIA", "MEDIUM")]), ("fake D", 66, [("SAI", "MEDIUM"), ("ALO", "HARD")]),
              ("fake E", 70, [("PER", "HARD"), ("GAS", "MEDIUM")])]
    tables = [fake_race(lab, laps, entries, truth) for lab, laps, entries in layout]
    before = dict(T.compound_pace_offset)
    lines = []
    result = run(tables, lambda t="": (print(t), lines.append(t)), soft_values=[-0.7, -0.3, 0.1], hard_values=[0.0, 0.5])
    bad = 0
    if T.compound_pace_offset != before:
        print("CHECK: the offsets in TIRES_AND_CARS were left changed"); bad += 1
    if result is None or result["out_of_sample"] is None:
        print("CHECK: no out-of-sample result"); bad += 1
    else:
        oos = result["out_of_sample"]
        ok = oos["model_within3"] >= 0.9 and oos["compound_agree"] >= 0.9
        print(f"\nout-of-sample within 3 laps {oos['model_within3']:.0%}, compound agreement {oos['compound_agree']:.0%}  {'OK' if ok else 'CHECK'}")
        bad += not ok
        same = result["in_sample"]["model_within3"] >= 0.9
        print(f"in-sample within 3 laps {result['in_sample']['model_within3']:.0%}  {'OK' if same else 'CHECK'}")
        bad += not same
    print("\n" + ("SELF-TEST PASSED" if bad == 0 else f"SELF-TEST: {bad} check(s) need a look"))
    return bad


def main(argv):
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    args = list(argv)
    years, limit = [], None
    for flag in ("--years", "--max"):
        if flag in args:
            i = args.index(flag)
            value = args[i + 1]
            del args[i:i + 2]
            if flag == "--years":
                years = [int(y) for y in value.split(",")]
            else:
                limit = int(value)
    if years:
        import CALIBRATE_FASTF1 as cf
        races = [(y, rec) for y in years for rec in cf.season_races(y)]
        races = races[:limit] if limit else races
    else:
        races = [(int(a.split(":")[0]), a.split(":", 1)[1]) for a in args]
    if not races:
        say("Name some races, e.g.  python HOLDOUT.py 2023:Bahrain 2023:Spain 2024:Bahrain 2024:Spain  or  --years 2023,2024 --max 16")
        return
    tables = load_tables(races, say)
    result = run(tables, say)
    with open("holdout_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if result:
        with open("holdout_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    say("\nSaved holdout_report.txt and holdout_result.json - send them back.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(1 if selftest() else 0)
    main(sys.argv[1:])
