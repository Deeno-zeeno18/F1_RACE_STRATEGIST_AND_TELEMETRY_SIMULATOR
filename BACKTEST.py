"""
BACKTEST.py - does the model's advice match what real teams did?

It takes real races (from the OpenF1 data your CALIBRATE_FASTF1.py already downloaded), puts the model in the
real situation of each driver, and compares the model's answer with what the driver's team actually did.

TEST 1  "When will he stop?"   At the start of every real stint (a fresh set just fitted) we ask the model for
        the best plan for the rest of the race, and compare its NEXT STOP LAP with the real one.
        A trivial baseline is shown next to it: "he will stop after the usual stint length for that compound".
        If the model cannot beat the baseline, it is not adding anything.
TEST 2  "Pit now or stay?"     In the middle of a stint, with the real lap times so far, we ask the engineer
        (the same code as the dashboard) PIT / STAY / ASK, and compare with whether the driver really stopped
        at the end of that lap or the next, within 5 laps, or not at all in the next 5 laps.
TEST 3  "Replay the race"  (--replay) Every driver who finished on the lead lap is re-run through the model with the
        stops and tyres he REALLY used. Predicted finishing gaps and order are compared with the real ones.

Only quiet stretches are used: no safety car, VSC, red flag or wet tyres during the stint (the model's advice
for those depends on things the real teams knew and we do not). Everything printed is also saved in
backtest_report.txt.

Run:    python BACKTEST.py 2024:Bahrain 2024:Spain            (races by place name, like CALIBRATE_FASTF1.py)
        python BACKTEST.py --years 2024 --max 6               (the first 6 races of a season)
        python BACKTEST.py 2024:Bahrain ... --sweep           (which SOFT / HARD pace offsets make the model choose like the teams?)
        python BACKTEST.py 2024:Bahrain 2024:Spain --replay     (replay whole races: finishing gaps and order, quiet races only)
        python BACKTEST.py --selftest                         (the model against races it generated itself)
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import CALIBRATE_FASTF1 as cf
from LIVE_SHOW import RaceState, make_car, calibrate
from SAFETY_CAR_VSC import advise_with_track, best_plan
from PIT_STRATEGY import simulate_strategy
from TIRES_AND_CARS import drivers

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
SAMPLE_EVERY = 2            # laps between live-decision samples inside a stint
MIN_AGE = 6                 # do not ask "pit now?" on a tyre younger than this
NOW_LAPS, SOON_LAPS = 1, 5   # 'now' = he stopped at the end of this lap or the next; 'soon' = within 5 laps


# ------------------------------------------------------------------ helpers
def profile_name(code):
    """The driver's entry in the model if he has one, else the average driver."""
    return code if code in drivers else "balanced"


def model_lap(code, compound, age, lap, race_laps):
    """The model's lap time for `lap` (1 = first lap) on a set of this age."""
    st = RaceState(driver=profile_name(code), compound=compound, tyre_age=age, lap=lap - 1, race_laps=race_laps)
    return make_car(st, age, lap - 1).calculate_lap_time()


def race_level(d, race_laps):
    """
    Real tracks have their own lap time (Bahrain ~96 s, Monaco ~76 s) but the model's base time is 92 s.
    This is the typical gap between real and model on clean laps. It is removed from the real laps before the
    engineer sees them, so it only compares SHAPE (how the lap times move), not the circuit's length.
    """
    c = cf.clean_laps(d)
    if len(c) == 0:
        return 0.0
    rows = c.iloc[::max(1, len(c) // 300)]
    gaps = []
    for r in rows.itertuples():
        try:
            gaps.append(r.time_s - model_lap(r.driver, r.compound, int(r.tyre_age), int(r.lap), race_laps))
        except Exception:
            continue
    return float(np.median(gaps)) if gaps else 0.0


def stints_of(g):
    """One driver's table -> list of stints: compound, first lap, last lap, the lap he pitted at the end (or None),
    and whether the stint was quiet (all green, nothing odd)."""
    g = g.sort_values("lap")
    out, cur = [], None
    for r in g.itertuples():
        if cur is None or r.stint != cur["stint"]:
            cur = {"stint": r.stint, "compound": r.compound, "first": int(r.lap), "last": int(r.lap),
                   "pit_in": None, "quiet": True, "rows": []}
            out.append(cur)
        cur["last"] = int(r.lap)
        cur["rows"].append(r)
        if r.pit_in:
            cur["pit_in"] = int(r.lap)
        if r.cond != "GREEN" and not (r.pit_in or r.pit_out):
            cur["quiet"] = False
        if r.compound not in COMPOUNDS:
            cur["quiet"] = False
    return out


def interrupted_in(d_cond, first, last):
    """True when some lap of the stint was not green for the field (a stop under a safety car is not a normal stop)."""
    return any(d_cond.get(l, "GREEN") != "GREEN" for l in range(first, last + 1))


# ------------------------------------------------------------------ test 1
def collect_lengths(d, cond_map):
    """Real stint lengths (laps) per compound in one race, quiet stints that ended in a stop."""
    lengths = {c: [] for c in COMPOUNDS}
    for code, g in d.groupby("driver"):
        for s in stints_of(g):
            if s["quiet"] and s["pit_in"] is not None and s["compound"] in COMPOUNDS \
                    and not interrupted_in(cond_map, s["first"], s["last"]):
                lengths[s["compound"]].append(s["last"] - s["first"] + 1)
    return lengths


def test_stop_timing(d, race_laps, cond_map, ref_lengths=None):
    """ref_lengths: usual stint lengths for the baseline. Pass lengths from OTHER races for a fair comparison."""
    results, lengths = [], {c: [] for c in COMPOUNDS}
    all_stints = {}
    for code, g in d.groupby("driver"):
        sts = stints_of(g)
        all_stints[code] = sts
        for s in sts:
            if s["quiet"] and s["pit_in"] is not None and not interrupted_in(cond_map, s["first"], s["last"]):
                lengths[s["compound"]].append(s["last"] - s["first"] + 1)

    for code, sts in all_stints.items():
        used = set()
        for k, s in enumerate(sts):
            used.add(s["compound"])
            if s["compound"] not in COMPOUNDS or k == 0:               # stint 0 starts from the grid, not from a stop
                continue
            if not s["quiet"] or interrupted_in(cond_map, s["first"], s["last"]):
                continue
            in_lap = s["first"] - 1                                    # laps completed before this set's first lap
            n = race_laps - in_lap
            if n < 8:
                continue
            real_stop = None if s["pit_in"] is None else s["pit_in"] + 1   # the model counts the stop on the NEW set's first lap
            try:
                state = RaceState(driver=profile_name(code), compound=s["compound"], tyre_age=0, lap=in_lap,
                                  race_laps=race_laps, compounds_used=set(used))
                found = best_plan(state, ["GREEN"] * n, range(0, n), True, 1.0)
            except Exception:
                continue
            if found is None:
                continue
            stops = found[1]
            predicted = None if not stops else state.lap + stops[0][0] + 1
            comp_next = None if not stops else stops[0][1]
            own = (ref_lengths if ref_lengths is not None else lengths)[s["compound"]]
            base = None
            if len(own) >= 3:
                base = s["first"] + int(round(float(np.median(own))))
            results.append({"driver": code, "compound": s["compound"], "start": s["first"], "real": real_stop,
                            "model": predicted, "baseline": base,
                            "next_real": sts[k + 1]["compound"] if k + 1 < len(sts) else None, "next_model": comp_next})
    return results


# ------------------------------------------------------------------ test 2
def test_live(d, race_laps, cond_map, level, max_samples):
    rows = []
    for code, g in d.groupby("driver"):
        sts = stints_of(g)
        used = set()
        for k, s in enumerate(sts):
            used.add(s["compound"])
            if s["compound"] not in COMPOUNDS or not s["quiet"] or interrupted_in(cond_map, s["first"], s["last"] + 8):
                continue
            by_lap = {int(r.lap): r for r in s["rows"]}
            for L in range(s["first"] + MIN_AGE, s["last"] + 1, SAMPLE_EVERY):
                if race_laps - L < 10:
                    break
                age_next = int(by_lap[L].tyre_age) + 1
                # real laps in the stint so far: clean, consecutive, ending at lap L (at most 6)
                recent = []
                for lap in range(L, s["first"] - 1, -1):
                    r = by_lap[lap]
                    if r.pit_in or r.pit_out or r.cond != "GREEN" or not r.accurate or np.isnan(r.time_s) or len(recent) >= 6:
                        break
                    recent.append(r.time_s - level)
                recent.reverse()
                steps = None if s["pit_in"] is None else s["pit_in"] - L        # 0 = he pitted at the end of this very lap
                if steps is not None and steps < 0:
                    continue
                if steps is None:
                    truth = "later"
                elif steps <= NOW_LAPS:
                    truth = "now"
                elif steps <= SOON_LAPS:
                    truth = "soon"
                else:
                    truth = "later"
                try:
                    state = RaceState(driver=profile_name(code), compound=s["compound"], tyre_age=age_next, lap=L,
                                      race_laps=race_laps, compounds_used=set(used), recent_laps=recent)
                    cal = calibrate(state)
                    out = advise_with_track(state, [], wear_mult=cal["multiplier"], err_per_lap=cal["error_per_lap"])
                except Exception:
                    continue
                options = [b for b in (out.get("best_now"), out.get("best_later")) if b]
                model_stop = None
                if options:
                    best = min(options, key=lambda b: b["total"])
                    model_stop = None if not best["stops"] else L + best["stops"][0][0] + 1
                real_stop = None if s["pit_in"] is None else s["pit_in"] + 1
                rows.append({"driver": code, "lap": L, "age": age_next, "truth": truth, "decision": out["decision"],
                             "diff": out.get("diff"), "mult": cal["multiplier"], "model_stop": model_stop,
                             "real_stop": real_stop, "window": out.get("stop_window")})
                if len(rows) >= max_samples:
                    return rows
    return rows


# ------------------------------------------------------------------ reporting
def report_stop_timing(results, say):
    done = [r for r in results if r["real"] is not None and r["model"] is not None]
    say("   (baseline = the usual stint length for that compound in the OTHER races, so it has no hindsight; with one race it uses that race)")
    say(f"\nTEST 1 - when will he stop?   ({len(results)} stints asked, {len(done)} where both model and driver stopped again)")
    if done:
        err = np.array([r["model"] - r["real"] for r in done], float)
        say(f"   model      mean error {np.mean(np.abs(err)):5.1f} laps   median signed {np.median(err):+5.1f}   within 3 laps {np.mean(np.abs(err) <= 3):4.0%}")
        bdone = [r for r in done if r["baseline"] is not None]
        if bdone:
            berr = np.array([r["baseline"] - r["real"] for r in bdone], float)
            merr = np.array([r["model"] - r["real"] for r in bdone], float)
            say(f"   baseline   mean error {np.mean(np.abs(berr)):5.1f} laps   median signed {np.median(berr):+5.1f}   within 3 laps {np.mean(np.abs(berr) <= 3):4.0%}"
                f"   (usual stint length per compound, same stints: model {np.mean(np.abs(merr)):.1f})")
        by_comp = {}
        for r in done:
            by_comp.setdefault(r["compound"], []).append(r["model"] - r["real"])
        for c_, v in sorted(by_comp.items()):
            say(f"      on {c_:6} n={len(v):3}  model minus real (median) {np.median(v):+5.1f} laps")
    no_stop_both = sum(1 for r in results if r["real"] is None and r["model"] is None)
    model_only = sum(1 for r in results if r["real"] is None and r["model"] is not None)
    real_only = sum(1 for r in results if r["real"] is not None and r["model"] is None)
    say(f"   he did not stop again and the model agreed: {no_stop_both};  model wanted a stop he did not make: {model_only};  "
        f"he stopped and the model said no more stops: {real_only}")
    cn = [r for r in results if r["next_real"] is not None and r["next_model"] is not None]
    if cn:
        say(f"   next compound chosen by the model equals the real one in {np.mean([r['next_real'] == r['next_model'] for r in cn]):.0%} of {len(cn)} stops")


def window_for_scale(w, scale):
    """The stop window if the margin were `scale` times bigger (needs the per-lap costs the engineer returns)."""
    costs = {int(k): v for k, v in w["costs"].items()}
    best = w["best_lap"]
    limit = w["best_cost"] + scale * w["margin"]
    lo = hi = best
    while lo - 1 in costs and costs[lo - 1] <= limit:
        lo -= 1
    while hi + 1 in costs and costs[hi + 1] <= limit:
        hi += 1
    return lo, hi


def report_window_scales(rows, say):
    """How wide must the window be to contain the real stop lap most of the time?"""
    win = [r for r in rows if r.get("window") and r["real_stop"] is not None and r["window"].get("costs")]
    if not win:
        return
    say(f"\n   Stop window width needed (n={len(win)}): scale x the forecast-error margin")
    say(f"   {'scale':>7}{'contains the real stop':>25}{'median width (laps)':>22}")
    for scale in (1, 2, 3, 4, 6, 8):
        spans = [window_for_scale(r["window"], scale) for r in win]
        inside = np.mean([lo <= r["real_stop"] <= hi for (lo, hi), r in zip(spans, win)])
        width = np.median([hi - lo + 1 for lo, hi in spans])
        say(f"   {scale:>7}{inside:>25.0%}{width:>22.0f}")


def report_live(rows, say):
    say(f"\nTEST 2 - pit now or stay?   ({len(rows)} situations)")
    if not rows:
        return
    say(f"   {'what he really did':28}{'n':>5}{'model said PIT':>16}{'STAY':>8}{'ASK':>8}")
    for truth, label in (("now", "stopped this lap or next"), ("soon", "stopped 2-5 laps later"), ("later", "no stop within 5 laps")):
        sub = [r for r in rows if r["truth"] == truth]
        if not sub:
            continue
        n = len(sub)
        f = lambda k: np.mean([r["decision"] == k for r in sub])
        say(f"   {label:28}{n:>5}{f('PIT'):>15.0%}{f('STAY'):>8.0%}{f('ASK'):>8.0%}")
    decided = [r for r in rows if r["decision"] != "ASK" and r["truth"] != "soon"]
    if decided:
        right = [(r["decision"] == "PIT") == (r["truth"] == "now") for r in decided]
        say(f"   when the model made a call (not ASK) on the clear cases: right {np.mean(right):.0%} of {len(decided)}")
    say(f"   ASK share overall: {np.mean([r['decision'] == 'ASK' for r in rows]):.0%}")
    report_window_scales(rows, say)
    both = [r for r in rows if r["model_stop"] is not None and r["real_stop"] is not None]
    if both:
        e = np.array([r["model_stop"] - r["real_stop"] for r in both], float)
        say(f"   live-updated next-stop lap (uses the real laps so far): mean error {np.mean(np.abs(e)):.1f} laps, "
            f"within 3 laps {np.mean(np.abs(e) <= 3):.0%}, median signed {np.median(e):+.1f}   (n={len(both)})")
        win = [r for r in rows if r.get("window") and r["real_stop"] is not None
               and r["real_stop"] <= r["lap"] + 25]
        if win:
            inside = [r["window"]["earliest"] <= r["real_stop"] <= r["window"]["latest"] for r in win]
            widths = [r["window"]["latest"] - r["window"]["earliest"] + 1 for r in win]
            say(f"   the engineer's STOP WINDOW contained the real stop lap in {np.mean(inside):.0%} of {len(win)} cases "
                f"(median window {np.median(widths):.0f} laps wide)")
        near = [x for x, r in zip(e, both) if r["truth"] in ("now", "soon")]
        if near:
            say(f"      when the stop was really within 5 laps: mean error {np.mean(np.abs(near)):.1f} laps, "
                f"within 3 laps {np.mean(np.abs(np.array(near)) <= 3):.0%}   (n={len(near)})")


# ------------------------------------------------------------------ replay
def rank_correlation(a, b):
    """Spearman rank correlation of two lists (no ties handling needed at this precision)."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def replay_race(d, race_laps, cond_map, level, min_drivers=8):
    """
    Re-run every lead-lap finisher through the model with his real stints. Returns a list of rows
    (driver, real total, predicted total, strategy) or None if the race is not quiet enough to replay.
    """
    non_green = np.mean([v != "GREEN" for v in cond_map.values()]) if cond_map else 1.0
    if non_green > 0.03:
        return None, f"{non_green:.0%} of laps were under a safety car, VSC or odd conditions"
    rows = []
    for code, g in d.groupby("driver"):
        g = g.sort_values("lap")
        if len(g) != race_laps or g.time_s.isna().any() or int(g.lap.min()) != 1:
            continue
        sts = stints_of(g)
        if any(s_["compound"] not in COMPOUNDS for s_ in sts):
            continue
        strategy = [(s_["compound"], s_["last"] - s_["first"] + 1) for s_ in sts]
        try:
            pred = simulate_strategy(strategy, race_laps, driver=profile_name(code)) + level * race_laps
        except Exception:
            continue
        rows.append({"driver": code, "real": float(g.time_s.sum()), "pred": float(pred), "strategy": strategy})
    if len(rows) < min_drivers:
        return None, f"only {len(rows)} lead-lap finishers could be replayed"
    return rows, ""


def replay_summary(rows):
    real = np.array([r["real"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    real_gap, pred_gap = real - real.min(), pred - pred.min()
    err = np.abs(pred_gap - real_gap)
    return {"n": len(rows), "gap_err": float(err.mean()), "gap_err_baseline": float(real_gap.mean()),
            "total_err": float(np.abs(pred - real).mean()), "rank": rank_correlation(real, pred),
            "winner_top3": bool(int(np.argmin(pred)) in set(np.argsort(real)[:3].tolist()))}


def run_replay(tables, say):
    say("TEST 3 - replay: each driver's real stops and tyres run through the model; predicted finishing gaps vs real")
    done = []
    for label, d, race_laps, cond_map in tables:
        level = race_level(d, race_laps)
        rows, why = replay_race(d, race_laps, cond_map, level)
        if rows is None:
            say(f"\n--- {label}: not replayed ({why})")
            continue
        sm = replay_summary(rows)
        done.append(sm)
        say(f"\n=== {label}: {sm['n']} drivers replayed ===")
        say(f"   finishing gap to the best replayed car: model off by {sm['gap_err']:.1f} s on average "
            f"(predicting everyone level would be off by {sm['gap_err_baseline']:.1f} s)")
        say(f"   absolute race time off by {sm['total_err']:.1f} s on average (after removing the circuit's lap-time level)")
        say(f"   finishing order: rank correlation {sm['rank']:.2f}; model's winner is in the real top 3: {'yes' if sm['winner_top3'] else 'no'}")
    if done:
        say("\n" + "=" * 70 + "\nALL REPLAYED RACES")
        say(f"   {len(done)} races, {sum(s_['n'] for s_ in done)} drivers")
        say(f"   mean finishing-gap error {np.mean([s_['gap_err'] for s_ in done]):.1f} s "
            f"(level-guess baseline {np.mean([s_['gap_err_baseline'] for s_ in done]):.1f} s)")
        say(f"   mean absolute race-time error {np.mean([s_['total_err'] for s_ in done]):.1f} s")
        ranks = [s_["rank"] for s_ in done if s_["rank"] == s_["rank"]]
        if ranks:
            say(f"   mean rank correlation {np.mean(ranks):.2f};  winner inside the real top 3 in {sum(s_['winner_top3'] for s_ in done)} of {len(done)} races")
    else:
        say("no race could be replayed")


# ------------------------------------------------------------------ sweep
def score_stop_timing(results):
    """Numbers used to compare settings: (within-3 share, mean error, compound agreement, n)."""
    done = [r for r in results if r["real"] is not None and r["model"] is not None]
    cn = [r for r in results if r["next_real"] is not None and r["next_model"] is not None]
    if not done or not cn:
        return None
    err = np.array([r["model"] - r["real"] for r in done], float)
    return (float(np.mean(np.abs(err) <= 3)), float(np.mean(np.abs(err))),
            float(np.mean([r["next_real"] == r["next_model"] for r in cn])), len(done))


def sweep(tables, say):
    """
    Which SOFT and HARD pace offsets (seconds against MEDIUM) make the optimiser choose like the real teams?
    The race data could not measure these directly, but the teams' choices carry the information: if SOFT were
    really 0.7 s quicker, everyone would run SOFT. Each setting is scored on TEST 1 (next-stop lap and next compound).
    """
    import TIRES_AND_CARS as T
    keep = dict(T.compound_pace_offset)
    soft_values = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1]
    hard_values = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1]
    say(f"SWEEP: {len(soft_values) * len(hard_values)} settings, scored on {len(tables)} races (TEST 1 only)")
    rows = []
    try:
        for sv in soft_values:
            for hv in hard_values:
                T.compound_pace_offset["SOFT"], T.compound_pace_offset["HARD"] = sv, hv
                results = []
                for d, race_laps, cond_map in tables:
                    results += test_stop_timing(d, race_laps, cond_map)
                sc = score_stop_timing(results)
                if sc:
                    rows.append((sv, hv) + sc)
            say(f"   SOFT {sv:+.1f} done")
    finally:
        T.compound_pace_offset.update(keep)
    if not rows:
        say("nothing to score")
        return
    say(f"\n{'SOFT':>6}{'HARD':>6}{'within 3 laps':>15}{'mean error':>12}{'next compound right':>21}")
    ranked = sorted(rows, key=lambda r: -(r[2] + r[4]))
    for r in ranked[:10]:
        say(f"{r[0]:>+6.1f}{r[1]:>+6.1f}{r[2]:>15.0%}{r[3]:>12.1f}{r[4]:>21.0%}")
    cur = [r for r in rows if abs(r[0] + 0.7) < 1e-9 and abs(r[1] - 0.5) < 1e-9]
    if cur:
        r = cur[0]
        say(f"\ncurrent assumption (-0.7, +0.5): within 3 laps {r[2]:.0%}, mean error {r[3]:.1f}, next compound right {r[4]:.0%}")
    best = ranked[0]
    say(f"best by (within-3 + compound agreement): SOFT {best[0]:+.1f}, HARD {best[1]:+.1f}")


# ------------------------------------------------------------------ running
def backtest_table(d, label, say, max_samples=60, ref_lengths=None):
    race_laps = int(d.lap.max())
    cond_map = cf.lap_conditions(d).to_dict()
    level = race_level(d, race_laps)
    say(f"\n=== {label}: {race_laps} laps, real laps run {level:+.1f} s from the model's base time (removed before asking) ===")
    t1 = test_stop_timing(d, race_laps, cond_map, ref_lengths)
    t2 = test_live(d, race_laps, cond_map, level, max_samples)
    return t1, t2


def main(argv):
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    races, limit = [], None
    args = list(argv)
    do_sweep = "--sweep" in args
    do_replay = "--replay" in args
    args = [a for a in args if a not in ("--sweep", "--replay")]
    years = []
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
        for y in years:
            races += [(y, rec) for rec in cf.season_races(y)]
        if limit:
            races = races[:limit]
    else:
        races = [(int(a.split(":")[0]), a.split(":", 1)[1]) for a in args]
    if not races:
        say("Name some races, e.g.  python BACKTEST.py 2024:Bahrain 2024:Spain   or   --years 2024 --max 6")
        return

    all1, all2, tables, loaded = [], [], [], []
    for year, name in races:
        try:
            table, label, _ = cf.load_race_openf1(year, name)
            d = cf.prepare(table)
            if d.compound.isin(["INTERMEDIATE", "WET"]).mean() > 0.03:
                say(f"\n--- {label}: wet race, skipped")
                continue
            cond_map = cf.lap_conditions(d).to_dict()
            if do_sweep:
                tables.append((d, int(d.lap.max()), cond_map))
                say(f"loaded {label}")
            loaded.append((label, d, collect_lengths(d, cond_map)))
        except Exception as problem:
            say(f"\n--- {year} {name}: {type(problem).__name__}: {str(problem)[:150]}")
    if do_replay:
        run_replay([(label, d, int(d.lap.max()), cf.lap_conditions(d).to_dict()) for label, d, _ in loaded], say)
        with open("replay_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        say("\nSaved replay_report.txt - send that file back.")
        return
    if not do_sweep:
        for i, (label, d, own) in enumerate(loaded):
            others = {c: [x for j, (_, _, ls) in enumerate(loaded) if j != i for x in ls[c]] for c in COMPOUNDS}
            ref = others if all(len(v) >= 3 for v in others.values()) else None      # one race only: own lengths
            try:
                t1, t2 = backtest_table(d, label, say, ref_lengths=ref)
                all1 += t1
                all2 += t2
            except Exception as problem:
                say(f"\n--- {label}: {type(problem).__name__}: {str(problem)[:150]}")
    if do_sweep:
        sweep(tables, say)
        with open("sweep_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        say("\nSaved sweep_report.txt - send that file back.")
        return
    say("\n" + "=" * 70 + "\nALL RACES TOGETHER")
    report_stop_timing(all1, say)
    report_live(all2, say)
    with open("backtest_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    say("\nSaved backtest_report.txt - send that file back.")


# ------------------------------------------------------------------ self-test
def selftest():
    """A race the MODEL generated itself (with a +4 s circuit offset). The model must predict its own stops exactly."""
    from SAFETY_CAR_VSC import simulate_stops
    print("SELF-TEST: a race made by the model itself; it must find its own stops and read the situation sensibly\n")
    race_laps = 57
    start = RaceState(driver="NOR", compound="MEDIUM", tyre_age=0, lap=0, race_laps=race_laps, compounds_used={"MEDIUM"})
    n = race_laps
    found = best_plan(start, ["GREEN"] * n, range(0, n), False, 1.0)
    stops = found[1]
    times = simulate_stops(start, ["GREEN"] * n, stops, 1.0)
    print("plan the model chose:", [(s[0] + 1, s[1]) for s in stops], "(lap on which the new set goes on)")
    stop_lap = {off + 1: comp for off, comp in stops}
    rows, comp, age, stint, used = [], "MEDIUM", 0, 1, {"MEDIUM"}
    for lap in range(1, n + 1):
        if lap in stop_lap:
            comp, age, stint = stop_lap[lap], 0, stint + 1
        rows.append({"driver": "NOR", "lap": lap, "time_s": times[lap - 1] + 4.0, "end_s": 0.0, "compound": comp,
                     "tyre_age": age, "stint": stint, "cond": "GREEN", "pos": 1.0,
                     "pit_in": (lap + 1) in stop_lap, "pit_out": lap in stop_lap, "accurate": True, "deleted": False})
        age += 1
    d = pd.DataFrame(rows)
    lines = []
    t1, t2 = backtest_table(d, "model-made race", lambda t="": lines.append(t))
    bad = 0
    print("TEST 1 results:")
    for r in t1:
        ok = r["real"] == r["model"]
        bad += not ok
        print(f"   stint starting lap {r['start']}: model says stop on lap {r['model']}, real {r['real']}  {'OK' if ok else 'CHECK'}")
    if not t1:
        print("   no stints to test (the plan had no stops after the first)"); bad += 1
    report_live(t2, print)
    clear_pit = [r for r in t2 if r["truth"] == "now" and r["decision"] == "STAY"]
    clear_stay = [r for r in t2 if r["truth"] == "later" and r["decision"] == "PIT"]
    print(f"\n   wrong-way calls: STAY just before a stop {len(clear_pit)}, PIT with no stop near {len(clear_stay)}")
    bad += (len(clear_pit) + len(clear_stay)) > max(1, len(t2) // 5)
    # replay: three model-made drivers with different strategies; the replay must reproduce them exactly
    from PIT_STRATEGY import simulate_strategy_laps
    plans = {"VER": [("MEDIUM", 20), ("HARD", 22), ("MEDIUM", 15)], "NOR": [("HARD", 28), ("MEDIUM", 29)],
             "LEC": [("MEDIUM", 18), ("HARD", 21), ("HARD", 18)], "HAM": [("SOFT", 14), ("HARD", 24), ("HARD", 19)],
             "RUS": [("MEDIUM", 25), ("HARD", 32)], "PIA": [("HARD", 15), ("MEDIUM", 20), ("MEDIUM", 22)],
             "SAI": [("MEDIUM", 22), ("SOFT", 17), ("HARD", 18)], "ALO": [("HARD", 30), ("MEDIUM", 27)]}
    rows2 = []
    for code, plan in plans.items():
        for k, rec in enumerate(simulate_strategy_laps(plan, race_laps, driver=profile_name(code))):
            rows2.append({"driver": code, "lap": k + 1, "time_s": rec["lap_time"] + 4.0, "end_s": 0.0, "compound": rec["compound"],
                          "tyre_age": rec.get("tyre_age", 0), "stint": 0, "cond": "GREEN", "pos": 1.0, "pit_in": False,
                          "pit_out": False, "accurate": True, "deleted": False})
    d2 = pd.DataFrame(rows2)
    # give each driver's laps stint numbers from his plan
    for code, plan in plans.items():
        stints, k = [], 0
        for n_, (c_, laps_) in enumerate(plan, start=1):
            stints += [n_] * laps_
        d2.loc[d2.driver == code, "stint"] = stints
    rr, why = replay_race(d2, race_laps, {l: "GREEN" for l in range(1, race_laps + 1)}, 4.0)
    if rr is None:
        print("replay self-test: could not replay:", why); bad += 1
    else:
        sm = replay_summary(rr)
        ok = sm["gap_err"] < 0.01 and sm["total_err"] < 0.01
        bad += not ok
        print(f"replay of model-made drivers: gap error {sm['gap_err']:.4f} s, time error {sm['total_err']:.4f} s  {'OK' if ok else 'CHECK'}")
    print("\n" + ("SELF-TEST PASSED" if bad == 0 else f"SELF-TEST: {bad} check(s) need a look"))
    return bad


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(1 if selftest() else 0)
    main(sys.argv[1:])