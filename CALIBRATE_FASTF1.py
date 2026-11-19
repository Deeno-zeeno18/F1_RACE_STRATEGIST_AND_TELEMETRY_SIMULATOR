"""
CALIBRATE_FASTF1.py - turns real races into the numbers that replace our placeholders.

RUN IT ON YOUR OWN COMPUTER (it needs to download race data):
    pip install fastf1
    python CALIBRATE_FASTF1.py                              # a default set of 2024 races
    python CALIBRATE_FASTF1.py 2024:Bahrain 2024:Monza      # your own choice (year:race)
    python CALIBRATE_FASTF1.py --selftest                   # no internet: checks this script on a fake race
    python CALIBRATE_FASTF1.py --diagnose                   # if the download fails: which website is blocked?

It writes calibration.json. Send that file back.

WHAT IT MEASURES, AND WHICH PLACEHOLDER IT REPLACES
  compound pace offsets     compound_pace_offset  (SOFT -0.7, HARD +0.5 s per lap vs MEDIUM)
  fuel effect               max_penalty 2.5 s for a full tank
  tyre degradation          (seconds per lap of tyre age, per compound) and real stint lengths
  safety car / VSC          lap_mult 1.45 / 1.30, pit loss 11 s / 15 s, green pit loss 22 s
  bunching                  BUNCH_GAP 1.0 s (gap between cars behind a safety car)
  drivers                   pace and tyre_care for every driver in those races

HOW THE PACE NUMBERS ARE FOUND (one regression per race, on clean laps only):
  lap time = driver's own pace + compound offset + fuel/track-evolution slope per lap
             + degradation slope per lap of tyre age (separate for each compound)
A clean lap is a green-flag lap that is not lap 1, not an in-lap or out-lap, is timed accurately,
and is within 4% of that driver's median. Wet races are skipped.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
import os
import sys

import numpy as np
import pandas as pd

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
MIN_DRIVER_LAPS = 150      # a driver needs this many clean laps (over all the races) to get his own numbers

DEFAULT_RACES = [(2024, "Bahrain"), (2024, "Australia"), (2024, "Spain"), (2024, "Hungary"),
                 (2024, "Monza"), (2024, "Suzuka"), (2024, "Canada"), (2024, "Silverstone")]

# the placeholders currently in the model, for the comparison table at the end
PLACEHOLDERS = {"SOFT": -0.7, "HARD": 0.5, "fuel_full_tank": 2.5, "pit_GREEN": 22.0, "pit_SC": 11.0,
                "pit_VSC": 15.0, "ratio_SC": 1.45, "ratio_VSC": 1.30, "bunch": 1.0}


# ======================================================================= preparing the table
def lap_category(status):
    """GREEN / SC / VSC for a lap that was entirely under that condition, otherwise OTHER."""
    if not isinstance(status, str) or not status:
        return "OTHER"
    codes = set(status)
    if codes == {"1"}:
        return "GREEN"
    if "4" in codes and codes <= {"2", "4"}:
        return "SC"
    if "6" in codes and codes <= {"2", "6"}:
        return "VSC"
    return "OTHER"


def prepare(laps):
    """FastF1's laps table -> a plain table with the few columns we need, in seconds."""
    def seconds(col):
        return pd.to_timedelta(laps[col]).dt.total_seconds().to_numpy()

    deleted = laps["Deleted"].fillna(False).astype(bool).to_numpy() if "Deleted" in laps else np.zeros(len(laps), bool)
    d = pd.DataFrame({
        "driver": laps["Driver"].astype(str).to_numpy(),
        "lap": laps["LapNumber"].astype(float).to_numpy(),
        "time_s": seconds("LapTime"),
        "end_s": seconds("Time"),
        "compound": laps["Compound"].astype(str).str.upper().to_numpy(),
        "tyre_age": laps["TyreLife"].astype(float).to_numpy() - 1.0,      # age 0 = first lap on a new set
        "stint": laps["Stint"].astype(float).to_numpy(),
        "cond": [lap_category(s) for s in laps["TrackStatus"]],
        "pos": laps["Position"].astype(float).to_numpy(),
        "pit_in": laps["PitInTime"].notna().to_numpy(),
        "pit_out": laps["PitOutTime"].notna().to_numpy(),
        "accurate": laps["IsAccurate"].fillna(False).astype(bool).to_numpy(),
        "deleted": deleted,
    })
    d = d.dropna(subset=["lap"]).copy()
    d["lap"] = d["lap"].astype(int)
    return d.sort_values(["driver", "lap"]).reset_index(drop=True)


def clean_laps(d):
    """Green-flag, normal, accurately timed laps on dry tyres, without outliers."""
    ok = ((d.cond == "GREEN") & (d.lap > 1) & ~d.pit_in & ~d.pit_out & d.accurate & ~d.deleted
          & d.compound.isin(COMPOUNDS) & d.time_s.notna() & d.tyre_age.notna())
    c = d[ok].copy()
    med = c.groupby("driver")["time_s"].transform("median")
    return c[c.time_s <= 1.04 * med].copy()


def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


def summarize(values):
    if len(values) == 0:
        return None
    a = np.asarray(values, float)
    return {"median": round(float(np.median(a)), 3), "p25": round(float(np.percentile(a, 25)), 3),
            "p75": round(float(np.percentile(a, 75)), 3), "n": int(len(a))}


# ============================================================================ the pace model
def _solve(keep, drivers, present, care_row, ref="MEDIUM"):
    """
    Least squares for: lap time = driver + compound offset (vs MEDIUM) + fuel slope per lap
                                  + degradation slope per compound x (tyre age x that driver's care).
    Returns the design matrix, the coefficients (by name) and the residuals.
    """
    cols, names = [], []
    for dr in drivers:
        cols.append((keep.driver == dr).to_numpy(float)); names.append(("driver", dr))
    for k in present:
        if k != ref:
            cols.append((keep.compound == k).to_numpy(float)); names.append(("offset", k))
    cols.append((keep.lap - keep.lap.mean()).to_numpy(float)); names.append(("fuel", None))
    for k in present:
        cols.append(((keep.compound == k) * keep.tyre_age * care_row).to_numpy(float)); names.append(("deg", k))
    X = np.column_stack(cols)
    y = keep.time_s.to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return X, y, beta, names


def fit_pace_model(c, min_laps=40, rounds=8):
    """
    The regression for one race. A driver's PACE and his TYRE CARE are fitted together: if they were
    fitted separately, a driver who is gentle on tyres would look faster than he is (his tyres wear
    less than the common model says, and the difference would be hidden in his base pace).
    Returns None if the race cannot support it (fewer than two compounds with enough laps).
    Offsets are against MEDIUM when the race has it, otherwise against another compound (see out['ref']).
    """
    present = [k for k in COMPOUNDS if (c.compound == k).sum() >= min_laps]
    if len(present) < 2:
        return None
    ref = "MEDIUM" if "MEDIUM" in present else present[-1]      # offsets are measured against this compound
    keep = c[c.compound.isin(present)]
    keep = keep[keep.groupby("driver")["time_s"].transform("size") >= 15]
    drivers = sorted(keep.driver.unique())
    care = {dr: 1.0 for dr in drivers}

    X, y, beta, names = _solve(keep, drivers, present, np.ones(len(keep)), ref)
    slopes = {key: b for (kind, key), b in zip(names, beta) if kind == "deg"}
    fit_care = all(slopes[k] > 0.01 for k in present)

    if fit_care:                                     # alternate: care given slopes, then slopes given care
        for _ in range(rounds):
            parts = {(kind, key): b for (kind, key), b in zip(names, beta)}
            slope_row = keep.compound.map(parts_slopes(parts, present)).to_numpy(float)
            x_all = slope_row * keep.tyre_age.to_numpy(float)
            other = y - X[:, :-len(present)] @ beta[:-len(present)]          # lap time minus everything but degradation
            for dr in drivers:
                m = (keep.driver == dr).to_numpy()
                den = float((x_all[m] ** 2).sum())
                if den > 0:
                    care[dr] = float((x_all[m] * other[m]).sum() / den)
            mean_care = float(np.mean([care[dr] for dr in drivers]))          # keep the average driver at 1.0
            care = {dr: v / mean_care for dr, v in care.items()}
            X, y, beta, names = _solve(keep, drivers, present, keep.driver.map(care).to_numpy(float), ref)

    resid = y - X @ beta
    dof = max(len(y) - np.linalg.matrix_rank(X), 1)
    sigma2 = float(resid @ resid / dof)
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))

    out = {"n_laps": int(len(y)), "ref": ref, "offset": {}, "offset_se": {}, "deg": {}, "deg_se": {},
           "driver_fe": {}, "sigma": float(np.sqrt(sigma2))}
    for (kind, key), b, s in zip(names, beta, se):
        if kind == "offset":
            out["offset"][key], out["offset_se"][key] = float(b), float(s)
        elif kind == "deg":
            out["deg"][key], out["deg_se"][key] = float(b), float(s)
        elif kind == "fuel":
            out["fuel_per_lap"], out["fuel_se"] = float(b), float(s)
        else:
            out["driver_fe"][key] = float(b)

    field = float(np.median(list(out["driver_fe"].values())))
    age = keep.tyre_age.to_numpy(float)
    slope_row = keep.compound.map(out["deg"]).to_numpy(float)
    drv = {}
    for dr in drivers:
        m = (keep.driver == dr).to_numpy()
        info = float(((slope_row[m] * age[m]) ** 2).sum())       # how much tyre wear this driver's laps actually show
        drv[dr] = {"pace": round(out["driver_fe"][dr] - field, 3), "care_raw": round(care[dr], 4),
                   "info": round(info, 3), "n": int(m.sum())}
    out["drivers"] = drv
    out["care_fitted"] = bool(fit_care)
    return out


def parts_slopes(parts, present):
    return {k: parts[("deg", k)] for k in present}


# ================================================================ safety car, pit loss, bunching
def lap_conditions(d):
    """What the track was doing on each lap number (the most common category across the cars)."""
    return d.groupby("lap")["cond"].agg(lambda s: s.mode().iat[0])


def track_effects(d, c, bunch_override=None):
    cond = lap_conditions(d)
    prev = cond.shift(1)
    steady = {k: set(cond.index[(cond == k) & (prev == k)]) for k in ("SC", "VSC")}
    nopit = ~d.pit_in & ~d.pit_out & d.time_s.notna()
    green_ref = float(c.time_s.median())

    out = {"ratio": {}, "ref_time": {}, "n_laps": {}}
    for k in ("SC", "VSC"):
        rows = d[nopit & d.lap.isin(steady[k]) & (d.cond == k)]
        out["n_laps"][k] = int(len(rows))
        if len(rows) >= 10:
            out["ref_time"][k] = float(rows.time_s.median())
            out["ratio"][k] = out["ref_time"][k] / green_ref

    # pit stops: in-lap + out-lap against two ordinary laps
    losses = {"GREEN": [], "SC": [], "VSC": []}
    by_driver = {k: g for k, g in d.groupby("driver")}
    clean_by_driver = {k: g for k, g in c.groupby("driver")}
    for dr, g in by_driver.items():
        g = g.set_index("lap")
        for lap in g.index[g.pit_in]:
            if lap + 1 not in g.index or not g.loc[lap + 1, "pit_out"]:
                continue
            a, b = g.loc[lap], g.loc[lap + 1]
            if a.cond != b.cond or a.cond not in losses or pd.isna(a.time_s) or pd.isna(b.time_s):
                continue
            if a.cond == "GREEN":
                cd = clean_by_driver.get(dr)
                if cd is None:
                    continue
                before = cd[cd.lap < lap].time_s.tail(3)
                after = cd[cd.lap > lap + 1].time_s.head(3)
                if len(before) == 0 or len(after) == 0:
                    continue
                ref = 0.5 * (before.median() + after.median())
                ceiling = 60
            else:
                if a.cond not in out["ref_time"]:
                    continue
                ref, ceiling = out["ref_time"][a.cond], 45
            loss = a.time_s + b.time_s - 2 * ref
            if 0 < loss < ceiling:
                losses[a.cond].append(float(loss))
    out["pit_loss"] = losses

    # bunching: time gap to the car ahead, for cars in P2..P12 on consecutive positions
    def gaps(laps_wanted):
        found = []
        for lap in laps_wanted:
            rows = d[(d.lap == lap) & ~d.pit_in & ~d.pit_out & d.pos.notna() & d.end_s.notna()].sort_values("pos")
            p, t = rows.pos.to_numpy(), rows.end_s.to_numpy()
            for i in range(1, len(rows)):
                if p[i] - p[i - 1] == 1 and p[i] <= 12:
                    found.append(float(t[i] - t[i - 1]))
        return found
    if bunch_override is not None:                      # the data source gave the gaps directly (OpenF1)
        out["bunch"] = bunch_override
    else:
        green_laps = [lap for lap in cond.index[cond == "GREEN"] if lap >= 5]
        out["bunch"] = {"SC": gaps(sorted(steady["SC"])), "GREEN": gaps(green_laps)}
    return out


def stint_lengths(d):
    """Laps on each set, for stints that ended in a pit stop under green (not the last stint, not a flag stop)."""
    found = {k: [] for k in COMPOUNDS}
    for dr, g in d.groupby("driver"):
        stints = sorted(g.stint.dropna().unique())
        for s in stints[:-1]:
            rows = g[g.stint == s]
            last = rows.iloc[-1]
            comp = rows.compound.mode().iat[0]
            if comp in found and last.pit_in and last.cond == "GREEN":
                found[comp].append(int(len(rows)))
    return found


# ===================================================================== one race, many races
def analyze_race(laps, name="race", extra=None):
    d = prepare(laps)
    wet = d.compound.isin(["INTERMEDIATE", "WET"]).mean()
    if wet > 0.03:
        return {"skipped": f"{name}: wet race ({wet:.0%} of laps on wet tyres)"}
    c = clean_laps(d)
    model = fit_pace_model(c)
    if model is None:
        return {"skipped": f"{name}: not enough clean laps on enough compounds"}
    return {"event": name, "race_laps": int(d.lap.max()), "clean_laps": int(len(c)),
            "pace": model, "track": track_effects(d, c, (extra or {}).get("bunch")),
            "stints": stint_lengths(d), "info": (extra or {}).get("info", {})}


def load_race(year, name):
    import fastf1                                          # only needed when running on real data
    os.makedirs("ff1_cache", exist_ok=True)
    fastf1.Cache.enable_cache("ff1_cache")
    session = fastf1.get_session(year, name, "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return session.laps, f"{year} {session.event['EventName']}"


def combine_offsets(used):
    """
    SOFT and HARD pace against MEDIUM, from every race. A race with MEDIUM gives them directly. A race without
    MEDIUM (Bahrain 2024 had almost none) only gives the difference between the two compounds it has; that
    difference is used as one more equation in a least-squares fit for (SOFT, HARD) with MEDIUM = 0.
    """
    rows, rhs, direct = [], [], {"SOFT": [], "HARD": []}
    idx = {"SOFT": 0, "HARD": 1}
    for r in used:
        off = dict(r["pace"]["offset"])
        ref = r["pace"].get("ref", "MEDIUM")
        off[ref] = 0.0
        if "MEDIUM" in off:
            for k in ("SOFT", "HARD"):
                if k in off:
                    row = [0.0, 0.0]; row[idx[k]] = 1.0
                    rows.append(row); rhs.append(off[k] - off["MEDIUM"]); direct[k].append(off[k] - off["MEDIUM"])
        elif "SOFT" in off and "HARD" in off:
            rows.append([-1.0, 1.0]); rhs.append(off["HARD"] - off["SOFT"])
    out = {"SOFT": None, "HARD": None}
    if rows and np.linalg.matrix_rank(np.array(rows)) == 2:
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        for k in ("SOFT", "HARD"):
            out[k] = {"median": round(float(sol[idx[k]]), 3), "per_race": [round(v, 3) for v in direct[k]]}
    else:
        for k in ("SOFT", "HARD"):
            if direct[k]:
                out[k] = {"median": round(float(np.median(direct[k])), 3), "per_race": [round(v, 3) for v in direct[k]]}
    return out


def aggregate(results):
    used = [r for r in results if "skipped" not in r]
    skipped = [r["skipped"] for r in results if "skipped" in r]
    if not used:
        return {"races_used": [], "races_skipped": skipped, "error": "no race could be analysed"}

    def over_races(getter):
        vals = [getter(r) for r in used]
        vals = [v for v in vals if v is not None]
        return {"median": round(float(np.median(vals)), 3), "per_race": [round(v, 3) for v in vals]} if vals else None

    cal = {"races_used": [r["event"] for r in used], "races_skipped": skipped}
    cal["compound_pace_offset"] = {"MEDIUM": 0.0}
    cal["compound_pace_offset"].update(combine_offsets(used))
    cal["fuel_seconds_per_lap"] = over_races(lambda r: r["pace"]["fuel_per_lap"])
    cal["fuel_full_tank_penalty"] = over_races(lambda r: -r["pace"]["fuel_per_lap"] * r["race_laps"])
    cal["degradation_per_lap"] = {k: over_races(lambda r, k=k: r["pace"]["deg"].get(k)) for k in COMPOUNDS}
    cal["stint_length"] = {k: summarize([n for r in used for n in r["stints"][k]]) for k in COMPOUNDS}

    cal["lap_time_ratio"] = {k: over_races(lambda r, k=k: r["track"]["ratio"].get(k)) for k in ("SC", "VSC")}
    cal["track_laps_found"] = {k: sum(r["track"]["n_laps"][k] for r in used) for k in ("SC", "VSC")}
    cal["pit_loss"] = {k: summarize([x for r in used for x in r["track"]["pit_loss"][k]]) for k in ("GREEN", "SC", "VSC")}
    cal["bunch_gap"] = {k: summarize([x for r in used for x in r["track"]["bunch"][k]]) for k in ("SC", "GREEN")}

    # drivers: pace is laps-weighted; tyre care is pooled by how much wear each race showed, then shrunk
    # toward 1.0 ONCE (prior spread of 0.15, against this race's lap-time noise)
    pooled, sigma2 = {}, float(np.mean([r["pace"]["sigma"] ** 2 for r in used]))
    for r in used:
        for dr, v in r["pace"]["drivers"].items():
            pooled.setdefault(dr, []).append(v)
    lam = sigma2 / 0.15 ** 2
    drivers = {}
    for dr, rows in pooled.items():
        n = sum(v["n"] for v in rows)
        if n >= MIN_DRIVER_LAPS:
            info = sum(v["info"] for v in rows)
            care = (sum(v["info"] * v["care_raw"] for v in rows) + lam) / (info + lam)
            drivers[dr] = {"pace": round(sum(v["pace"] * v["n"] for v in rows) / n, 3),
                           "tyre_care": round(care, 3), "clean_laps": n, "races": len(rows)}
    cal["drivers"] = dict(sorted(drivers.items(), key=lambda kv: kv[1]["pace"]))

    def med(x):
        return None if x is None else x["median"]
    cal["suggested"] = {
        "compound_pace_offset": {k: (0.0 if k == "MEDIUM" else med(cal["compound_pace_offset"][k])) for k in COMPOUNDS},
        "fuel_full_tank_penalty": med(cal["fuel_full_tank_penalty"]),
        "pit_loss": {k: med(cal["pit_loss"][k]) for k in ("GREEN", "SC", "VSC")},
        "lap_mult": {k: med(cal["lap_time_ratio"][k]) for k in ("SC", "VSC")},
        "bunch_gap": med(cal["bunch_gap"]["SC"]),
    }
    return cal


def print_report(cal):
    if "error" in cal:
        print("\n" + cal["error"]); [print("  skipped:", s) for s in cal["races_skipped"]]
        return
    s = cal["suggested"]
    print(f"\nRaces used: {len(cal['races_used'])}   skipped: {len(cal['races_skipped'])}")
    for line in cal["races_skipped"]:
        print("   skipped:", line)
    print(f"Safety-car laps found: {cal['track_laps_found']['SC']}   VSC laps found: {cal['track_laps_found']['VSC']}")
    rows = [("SOFT offset vs MEDIUM (s/lap)", PLACEHOLDERS["SOFT"], s["compound_pace_offset"]["SOFT"]),
            ("HARD offset vs MEDIUM (s/lap)", PLACEHOLDERS["HARD"], s["compound_pace_offset"]["HARD"]),
            ("fuel penalty, full tank (s)", PLACEHOLDERS["fuel_full_tank"], s["fuel_full_tank_penalty"]),
            ("pit loss, green (s)", PLACEHOLDERS["pit_GREEN"], s["pit_loss"]["GREEN"]),
            ("pit loss, safety car (s)", PLACEHOLDERS["pit_SC"], s["pit_loss"]["SC"]),
            ("pit loss, VSC (s)", PLACEHOLDERS["pit_VSC"], s["pit_loss"]["VSC"]),
            ("safety-car lap time, x green", PLACEHOLDERS["ratio_SC"], s["lap_mult"]["SC"]),
            ("VSC lap time, x green", PLACEHOLDERS["ratio_VSC"], s["lap_mult"]["VSC"]),
            ("gap behind safety car (s)", PLACEHOLDERS["bunch"], s["bunch_gap"])]
    print(f"\n{'':34}{'placeholder':>12}{'measured':>12}")
    for name, old, new in rows:
        print(f"{name:34}{old:>12}{('n/a' if new is None else new):>12}")
    print("\ndegradation (s per lap of tyre age):", {k: (v or {}).get("median") for k, v in cal["degradation_per_lap"].items()})
    print("stint length (median laps):", {k: (v or {}).get("median") for k, v in cal["stint_length"].items()})
    if cal["drivers"]:
        names = list(cal["drivers"])
        print(f"drivers measured: {len(names)}  (fastest {names[0]}, slowest {names[-1]})")


# ============================================================================== the self-test
PLANT = {"offset": {"SOFT": -0.7, "MEDIUM": 0.0, "HARD": 0.5}, "deg": {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04},
         "fuel": -0.04, "base": 92.0, "SC": 1.45, "VSC": 1.30, "pit": {"GREEN": 22.0, "SC": 11.0, "VSC": 15.0}, "bunch": 0.9}


def make_synthetic_race(seed=1, n_drivers=20, n_laps=57):
    """A fake race in FastF1's table layout, built from the numbers in PLANT. Used to test the analysis."""
    rng = np.random.default_rng(seed)
    names = [f"D{i:02d}" for i in range(n_drivers)]
    pace = {n: float(rng.normal(0, 0.35)) for n in names}
    care = {n: float(np.clip(rng.normal(1, 0.10), 0.75, 1.3)) for n in names}
    pace["D00"], care["D00"] = -0.30, 0.80          # a fast, gentle driver
    pace["D01"], care["D01"] = 0.40, 1.25           # a slow, hard-on-tyres driver

    status = {n: "1" for n in range(1, n_laps + 1)}
    status.update({20: "14", 21: "4", 22: "4", 23: "4", 24: "41", 40: "16", 41: "6", 42: "6", 43: "67"})
    sc_common = {n: PLANT["SC"] * PLANT["base"] + rng.normal(0, 0.8) for n in (21, 22, 23)}
    vsc_common = {n: PLANT["VSC"] * PLANT["base"] + rng.normal(0, 0.6) for n in (41, 42)}

    plans = {}
    for n in names:
        stops = []
        if rng.random() < 0.5:
            stops.append((22, "SC"))
        else:
            stops.append((int(rng.choice([l for l in range(12, 36) if not 18 <= l <= 25])), "GREEN"))
        if rng.random() < 0.3:
            stops.append((41, "VSC"))
        comps = [str(rng.choice(COMPOUNDS))]
        for _ in stops:
            comps.append(str(rng.choice([k for k in COMPOUNDS if k != comps[-1]])))
        plans[n] = (stops, comps)

    rows, cum = [], {n: 0.0 for n in names}
    for lap in range(1, n_laps + 1):
        this = {}
        for n in names:
            stops, comps = plans[n]
            in_laps = [s[0] for s in stops]
            stint = sum(1 for l in in_laps if l < lap)
            start = (in_laps[stint - 1] + 1) if stint > 0 else 1
            age = lap - start
            comp = comps[stint]
            cat = lap_category(status[lap])
            normal = (PLANT["base"] + PLANT["offset"][comp] + pace[n] + PLANT["fuel"] * lap
                      + PLANT["deg"][comp] * care[n] * age + rng.normal(0, 0.12))
            if cat == "SC":
                t = sc_common[lap] + rng.normal(0, 0.3)
            elif cat == "VSC":
                t = vsc_common[lap] + rng.normal(0, 0.3)
            elif status[lap] in ("14", "41", "16", "67"):
                t = 0.5 * (normal + PLANT["SC"] * PLANT["base"]) + rng.normal(0, 1.0)
            else:
                t = normal
            pit_in = pit_out = False
            for in_lap, kind in stops:
                if lap == in_lap:
                    t += PLANT["pit"][kind] / 2; pit_in = True
                if lap == in_lap + 1:
                    t += PLANT["pit"][kind] / 2; pit_out = True
            this[n] = (t, comp, age, stint, pit_in, pit_out)
            cum[n] += t
        if lap in (21, 22, 23, 41, 42):                          # the field closes up behind the safety car
            order = sorted(names, key=lambda n: cum[n])
            if lap in (21, 22, 23):
                lead = cum[order[0]]
                for i, n in enumerate(order):
                    cum[n] = lead + i * PLANT["bunch"] + float(rng.normal(0, 0.08)) * (i > 0)
        order = sorted(names, key=lambda n: cum[n])
        position = {n: i + 1 for i, n in enumerate(order)}
        for n in names:
            t, comp, age, stint, pit_in, pit_out = this[n]
            rows.append({"Driver": n, "LapNumber": float(lap), "LapTime": pd.to_timedelta(t, unit="s"),
                         "Time": pd.to_timedelta(cum[n], unit="s"), "Stint": float(stint + 1), "Compound": comp,
                         "TyreLife": float(age + 1), "TrackStatus": status[lap],
                         "PitInTime": pd.to_timedelta(cum[n], unit="s") if pit_in else pd.NaT,
                         "PitOutTime": pd.to_timedelta(cum[n], unit="s") if pit_out else pd.NaT,
                         "IsAccurate": True, "Position": float(position[n]), "Deleted": False})
    return pd.DataFrame(rows), pace, care


def selftest():
    print("SELF-TEST: a fake race with KNOWN answers. The extractor must find them again.\n")
    results = []
    for seed in range(1, 7):
        laps, pace, care = make_synthetic_race(seed)
        results.append(analyze_race(laps, f"fake race {seed}"))
    cal = aggregate(results)
    s = cal["suggested"]
    checks = [("SOFT offset", PLANT["offset"]["SOFT"], s["compound_pace_offset"]["SOFT"], 0.12),
              ("HARD offset", PLANT["offset"]["HARD"], s["compound_pace_offset"]["HARD"], 0.12),
              ("fuel full tank (-slope x 57 laps)", -PLANT["fuel"] * 57, s["fuel_full_tank_penalty"], 0.35),
              ("degradation SOFT", PLANT["deg"]["SOFT"], cal["degradation_per_lap"]["SOFT"]["median"], 0.03),
              ("degradation MEDIUM", PLANT["deg"]["MEDIUM"], cal["degradation_per_lap"]["MEDIUM"]["median"], 0.03),
              ("degradation HARD", PLANT["deg"]["HARD"], cal["degradation_per_lap"]["HARD"]["median"], 0.03),
              ("pit loss green", PLANT["pit"]["GREEN"], s["pit_loss"]["GREEN"], 1.0),
              ("pit loss safety car", PLANT["pit"]["SC"], s["pit_loss"]["SC"], 1.5),
              ("pit loss VSC", PLANT["pit"]["VSC"], s["pit_loss"]["VSC"], 1.5),
              ("safety-car lap time ratio", PLANT["SC"], s["lap_mult"]["SC"], 0.04),
              ("VSC lap time ratio", PLANT["VSC"], s["lap_mult"]["VSC"], 0.04),
              ("gap behind safety car", PLANT["bunch"], s["bunch_gap"], 0.15)]
    bad = 0
    print(f"{'measure':36}{'planted':>10}{'found':>10}   verdict")
    for name, planted, found, tol in checks:
        ok = found is not None and abs(found - planted) <= tol
        bad += not ok
        print(f"{name:36}{planted:>10.3f}{('n/a' if found is None else format(found, '.3f')):>10}   {'OK' if ok else 'CHECK'} (tolerance {tol})")

    # drivers: D00 was planted fast and gentle, D01 slow and hard on tyres
    dr = cal["drivers"]
    print("\nplanted drivers (pace is against the field median, so compare the sign and rough size):")
    for n, p, c in (("D00", -0.30, 0.80), ("D01", 0.40, 1.25)):
        v = dr.get(n)
        if v is None:
            print(f"   {n}: not measured (needs {MIN_DRIVER_LAPS} clean laps in total)"); bad += 1; continue
        print(f"   {n}: planted pace {p:+.2f}, care {c:.2f}   found pace {v['pace']:+.2f}, care {v['tyre_care']:.2f}")
        bad += not (abs(v["pace"] - p) < 0.25 and (v["tyre_care"] - 1) * (c - 1) > 0)
    print("\nstint length (median laps):", {k: (v or {}).get("median") for k, v in cal["stint_length"].items()})
    print("\n" + ("SELF-TEST PASSED" if bad == 0 else f"SELF-TEST: {bad} check(s) need a look"))
    return bad


# ===================================================================================== OpenF1 loader
# OpenF1 (api.openf1.org) is free and needs no login. We ask it for one race and rebuild the SAME lap table
# that FastF1 would have given, so everything above this line works unchanged.
OPENF1 = "https://api.openf1.org/v1/"
MAX_PER_SECOND, MAX_PER_MINUTE = 3, 30          # OpenF1's published limits for the free tier
_sent = []                                      # when each request went out (for the rate limit)


def _pause_for_rate_limit():
    import time
    while True:
        now = time.time()
        _sent[:] = [t for t in _sent if now - t < 60]
        recent_second = [t for t in _sent if now - t < 1.0]
        if len(_sent) >= MAX_PER_MINUTE - 2:
            time.sleep(max(0.5, 60 - (now - _sent[0]) + 0.1))
        elif len(recent_second) >= MAX_PER_SECOND - 1:
            time.sleep(0.5)
        else:
            _sent.append(time.time())
            return


def http_get_json(url):
    """One polite GET that returns parsed JSON. Retries when OpenF1 says 'too many requests'. (Tests replace this.)"""
    import time
    import urllib.error
    import urllib.request
    last = None
    for attempt in range(5):
        _pause_for_rate_limit()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "f1-strategy-simulator", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as problem:
            last = problem
            if problem.code in (429, 500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            if problem.code == 404:                      # OpenF1 answers 404 when a filter matches nothing
                return []
            raise
        except (urllib.error.URLError, TimeoutError) as problem:
            last = problem
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"OpenF1 did not answer after 5 tries: {last}")


def openf1(endpoint, **params):
    """Ask OpenF1 for a list of records. Answers are cached on disk so a re-run does not ask again."""
    import hashlib
    from urllib.parse import urlencode
    url = OPENF1 + endpoint + ("?" + urlencode(params) if params else "")
    os.makedirs("openf1_cache", exist_ok=True)
    path = os.path.join("openf1_cache", hashlib.md5(url.encode()).hexdigest() + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    data = http_get_json(url)
    if not isinstance(data, list):                      # e.g. {"detail": "..."}: an error message, not records
        raise RuntimeError(f"OpenF1 said: {str(data)[:150]}")
    if data or endpoint == "sessions":
        with open(path, "w") as f:
            json.dump(data, f)
    return data


def find_race_session(year, name):
    """The session record of the race at `name` (matched against country, place and circuit)."""
    sessions = openf1("sessions", year=year, session_name="Race")
    key = name.strip().lower()
    fields = ("country_name", "location", "circuit_short_name", "meeting_name")
    exact = [s for s in sessions if any(str(s.get(f, "")).lower() == key for f in fields)]
    found = exact or [s for s in sessions if any(key in str(s.get(f, "")).lower() for f in fields)]
    if len(found) == 1:
        return found[0]
    names = sorted({f"{s.get('country_name')}/{s.get('location')}" for s in (found or sessions)})
    if not found:
        raise ValueError(f"no {year} race matches '{name}'. Races OpenF1 has: {', '.join(names)}")
    raise ValueError(f"'{name}' matches several {year} races ({', '.join(names)}): use a more specific name")


def _ts(values):
    return pd.to_datetime(pd.Series(values, dtype="object"), utc=True, errors="coerce", format="ISO8601")


def track_intervals(race_control):
    """
    Safety car / VSC periods from the race-control messages: [(start, end, '4' or '6'), ...] in UTC.
    OpenF1 does not document the exact wording, so this looks for the words and reports what it saw.
    """
    msgs = sorted(((pd.Timestamp(m["date"]), str(m.get("message", "")).upper(),
                    str(m.get("flag") or "").upper() if str(m.get("scope") or "TRACK").upper() == "TRACK" else "",
                    str(m.get("category") or "").upper()) for m in race_control if m.get("date")), key=lambda x: x[0])
    # (a green or clear flag only counts when it is for the whole TRACK; "clear in sector 5" must not end a safety car)
    periods, seen, open_ = [], set(), None
    for when, text, flag, category in msgs:
        is_vsc = "VIRTUAL SAFETY CAR" in text or "VSC" in text.split()
        is_sc = ("SAFETY CAR" in text) and not is_vsc
        if category == "SAFETYCAR" or is_vsc or is_sc:
            seen.add(text)
        starts = "DEPLOYED" in text
        ends_vsc = is_vsc and "ENDING" in text
        green = flag in ("GREEN", "CLEAR") or "TRACK CLEAR" in text
        if starts and (is_vsc or is_sc):
            if open_:
                periods.append((open_[0], when, open_[1]))
            open_ = (when, "6" if is_vsc else "4")
        elif open_ and ((open_[1] == "6" and ends_vsc) or (green and "PIT" not in text) or "RED FLAG" in text
                        or (open_[1] == "4" and "IN THIS LAP" in text)):
            end = when + (pd.Timedelta(seconds=90) if (open_[1] == "4" and "IN THIS LAP" in text) else pd.Timedelta(0))
            periods.append((open_[0], end, open_[1]))
            open_ = None
    if open_:
        periods.append((open_[0], msgs[-1][0], open_[1]))
    return periods, sorted(seen)


def season_races(year):
    """Every race session OpenF1 has for a year, in date order."""
    sessions = openf1("sessions", year=year, session_name="Race")
    return sorted(sessions, key=lambda x: str(x.get("date_start", "")))


def load_race_openf1(year, name):
    """Build a FastF1-style laps table for one race from OpenF1. `name` is a place name, or a session record."""
    session = name if isinstance(name, dict) else find_race_session(year, name)
    key = session["session_key"]
    label = f"{year} {session.get('country_name', '?')} ({session.get('location', '')})".replace(" ()", "")
    laps = openf1("laps", session_key=key)
    stints = openf1("stints", session_key=key)
    pits = openf1("pit", session_key=key)
    control = openf1("race_control", session_key=key)
    positions = openf1("position", session_key=key)
    roster = openf1("drivers", session_key=key)
    if not laps:
        raise RuntimeError("OpenF1 has no lap data for this race")

    code = {int(r["driver_number"]): str(r.get("name_acronym") or r["driver_number"])
            for r in roster if r.get("driver_number") is not None}
    L = pd.DataFrame([x for x in laps if x.get("driver_number") is not None and x.get("lap_number") is not None])
    L["driver"] = [code.get(int(n), str(n)) for n in L["driver_number"]]
    L["start"] = _ts(L["date_start"].tolist())
    L["dur"] = pd.to_numeric(L["lap_duration"], errors="coerce")
    L = L.sort_values(["driver", "lap_number"]).reset_index(drop=True)

    # lap start / end moments. A lap ends when the next one starts; lap 1 often has no start time of its own.
    next_start = L.groupby("driver")["start"].shift(-1)
    L["end"] = next_start.fillna(L["start"] + pd.to_timedelta(L["dur"], unit="s"))
    L["start"] = L["start"].fillna(L["end"] - pd.to_timedelta(L["dur"], unit="s"))
    L = L[L["end"].notna()].copy()
    t0 = L["start"].min()

    # tyres: which stint each lap belongs to, the compound and the age
    comp, age, stint_no = {}, {}, {}
    for s in stints:
        if s.get("driver_number") is None:
            continue
        d_ = code.get(int(s["driver_number"]), str(s["driver_number"]))
        first = int(s["lap_start"]) if s.get("lap_start") is not None else 1     # a first stint may have no start lap
        last = int(s["lap_end"]) if s.get("lap_end") is not None else 10 ** 4
        for lap in range(first, min(last, 200) + 1):
            comp[(d_, lap)] = str(s.get("compound") or "UNKNOWN").upper()
            age[(d_, lap)] = int(s.get("tyre_age_at_start") or 0) + (lap - first) + 1       # TyreLife counts this lap
            stint_no[(d_, lap)] = float(s["stint_number"]) if s.get("stint_number") is not None else np.nan
    pairs = list(zip(L["driver"], L["lap_number"].astype(int)))
    L["Compound"] = [comp.get(p, "UNKNOWN") for p in pairs]
    L["TyreLife"] = [float(age.get(p, np.nan)) for p in pairs]
    L["Stint"] = [stint_no.get(p, np.nan) for p in pairs]

    # pit stops: the in-lap is the lap the pit record names; the out-lap is flagged by OpenF1 itself
    pit_in = {(code.get(int(p["driver_number"]), str(p["driver_number"])), int(p["lap_number"])) for p in pits
              if p.get("lap_number") is not None and p.get("driver_number") is not None}
    out_flag = L["is_pit_out_lap"].fillna(False).astype(bool) if "is_pit_out_lap" in L else pd.Series(False, index=L.index)
    L["pit_in_flag"] = [p in pit_in for p in pairs]
    L["pit_out_flag"] = out_flag.to_numpy() & (L["lap_number"].astype(int).to_numpy() > 1)

    # track status: '4' while a safety car is out, '6' during a VSC, otherwise '1'
    periods, sc_messages = track_intervals(control)
    status = []
    for s, e in zip(L["start"], L["end"]):
        codes = {c for (a, b, c) in periods if a < e and b > s} or {"1"}
        status.append("".join(sorted(codes)) if codes != {"1"} else "1")
    L["TrackStatus"] = status
    # a lap that was only partly under the flag is not "clean": give it the mixed string so it is called OTHER
    for i, (s, e) in enumerate(zip(L["start"], L["end"])):
        covered = sum(max(0.0, (min(b, e) - max(a, s)).total_seconds()) for (a, b, c) in periods)
        whole = (e - s).total_seconds()
        if status[i] != "1" and whole > 0 and covered < 0.85 * whole:
            status[i] = "1" + status[i]
    L["TrackStatus"] = status

    # position at the end of each lap: the last position record at or before that moment
    P = pd.DataFrame([x for x in positions if x.get("driver_number") is not None and x.get("position") is not None])
    L["Position"] = np.nan
    if not P.empty:
        P["driver"] = [code.get(int(n), str(n)) for n in P["driver_number"]]
        P["date"] = _ts(P["date"].tolist())
        P = P.dropna(subset=["date"]).sort_values("date")
        for d_, g in L.groupby("driver"):
            pg = P[P["driver"] == d_]
            if pg.empty:
                continue
            idx = np.searchsorted(pg["date"].to_numpy(), g["end"].to_numpy(), side="right") - 1
            vals = np.where(idx >= 0, pg["position"].to_numpy()[np.clip(idx, 0, None)], np.nan)
            L.loc[g.index, "Position"] = vals.astype(float)

    table = pd.DataFrame({
        "Driver": L["driver"].to_numpy(),
        "LapNumber": L["lap_number"].astype(float).to_numpy(),
        "LapTime": pd.to_timedelta(L["dur"].to_numpy(), unit="s"),
        "Time": pd.to_timedelta((L["end"] - t0).dt.total_seconds().to_numpy(), unit="s"),
        "Stint": L["Stint"].to_numpy(), "Compound": L["Compound"].to_numpy(), "TyreLife": L["TyreLife"].to_numpy(),
        "TrackStatus": L["TrackStatus"].to_numpy(),
        "PitInTime": pd.to_timedelta(np.where(L["pit_in_flag"], 1.0, np.nan), unit="s"),
        "PitOutTime": pd.to_timedelta(np.where(L["pit_out_flag"], 1.0, np.nan), unit="s"),
        "IsAccurate": L["dur"].notna().to_numpy(), "Position": L["Position"].to_numpy(), "Deleted": False})
    table = table[table["Compound"].isin(COMPOUNDS + ["INTERMEDIATE", "WET"])].copy()   # wet laps stay so wet races are spotted
    extra = {"info": {"session_key": key, "safety_car_messages": sc_messages,
                      "periods_found": [(str(a), str(b), c) for a, b, c in periods]}}
    return table, label, extra


def openf1_selftest():
    """Turns the fake FastF1-style race into OpenF1-shaped answers, loads it back through the loader, and
    checks the analysis gives the same numbers as on the original table."""
    global http_get_json
    fake, pace, care = make_synthetic_race(3)
    names = sorted(fake["Driver"].unique())
    number = {n: 10 + i for i, n in enumerate(names)}
    start = pd.Timestamp("2024-03-02T15:00:00+00:00")
    iso = lambda td: (start + td).isoformat()

    laps_json, stints_json, pit_json, pos_json = [], [], [], []
    for n in names:
        g = fake[fake.Driver == n].sort_values("LapNumber")
        prev_end = pd.Timedelta(0)
        for _, r in g.iterrows():
            lap = int(r.LapNumber)
            laps_json.append({"driver_number": number[n], "lap_number": lap,
                              "date_start": None if lap == 1 else iso(prev_end),
                              "lap_duration": r.LapTime.total_seconds(), "is_pit_out_lap": bool(pd.notna(r.PitOutTime))})
            if pd.notna(r.PitInTime):
                pit_json.append({"driver_number": number[n], "lap_number": lap, "date": iso(r.Time)})
            pos_json.append({"driver_number": number[n], "position": int(r.Position), "date": iso(r.Time - pd.Timedelta(seconds=0.5))})
            prev_end = r.Time
        for sno, sg in g.groupby("Stint"):
            stints_json.append({"driver_number": number[n], "stint_number": int(sno), "compound": sg.Compound.iloc[0],
                                "lap_start": int(sg.LapNumber.min()), "lap_end": int(sg.LapNumber.max()),
                                "tyre_age_at_start": int(sg.TyreLife.iloc[0]) - 1})
    stints_json[0]["lap_start"] = None                                    # real data has gaps like this
    laps_json.append({"driver_number": None, "lap_number": None, "lap_duration": None})
    pos_json.append({"driver_number": number[names[0]], "position": None, "date": iso(pd.Timedelta(seconds=5))})
    # race-control messages in the wording F1 uses (the real wording is exactly what the loader must discover)
    lap_end = fake.groupby("LapNumber")["Time"].median()
    control = [
        {"date": iso(lap_end[20] - pd.Timedelta(seconds=40)), "category": "SafetyCar", "message": "SAFETY CAR DEPLOYED"},
        {"date": iso(lap_end[23]), "category": "SafetyCar", "message": "SAFETY CAR IN THIS LAP"},
        {"date": iso(lap_end[40] - pd.Timedelta(seconds=30)), "category": "SafetyCar", "message": "VIRTUAL SAFETY CAR DEPLOYED"},
        {"date": iso(lap_end[42]), "category": "SafetyCar", "message": "VIRTUAL SAFETY CAR ENDING"}]
    answers = {"sessions": [{"session_key": 1, "year": 2024, "country_name": "Testland", "location": "Testville",
                             "session_name": "Race", "date_start": start.isoformat()}],
               "drivers": [{"driver_number": number[n], "name_acronym": n} for n in names],
               "laps": laps_json, "stints": stints_json, "pit": pit_json, "race_control": control, "position": pos_json}

    def fake_http(url):
        return answers[url.split("/v1/")[1].split("?")[0]]
    real_http, http_get_json = http_get_json, fake_http
    import shutil, tempfile
    here = os.getcwd()
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)
    try:
        table, label, extra = load_race_openf1(2024, "testville")
        mine, original = analyze_race(table, label, extra), analyze_race(fake, "original")
    finally:
        http_get_json = real_http
        os.chdir(here)
        shutil.rmtree(tmp, ignore_errors=True)

    print("OPENF1 LOADER SELF-TEST: fake OpenF1 answers -> lap table -> analysis, against the analysis of the original table\n")
    print("safety car messages the loader saw:", extra["info"]["safety_car_messages"])
    print("periods:", [(p[0][11:19], p[1][11:19], p[2]) for p in extra["info"]["periods_found"]])
    s, o = mine, original
    checks = [("SOFT offset", s["pace"]["offset"]["SOFT"], o["pace"]["offset"]["SOFT"], 0.05),
              ("HARD offset", s["pace"]["offset"]["HARD"], o["pace"]["offset"]["HARD"], 0.05),
              ("fuel per lap", s["pace"]["fuel_per_lap"], o["pace"]["fuel_per_lap"], 0.01),
              ("degradation MEDIUM", s["pace"]["deg"]["MEDIUM"], o["pace"]["deg"]["MEDIUM"], 0.02),
              ("SC lap ratio", s["track"]["ratio"].get("SC"), o["track"]["ratio"].get("SC"), 0.03),
              ("VSC lap ratio", s["track"]["ratio"].get("VSC"), o["track"]["ratio"].get("VSC"), 0.03),
              ("clean laps", s["clean_laps"], o["clean_laps"], 25)]
    for k in ("GREEN", "SC", "VSC"):
        a, b = s["track"]["pit_loss"][k], o["track"]["pit_loss"][k]
        checks.append((f"pit loss {k} (count)", len(a), len(b), 2))
        if a and b:
            checks.append((f"pit loss {k} (median)", float(np.median(a)), float(np.median(b)), 1.5))
    bs, bo = s["track"]["bunch"]["SC"], o["track"]["bunch"]["SC"]
    if bs and bo:
        checks.append(("gap behind safety car", float(np.median(bs)), float(np.median(bo)), 0.2))
    bad = 0
    print(f"\n{'measure':28}{'via OpenF1':>12}{'original':>12}   verdict")
    for name_, a, b, tol in checks:
        ok = a is not None and b is not None and abs(a - b) <= tol
        bad += not ok
        fmt = lambda v: "n/a" if v is None else format(v, ".3f")
        print(f"{name_:28}{fmt(a):>12}{fmt(b):>12}   {'OK' if ok else 'CHECK'} (tolerance {tol})")
    print("\n" + ("OPENF1 LOADER SELF-TEST PASSED" if bad == 0 else f"OPENF1 LOADER SELF-TEST: {bad} check(s) need a look"))
    return bad


# ===================================================================================== running
def diagnose():
    """Tests the websites race data can come from, and says which ones this computer can reach."""
    import requests
    try:
        import fastf1
        print("fastf1 version:", fastf1.__version__)
    except ImportError:
        print("fastf1 is NOT installed in this environment: run  pip install fastf1")
    base = "https://livetiming.formula1.com/static/2024/2024-03-02_Bahrain_Grand_Prix/2024-03-02_Race/"
    tests = [("the internet in general (PyPI)", "https://pypi.org/simple/fastf1/"),
             ("F1 live timing: a real race file (what FastF1 downloads)", base + "SessionInfo.json"),
             ("F1 live timing: the race's lap timing file", base + "TimingData.jsonStream"),
             ("OpenF1 (another source of the same race data)",
              "https://api.openf1.org/v1/sessions?year=2024&country_name=Bahrain&session_name=Race"),
             ("Ergast / Jolpica (race results)", "https://api.jolpi.ca/ergast/f1/2024/1.json")]
    print()
    for name, url in tests:
        print(name + "\n    " + url)
        try:
            r = requests.get(url, timeout=25)
            body = " ".join(r.text[:200].split())
            verdict = "reachable" if r.status_code == 200 else f"answered, but with an error: HTTP {r.status_code}"
            print(f"    -> {verdict}\n    -> starts with: {body[:120]!r}\n")
        except Exception as problem:
            print(f"    -> FAILED: {type(problem).__name__}: {str(problem)[:160]}\n")
    print("Send me this whole printout.")


def inspect_openf1(year, name):
    """Prints what each stage of the OpenF1 loader found, to see where laps are being lost."""
    session = find_race_session(year, name)
    key = session["session_key"]
    print("session:", {k: session.get(k) for k in ("session_key", "country_name", "location", "date_start")})
    laps = openf1("laps", session_key=key)
    stints = openf1("stints", session_key=key)
    control = openf1("race_control", session_key=key)
    print("lap records:", len(laps), "| with a lap_duration:", sum(1 for l in laps if l.get("lap_duration") is not None))
    print("first lap record:", laps[0] if laps else None)
    print("stint records:", len(stints), "| first:", stints[0] if stints else None)
    print("compounds in stints:", sorted({str(x.get("compound")) for x in stints}))
    cats = {}
    for m in control:
        cats[m.get("category")] = cats.get(m.get("category"), 0) + 1
    print("race_control categories:", cats)
    print("SafetyCar / flag messages:")
    for m in control:
        if m.get("category") == "SafetyCar" or "SAFETY" in str(m.get("message", "")).upper():
            print("   ", m.get("date"), "|", m.get("category"), "|", m.get("flag"), "|", m.get("message"))
    try:
        table, label, extra = load_race_openf1(year, name)
        print("\nlap table rows (known compounds only):", len(table))
        print("track status counts:", table["TrackStatus"].value_counts().head(8).to_dict())
        print("compounds:", table["Compound"].value_counts().to_dict())
        print("TyreLife missing:", int(table["TyreLife"].isna().sum()), "| Position missing:", int(table["Position"].isna().sum()))
        print("periods:", extra["info"]["periods_found"])
        d = prepare(table)
        print("categories:", d["cond"].value_counts().to_dict(), "| accurate:", int(d.accurate.sum()),
              "| pit in/out:", int(d.pit_in.sum()), int(d.pit_out.sum()))
        print("clean laps:", len(clean_laps(d)))
    except Exception as problem:
        print("loader stopped:", type(problem).__name__, problem)


def main(args):
    source = "openf1"
    years, limit = [], None
    if "--big" in args:                                  # old name: 2023 and 2024
        args.remove("--big")
        years = [2023, 2024]
    for flag in ("--years", "--max"):
        if flag in args:
            i = args.index(flag)
            value = args[i + 1]
            args = args[:i] + args[i + 2:]
            if flag == "--years":
                years = [int(y) for y in value.replace(" ", "").split(",") if y]
            else:
                limit = int(value)
    if "--source" in args:
        i = args.index("--source")
        source = args[i + 1].lower()
        args = args[:i] + args[i + 2:]
    if years:                                            # whole seasons: ask OpenF1 which races exist, no names needed
        races = []
        for y in years:
            try:
                found = season_races(y)
            except Exception as problem:
                print(f"could not list the {y} season: {problem}")
                continue
            print(f"{y}: {len(found)} races found")
            races += [(y, rec) for rec in found]
        if limit:
            races = races[:limit]
    else:
        races = [(int(a.split(":")[0]), a.split(":", 1)[1]) for a in args] or DEFAULT_RACES
    results, notes = [], {}
    for year, name in races:
        shown = f"{name.get('country_name')} ({name.get('location')})" if isinstance(name, dict) else name
        print(f"\n--- {year} {shown} ({source}) ---")
        try:
            if source == "fastf1":
                laps, label = load_race(year, name)
                extra = None
            else:
                laps, label, extra = load_race_openf1(year, name)
                notes[label] = extra["info"]
            result = analyze_race(laps, label, extra)
        except Exception as problem:
            import traceback
            print(traceback.format_exc())
            where = traceback.extract_tb(problem.__traceback__)[-1]
            result = {"skipped": f"{year} {shown}: {type(problem).__name__}: {str(problem)[:160]} "
                                 f"[line {where.lineno}: {where.line}]"}
        print("   ", result.get("skipped") or f"{result['clean_laps']} clean laps, {result['race_laps']} laps")
        if not result.get("skipped") and source == "openf1":
            info = notes[label]
            print("    safety car / VSC messages seen:", info["safety_car_messages"] or "NONE")
            for a, b, c in info["periods_found"]:
                print(f"    {'SC ' if c == '4' else 'VSC'} {a[11:19]} -> {b[11:19]}  ({(pd.Timestamp(b) - pd.Timestamp(a)).total_seconds():.0f} s)")
        results.append(result)
    cal = aggregate(results)
    cal["openf1_notes"] = notes
    print_report(cal)
    if "error" in cal:
        print("\nNothing could be analysed, so no file was written. Read the messages above: "
              "the usual causes are no internet or a misspelt race name.\n"
              "To check what this computer can reach, run  python CALIBRATE_FASTF1.py --diagnose")
        return
    out_name = f"calibration_{'_'.join(map(str, years))}.json" if years else "calibration.json"
    with open(out_name, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"\nWrote {out_name}. Send that file back.")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--selftest" in argv:
        bad = selftest()
        print("\n" + "=" * 70 + "\n")
        bad += openf1_selftest()
        sys.exit(1 if bad else 0)
    if "--inspect" in argv:
        a = [x for x in argv if x != "--inspect"][0]
        inspect_openf1(int(a.split(":")[0]), a.split(":", 1)[1])
        sys.exit(0)
    if "--diagnose" in argv:
        diagnose()
        sys.exit(0)
    class Tee:                                   # everything printed also goes into run_report.txt
        def __init__(self, *outs):
            self.outs = outs

        def write(self, text):
            for o in self.outs:
                o.write(text)

        def flush(self):
            for o in self.outs:
                o.flush()

    with open("run_report.txt", "w", encoding="utf-8") as report:
        real_out = sys.stdout
        sys.stdout = Tee(real_out, report)
        try:
            main(argv)
        finally:
            sys.stdout = real_out
    print("\nEverything above was also saved in run_report.txt")