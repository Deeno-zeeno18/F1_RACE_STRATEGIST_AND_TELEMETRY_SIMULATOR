"""
CALIBRATION_LOADER.py - the missing link between the calibration JSON files and the simulator's constants.

Until now the numbers in data/calibration_*.json were copied BY HAND into TIRES_AND_CARS.py, SAFETY_CAR_VSC.py and
DUEL.py. This file does not change any constant. It reads the JSON files and PRINTS, for each constant, the value in
the code next to what the files say, so a drift (or a deliberate difference) is visible at once.

Run:    python CALIBRATION_LOADER.py
        python CALIBRATION_LOADER.py data/calibration_2023.json data/calibration_2024.json
"""

import json
import sys

import TIRES_AND_CARS as T
import SAFETY_CAR_VSC as S
import DUEL

DEFAULT_FILES = ["data/calibration_2023.json", "data/calibration_2024.json"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def median_of(cal, *keys):
    """cal[key1][key2]...['median'] or None when a key is missing or the value is null."""
    node = cal
    for key in keys:
        if not isinstance(node, dict) or key not in node or node[key] is None:
            return None
        node = node[key]
    return node.get("median") if isinstance(node, dict) else node


def rows(cals):
    """(name, value in the code, [value in each file], status) for every constant that has a counterpart."""
    out = []

    def add(name, code_value, getter, status):
        out.append((name, code_value, [getter(c) for c in cals], status))

    add("pit_stop_penalty (green stop, s)", T.pit_stop_penalty, lambda c: median_of(c, "pit_loss", "GREEN"), "measured")
    for comp in ("SOFT", "MEDIUM", "HARD"):
        add(f"linear_wear {comp} (s per lap of age)", T.linear_wear[comp],
            lambda c, comp=comp: median_of(c, "degradation_per_lap", comp), "measured")
    add("fuel full-tank penalty (s)", T.solve_for_sensitivity(),
        lambda c: median_of(c, "fuel_full_tank_penalty"), "measured (rounded)")
    add("compound_pace_offset SOFT (s)", T.compound_pace_offset["SOFT"], lambda c: median_of(c, "compound_pace_offset", "SOFT"),
        "TUNED on the teams' choices, not taken from the files")
    add("compound_pace_offset HARD (s)", T.compound_pace_offset["HARD"], lambda c: median_of(c, "compound_pace_offset", "HARD"),
        "TUNED on the teams' choices, not taken from the files")
    add("SC lap_mult", S.TRACK_STATES["SC"]["lap_mult"], lambda c: median_of(c, "lap_time_ratio", "SC"), "measured")
    add("SC pit_loss (s)", S.TRACK_STATES["SC"]["pit_loss"], lambda c: median_of(c, "pit_loss", "SC"), "HAND-SET (data very noisy)")
    add("VSC lap_mult", S.TRACK_STATES["VSC"]["lap_mult"], lambda c: median_of(c, "lap_time_ratio", "VSC"), "HAND-SET (no data)")
    add("VSC pit_loss (s)", S.TRACK_STATES["VSC"]["pit_loss"], lambda c: median_of(c, "pit_loss", "VSC"), "HAND-SET (no data)")
    add("BUNCH_GAP (s)", DUEL.BUNCH_GAP, lambda c: median_of(c, "bunch_gap", "SC"), "measured (average of the files)")
    return out


def driver_rows(cal):
    """Drivers whose pace / tyre_care in the code differ from the file (empty list = identical)."""
    diffs = []
    for code, v in cal.get("drivers", {}).items():
        mine = T.drivers.get(code)
        if mine is None:
            diffs.append((code, "missing in code", v))
        elif abs(mine["pace"] - v["pace"]) > 1e-9 or abs(mine["tyre_care"] - v["tyre_care"]) > 1e-9:
            diffs.append((code, mine, {"pace": v["pace"], "tyre_care": v["tyre_care"]}))
    return diffs


def show(value):
    return "  --  " if value is None else f"{value:7.3f}"


def main(paths):
    cals = [load(p) for p in paths]
    print("constant".ljust(38) + "in code" + "".join(f"  {p.split('_')[-1][:-5]:>8}" for p in paths) + "   status")
    for name, code_value, values, status in rows(cals):
        print(name.ljust(38) + f"{code_value:7.3f}" + "".join("  " + show(v) + " " for v in values) + "  " + status)
    print()
    for path, cal in zip(paths, cals):
        diffs = driver_rows(cal)
        if path.endswith("2024.json"):
            print(f"drivers in {path}: {len(cal['drivers'])} in the file, "
                  f"{'all identical to real_drivers in the code' if not diffs else str(len(diffs)) + ' differ: ' + str([d[0] for d in diffs])}")
        else:
            print(f"drivers in {path}: {len(cal['drivers'])} in the file (the code uses the 2024 values)")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_FILES)
