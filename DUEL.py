"""
DUEL.py - you against one rival, over the rest of the race.

THE QUESTION: pit in the slow laps, or wait? The answer now depends on what the
rival does too, so every choice of yours is scored against every choice of his.

GAP CONVENTION (same as simulate_duel in PIT_STRATEGY.py):
    gap = seconds the rival is BEHIND you.  Positive = you lead.  Negative = he leads.
Each lap:  gap = gap + rival_lap_time - my_lap_time.

THE SAFETY-CAR RULE (the reason this file exists):
    During a safety car nobody may overtake, so POSITIONS are kept. But the field closes
    up behind the safety car, so GAPS are wiped out: at the first lap of every safety-car
    period the gap shrinks to at most BUNCH_GAP seconds (either way round).
    A virtual safety car does NOT do this: everyone slows by the same amount and the
    gaps stay as they were.
    So a driver who led by 15 s and pits under the safety car does not keep a 4 s cushion
    (15 s lead - 11 s stop). He rejoins about 10 s BEHIND, because his lead was wiped out
    before he stopped.

WHAT IS NOT IN HERE YET: dirty air (a faster car stuck behind a slower one), the order
of cars leaving the pits, other cars, lapped traffic. A pass is simply "the gap changed sign".
BUNCH_GAP is MEASURED (see its definition below): about 1.05 s between queued cars behind a safety car.
"""

from SAFETY_CAR_VSC import expand_plan, best_plan, simulate_stops, slow_window, describe
import ADVISOR

BUNCH_GAP = 1.05    # seconds. MEASURED: gap between queued cars behind the safety car, 1.14 (2023) and 1.02 (2024), 280 samples


def gap_trace(my_times, rival_times, states, gap0, bunch_gap=BUNCH_GAP):
    """
    Play the laps one by one and return (final_gap, per-lap gaps, swaps).
    swaps = the laps (counted from the first lap of the plan) on which the two cars
    change places, in order. Empty if nobody ever passes.
    """
    gap = gap0
    ahead = gap0 >= 0
    trace, swaps = [], []
    for i, (mine, his, state) in enumerate(zip(my_times, rival_times, states)):
        if state == "SC" and (i == 0 or states[i - 1] != "SC"):
            gap = max(-bunch_gap, min(bunch_gap, gap))       # the field closes up
        gap += his - mine
        trace.append(gap)
        if (gap >= 0) != ahead:
            swaps.append(i + 1)
            ahead = gap >= 0
    return gap, trace, swaps


def _options(state, states, split, n, mult):
    """The two things a driver can do: stop in the slow window, or wait. Each is his best plan."""
    found = {"pit": best_plan(state, states, range(0, split), False, mult),
             "wait": best_plan(state, states, range(split, n), True, mult)}
    out = {}
    for key, plan in found.items():
        if plan is None:
            continue
        stops = plan[1]
        times = simulate_stops(state, states, stops, mult)
        out[key] = {"stops": stops, "how": describe(state, stops), "times": times, "total": sum(times)}
    return out


def duel(me, rival, gap, plan, my_wear=1.0, rival_wear=1.0, err_per_lap=None, bunch_gap=BUNCH_GAP):
    """
    me, rival : RaceState objects for the same lap of the same race
    gap       : seconds the rival is behind you (negative = he is ahead)
    plan      : the announced track plan, e.g. [("SC", 4)] (empty = green flag)
    Returns a dict with the payoff table and a PIT / STAY / ASK decision.
    The decision looks at the WORST case over the rival's choices for each of yours, so it
    is the choice that holds up whatever he does.
    """
    if (me.lap, me.race_laps) != (rival.lap, rival.race_laps):
        raise ValueError("both cars must be on the same lap of the same race")
    n = me.laps_remaining
    states = expand_plan(plan, n)
    window, split = slow_window(states)
    err = ADVISOR.DEFAULT_ERROR if err_per_lap is None else err_per_lap

    mine = _options(me, states, split, n, my_wear)
    his = _options(rival, states, split, n, rival_wear)
    if not mine:
        raise ValueError("you have no legal plan for the rest of the race")
    if not his:
        raise ValueError("the rival has no legal plan for the rest of the race")

    cells = []
    for my_key, m in mine.items():
        for his_key, r in his.items():
            final, trace, swaps = gap_trace(m["times"], r["times"], states, gap, bunch_gap)
            cells.append({"me": my_key, "rival": his_key,
                          "my_how": m["how"], "rival_how": r["how"],
                          "final_gap": round(final, 2), "ahead_at_flag": final > 0,
                          "swap_laps": [me.lap + s for s in swaps],
                          "my_total": round(m["total"], 2), "rival_total": round(r["total"], 2)})

    worst = {key: min(c["final_gap"] for c in cells if c["me"] == key) for key in mine}
    result = {"track_states": states, "slow_laps": window, "gap": gap, "bunch_gap": bunch_gap,
              "my_options": {k: {"how": v["how"], "total": round(v["total"], 2)} for k, v in mine.items()},
              "rival_options": {k: {"how": v["how"], "total": round(v["total"], 2)} for k, v in his.items()},
              "cells": cells, "worst_case": worst}

    when = f"during the {window} slow laps" if window else "now"
    if len(worst) == 1:
        only = next(iter(worst))
        result.update({"decision": "PIT" if only == "pit" else "STAY", "diff": None, "margin": None,
                       "why": "only one of your two options is legal"})
        return result

    verdict = ADVISOR.decide(pit_time=-worst["pit"], stay_time=-worst["wait"],
                             laps_remaining=n, err_per_lap=err)
    safe_pit, safe_wait = worst["pit"] > 0, worst["wait"] > 0
    if safe_pit and not safe_wait:
        position = f"Pitting {when} keeps you ahead whatever he does; waiting can lose the place."
    elif safe_wait and not safe_pit:
        position = f"Waiting keeps you ahead whatever he does; pitting {when} can lose the place."
    elif safe_pit and safe_wait:
        position = "Both keep you ahead whatever he does."
    else:
        position = "Neither choice is safe: he can beat you at the flag whatever you do."
    verdict["why"] = {"PIT": "Pitting has the better worst case. ",
                      "STAY": "Waiting has the better worst case. ",
                      "ASK": "The two worst cases are within the forecast error. "}[verdict["decision"]] + position
    verdict["safe_if_pit"], verdict["safe_if_wait"] = safe_pit, safe_wait
    result.update(verdict)
    return result


# ----------------------------------------------------------------------------- tests
def _show(title, r):
    print(f"\n{title}")
    print(f"   you: {r['my_options']}")
    print(f"   him: {r['rival_options']}")
    print("   final gap at the flag (positive = you are ahead):")
    for c in r["cells"]:
        place = "ahead" if c["ahead_at_flag"] else "BEHIND"
        lc = f", places swap on lap {', '.join(map(str, c['swap_laps']))}" if c["swap_laps"] else ""
        print(f"      you {c['me']:4} / he {c['rival']:4}: {c['final_gap']:7.2f} s  ({place}{lc})")
    print(f"   worst cases: {r['worst_case']}")
    print(f"   decision: {r['decision']} | diff {r['diff']} | margin {r['margin']}")
    print(f"   why: {r['why']}")


if __name__ == "__main__":
    from LIVE_SHOW import RaceState
    from PIT_STRATEGY import simulate_duel
    from TIRES_AND_CARS import Car

    # CHECK 1: with no safety car and no stops by him, does my gap engine reproduce YOUR simulate_duel?
    print("CHECK 1: gap engine vs your simulate_duel (HARD age 40 vs SOFT age 8, 3 laps left)")
    me = RaceState("balanced", "HARD", tyre_age=40, lap=62, race_laps=65, compounds_used={"MEDIUM", "HARD"})
    rv = RaceState("balanced", "SOFT", tyre_age=8, lap=62, race_laps=65, compounds_used={"MEDIUM", "SOFT"})
    states = expand_plan([], 3)
    for gap in (15.0, 25.0):
        for label, stops, theirs in (("stay", [], "stay"), ("SOFT", [(0, "SOFT")], "SOFT")):
            my_times = simulate_stops(me, states, stops)
            his_times = simulate_stops(rv, states, [])
            mine, _, _ = gap_trace(my_times, his_times, states, gap)
            c1 = Car("HARD", 40, 100, 62); c2 = Car("SOFT", 8, 100, 62)
            theirs = simulate_duel(c1, c2, gap, 3, 65, "stay" if label == "stay" else "SOFT")["final_gap"]
            print(f"   gap {gap:4}  {label:4}: mine {mine:7.2f}   yours {theirs:7.2f}   {'OK' if abs(mine - theirs) < 0.006 else 'DIFFERENT'}")

    # CHECK 2: the safety-car rule. Same cars, same lead, once under SC and once under VSC.
    mine = dict(driver="balanced", compound="MEDIUM", tyre_age=22, lap=30, race_laps=65, compounds_used={"HARD", "MEDIUM"})
    his = dict(driver="balanced", compound="MEDIUM", tyre_age=22, lap=30, race_laps=65, compounds_used={"HARD", "MEDIUM"})
    _show("CHECK 2a: you lead by 15 s, SAFETY CAR for 4 laps (his lead gets wiped out)",
          duel(RaceState(**mine), RaceState(**his), 15.0, [("SC", 4)]))
    _show("CHECK 2b: you lead by 15 s, VIRTUAL safety car for 4 laps (gaps are kept)",
          duel(RaceState(**mine), RaceState(**his), 15.0, [("VSC", 4)]))

    # CHECK 3: he is ahead of you
    _show("CHECK 3: he leads by 6 s (gap = -6), safety car for 4 laps",
          duel(RaceState(**mine), RaceState(**his), -6.0, [("SC", 4)]))

    # CHECK 4: green flag, nothing special
    _show("CHECK 4: green flag, you lead by 2 s, same tyres",
          duel(RaceState(**mine), RaceState(**his), 2.0, []))