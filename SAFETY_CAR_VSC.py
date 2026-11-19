"""
SAFETY_CAR_VSC.py - safety car (SC) and virtual safety car (VSC) for the simulator.

A track state changes three things for every car on a lap:
  lap_mult  : the lap time is set by the safety car, not by the tyre, so it is
              lap_mult x the base lap time (92 s)
  wear      : how much of a normal lap's wear the tyre gets (slow laps are gentle)
  pit_loss  : the EXTRA seconds a stop costs on top of a normal lap in that state
              (green: your 22 s; under SC/VSC the field is slow, so a stop is cheaper)
Overtaking is not allowed under SC; positions come with the duel layer later.

WHERE THE NUMBERS COME FROM (see TRACK_STATES below):
  MEASURED from real races (OpenF1, 2023-24): the green pit loss and the SC lap_mult.
  HAND-SET: everything for the VSC (the data have no usable VSC laps), the SC pit loss
  (measured 12.6 but from only 25 noisy stops) and the wear factors.

An announced event is a plan: [("SC", 4)] means "the next 4 laps are under the
safety car, then green". [("VSC", 2), ("GREEN", 3), ("SC", 3)] also works.

THE QUESTION: pit NOW, or wait? Both answers are scored on the whole rest of
the race with the best possible set of stops (any number, two-compound rule,
no set used for fewer than min_stint laps between stops), found by dynamic
programming like best_strategy_any_stops. Comparing "pit now" with "one later
stop" would be unfair: a cheap safety-car stop makes an EXTRA stop worthwhile.
"""

from TIRES_AND_CARS import Tyre, drivers, compound_pace_offset, pit_stop_penalty
from PIT_STRATEGY import min_stint
from LIVE_SHOW import RaceState, make_car, COMPOUNDS
import math
import ADVISOR

TRACK_STATES = {
    "GREEN": {"lap_mult": 1.00, "wear": 1,    "pit_loss": pit_stop_penalty},
    "VSC":   {"lap_mult": 1.30, "wear": 0.60, "pit_loss": 15.0},
    "SC":    {"lap_mult": 1.56, "wear": 0.40, "pit_loss": 11.0},   # lap_mult MEASURED 1.56 (2023) / 1.57 (2024); pit_loss measured 12.6 but very noisy (25 stops)
}


def expand_plan(plan, laps_remaining):
    """Turn [("SC", 4)] into one track state per remaining lap (green afterwards)."""
    states = []
    for name, laps in plan:
        name = name.upper()
        if name not in TRACK_STATES:
            raise ValueError(f"unknown track state '{name}', choose from {list(TRACK_STATES)}")
        if laps < 1:
            raise ValueError("each part of the plan needs at least 1 lap")
        states += [name] * laps
    states = states[:laps_remaining]
    states += ["GREEN"] * (laps_remaining - len(states))
    return states


# ------------------------------------------------------------ full simulation
def drive_lap(car, track_state, race_laps, pit_compound=None):
    """
    One lap in the given track state, optionally pitting at the start of it.
    Returns the lap time, then ages the car the way the simulator's loops do
    (lap number, fuel), except that tyre age grows by the state's wear factor.
    """
    s = TRACK_STATES[track_state]
    if pit_compound is not None:
        car.replace_tyre_pit(pit_compound, 0)

    if track_state == "GREEN":
        lap_time = car.calculate_lap_time()
    else:
        lap_time = s["lap_mult"] * car._raw_lap_parts()["base time"]
    if pit_compound is not None:
        lap_time += s["pit_loss"]

    car.increase_lap_number()
    car.fuel_unit_reduction(race_laps)
    car.tyre.tyre_age += s["wear"]
    return lap_time


def simulate_stops(state, states, stops, care_mult=1.0):
    """
    Lap times for the rest of the race. stops = [(offset, compound), ...] where
    offset 0 means "pit on the next lap", 3 means "three laps from now".
    The feel multiplier belongs to the old tyre, so it stops at the first stop.
    """
    stop_at = dict(stops)
    car = make_car(state, state.tyre_age, state.lap, care_mult=care_mult)
    times = []
    for i, track_state in enumerate(states):
        if i in stop_at:
            car.driver = dict(drivers[state.driver])
            times.append(drive_lap(car, track_state, state.race_laps, pit_compound=stop_at[i]))
        else:
            times.append(drive_lap(car, track_state, state.race_laps))
    return times


# ------------------------------------------------------------------- the search
def _stint_costs(start, compound, start_age, care, profile, states):
    """
    costs[e] = tyre-dependent cost of driving laps start..e-1 on one set.
    Under SC/VSC the lap time does not depend on the tyre, so only green laps
    add cost; every lap still ages the tyre by the state's wear factor.
    The search stops where the Tyre class says the set is too old.
    """
    costs = {start: 0.0}
    total, age = 0.0, start_age
    for t in range(start, len(states)):
        try:
            tyre = Tyre(compound, age)
        except Exception:
            break
        if states[t] == "GREEN":
            total += compound_pace_offset[compound] + profile["pace"] + tyre.get_wear_penalty(care)
        age += TRACK_STATES[states[t]]["wear"]
        costs[t + 1] = total
    return costs


def best_plan(state, states, first_offsets, allow_no_stop, care_mult=1.0):
    """
    Cheapest set of stops for the rest of the race.
    first_offsets : the laps (offsets from now) on which the FIRST stop may happen
    allow_no_stop : whether finishing on the current set counts as a plan
    Returns (tyre-dependent cost, stops) or None if no legal plan exists.
    """
    profile = drivers[state.driver]
    n = len(states)
    pit_loss = [TRACK_STATES[s]["pit_loss"] for s in states]
    used0 = frozenset(state.compounds_used)

    current = _stint_costs(0, state.compound, state.tyre_age,
                           profile["tyre_care"] * care_mult, profile, states)
    fresh = {(k, c): _stint_costs(k, c, 0, profile["tyre_care"], profile, states)
             for k in range(n) for c in COMPOUNDS}

    g = [dict() for _ in range(n)]       # g[k][(compound fitted at k, compounds used)] = (cost, stops)

    def put(k, key, cost, stops):
        if key not in g[k] or cost < g[k][key][0]:
            g[k][key] = (cost, stops)

    candidates = []
    for k in first_offsets:
        if k not in current:
            continue
        for c in COMPOUNDS:
            put(k, (c, used0 | {c}), current[k] + pit_loss[k], [(k, c)])
    if allow_no_stop and len(used0) >= 2 and n in current:
        candidates.append((current[n], []))

    for k in range(n):
        for (c, used), (cost, stops) in g[k].items():
            run = fresh[(k, c)]
            if n in run and len(used) >= 2:
                candidates.append((cost + run[n], stops))
            for k2 in range(k + min_stint, n):
                if k2 not in run:
                    break
                for c2 in COMPOUNDS:
                    put(k2, (c2, used | {c2}), cost + run[k2] + pit_loss[k2], stops + [(k2, c2)])

    return min(candidates) if candidates else None


def slow_window(states):
    """
    The run of non-green laps at the start of the plan. A stop on any of those laps is cheap,
    so 'pit during the slow laps' is one choice. Returns (window, split): the number of slow
    laps, and the first lap offset that is NOT part of 'pit now'.
    """
    window = 0
    while window < len(states) and states[window] != "GREEN":
        window += 1
    return window, max(window, 1)


STOP_WINDOW_LOOKAHEAD = 25      # laps ahead searched for the stop window


def stop_window(state, states, care_mult, margin):
    """
    WHEN to stop, as a range instead of one lap. For every possible stop lap in the next
    STOP_WINDOW_LOOKAHEAD laps we price the best whole-race plan that stops then; the window is every lap
    whose plan costs no more than `margin` seconds above the best one (the forecast error, so laps inside the
    window cannot be told apart). Returns {"best_lap", "earliest", "latest"} (lap numbers), or None.
    """
    n = len(states)
    costs = {}
    for k in range(0, min(n, STOP_WINDOW_LOOKAHEAD)):
        found = best_plan(state, states, [k], False, care_mult)
        if found is not None:
            costs[k] = found[0]
    if not costs:
        return None
    best_k = min(costs, key=costs.get)
    inside = [k for k, c in costs.items() if c <= costs[best_k] + margin]
    lo = best_k
    while lo - 1 in inside:
        lo -= 1
    hi = best_k
    while hi + 1 in inside:
        hi += 1
    lap = lambda k: state.lap + k + 1
    return {"best_lap": lap(best_k), "earliest": lap(lo), "latest": lap(hi), "margin": round(margin, 2),
            "best_cost": costs[best_k], "costs": {lap(k): round(c, 2) for k, c in costs.items()}}


WIDE_WINDOW = 8         # a window this wide (laps) cannot say when: the driver's feel decides


def decide_from_window(state, w, best_later, margin):
    """
    PIT / STAY / ASK from the stop window instead of "now vs the best later lap" (which are almost always
    within the margin of each other, so that rule says ASK or STAY right up to the stop).
      the window has not opened yet                         -> STAY   (say when it opens)
      we are inside it, best lap is now/next, or the window
      is about to close                                     -> PIT
      inside a wide window, best lap still ahead            -> ASK    (any lap works within the error: how do the tyres feel?)
      inside a narrow window, best lap still ahead          -> STAY   (say which lap is best)
    If finishing on this set costs no more than the best stop plus the margin, no stop is needed: STAY.
    """
    now_lap = state.lap + 1
    if best_later and not best_later["stops"] and best_later["search_cost"] <= w["best_cost"] + margin:
        return "STAY", "finishing on this set is within the forecast error of the best stop plan: no stop needed", "stay out to the flag"
    width = w["latest"] - w["earliest"] + 1
    span = f"laps {w['earliest']}-{w['latest']}" if width > 1 else f"lap {w['best_lap']}"
    if w["earliest"] > now_lap:
        return "STAY", f"the stop window ({span}) has not opened yet; best lap {w['best_lap']}", \
               f"stay out; stop window opens lap {w['earliest']}, best lap {w['best_lap']}"
    if w["best_lap"] - now_lap <= 1:                      # (best_lap <= latest, so this also covers 'the window is closing')
        return "PIT", f"inside the stop window ({span}) and the best lap is now or the window is closing", \
               f"box now (best lap {w['best_lap']}, window {span})"
    if width >= WIDE_WINDOW:
        return "ASK", f"inside a wide stop window ({span}): any of these laps costs the same within the error, so the tyres' feel decides", \
               f"window open {span}, best lap {w['best_lap']}: how do the tyres feel?"
    return "STAY", f"inside the stop window ({span}) but the best lap is {w['best_lap']}", \
           f"stay out until lap {w['best_lap']} (window {span})"


def describe(state, stops):
    if not stops:
        return "no more stops"
    parts = []
    for offset, compound in stops:
        when = "now" if offset == 0 else f"lap {state.lap + offset + 1}"
        parts.append(f"{when} on {compound}")
    return "pit " + ", then ".join(parts)


def advise_with_track(state, plan, feel=None, laps_since_feel=0, wear_mult=None, err_per_lap=None):
    """
    Should the car pit DURING the slow period, or wait for the green flag?
    The slow period is the run of non-green laps at the start of the plan.
    A stop on any of those laps is cheap, so "pit now" and "pit one lap later,
    still under the safety car" are treated as the same choice (the best lap
    inside the window is reported). With no slow period at the start, the
    question is simply: pit now, or later?
    Totals are whole-race times from a full simulation.
    """
    n = state.laps_remaining
    states = expand_plan(plan, n)
    if wear_mult is not None:                    # from LIVE.calibrate(): fitted to the real laps
        m = wear_mult
    else:
        m = 1.0 if feel is None else ADVISOR.wear_multiplier_from_feel(feel, laps_since_feel)
    err = ADVISOR.DEFAULT_ERROR if err_per_lap is None else err_per_lap

    window, split = slow_window(states)

    now = best_plan(state, states, range(0, split), False, m)
    later = best_plan(state, states, range(split, n), True, m)

    def summary(found):
        if found is None:
            return None
        cost, stops = found
        total = sum(simulate_stops(state, states, stops, m))
        return {"how": describe(state, stops), "stops": stops,
                "total": round(total, 2), "search_cost": cost}

    best_now, best_later = summary(now), summary(later)
    result = {"track_states": states, "slow_laps": window, "wear_mult": m,
              "best_now": best_now, "best_later": best_later}
    if window == 0:                              # a plain green-flag situation: also say WHEN, as a range
        margin = err * math.sqrt(max(n, 1))
        try:
            result["stop_window"] = stop_window(state, states, m, margin)
        except Exception as problem:             # keep the answer, but say why there is no window
            result["stop_window"] = None
            result["stop_window_error"] = f"{type(problem).__name__}: {problem}"

    if best_now is None or best_later is None:
        result.update({"decision": "PIT" if best_later is None else "STAY",
                       "diff": None, "margin": None, "why": "only one of the two options is legal"})
        return result

    verdict = ADVISOR.decide(pit_time=best_now["total"], stay_time=best_later["total"],
                             laps_remaining=n, err_per_lap=err)
    when = f"during the {window} slow laps" if window else "now"
    verdict["why"] = {"PIT": f"pitting {when} beats waiting by more than the margin",
                      "STAY": f"waiting beats pitting {when} by more than the margin",
                      "ASK": f"pitting {when} and waiting are within the forecast error: ask the driver"
                      }[verdict["decision"]]
    result.update(verdict)
    w = result.get("stop_window")
    if window == 0 and w:                       # green flag: the stop window decides, the pit-now-vs-later numbers stay as support
        decision, why, advice = decide_from_window(state, w, best_later, err * math.sqrt(max(n, 1)))
        result.update({"decision": decision, "why": why, "advice": advice})
    return result


# ----------------------------------------------------------------------- tests
def _show(title, r):
    print(f"\n{title}")
    track = "".join({"GREEN": "g", "VSC": "V", "SC": "S"}[s] for s in r["track_states"][:12])
    print("   track (first 12 laps):", track)
    print("   best now  :", r["best_now"]["how"], "->", r["best_now"]["total"])
    print("   best later:", r["best_later"]["how"], "->", r["best_later"]["total"])
    print("   decision  :", r["decision"], "| diff", r["diff"], "| margin", r["margin"])
    print("   why       :", r["why"])


if __name__ == "__main__":
    from TIRES_AND_CARS import Car
    from PIT_STRATEGY import simulate_a_race, time_if_pit_now
    from LIVE_SHOW import fuel_at

    # CHECK 1: with no safety car, do the full simulations still match your functions?
    s = RaceState("balanced", "HARD", tyre_age=40, lap=62, race_laps=65, compounds_used={"MEDIUM", "HARD"})
    car = Car("HARD", 40, fuel_at(s, 62), 62)
    green = expand_plan([], 3)
    print("CHECK 1: green-flag totals vs your functions (3 laps left)")
    print("   stay :", round(sum(simulate_stops(s, green, [])), 6), "vs", round(simulate_a_race(car, 3, 65), 6))
    for c in COMPOUNDS:
        print(f"   {c:6}:", round(sum(simulate_stops(s, green, [(0, c)])), 6), "vs",
              round(time_if_pit_now(car, 3, 65, c), 6))

    # CHECK 2: does the search's cost agree with the full simulation?
    # Whatever the plan, (full-simulation total - search cost) is the same constant
    # (base time + fuel), so two different plans must give the same constant.
    base = dict(driver="balanced", compound="MEDIUM", tyre_age=20, lap=20, race_laps=65,
                compounds_used={"MEDIUM"})
    st = RaceState(**base)
    states = expand_plan([("SC", 4)], st.laps_remaining)
    n2 = st.laps_remaining
    for label, offsets, no_stop in (("inside the slow laps", range(0, 4), False),
                                    ("after them", range(4, n2), True)):
        cost, stops = best_plan(st, states, offsets, no_stop)
        total = sum(simulate_stops(st, states, stops))
        print(f"\nCHECK 2 ({label}): total - search cost = {total - cost:.6f}")

    # the race: lap 20 of 65, MEDIUM age 20, only MEDIUM used so far
    _show("CHECK 3: no safety car", advise_with_track(RaceState(**base), []))
    _show("CHECK 4: safety car for 4 laps, announced now", advise_with_track(RaceState(**base), [("SC", 4)]))
    _show("CHECK 5: virtual safety car for 2 laps", advise_with_track(RaceState(**base), [("VSC", 2)]))

    fresh = dict(driver="balanced", compound="HARD", tyre_age=3, lap=23, race_laps=65,
                 compounds_used={"MEDIUM", "HARD"})
    _show("CHECK 6: HARD set fitted 3 laps ago, safety car for 4 laps",
          advise_with_track(RaceState(**fresh), [("SC", 4)]))

    late = dict(driver="balanced", compound="HARD", tyre_age=20, lap=55, race_laps=65,
                compounds_used={"MEDIUM", "HARD"})
    _show("CHECK 7: lap 55, healthy HARD set, safety car for 4 laps",
          advise_with_track(RaceState(**late), [("SC", 4)]))

    worn = dict(driver="balanced", compound="MEDIUM", tyre_age=26, lap=40, race_laps=65,
                compounds_used={"HARD", "MEDIUM"})
    _show("CHECK 8: lap 40, MEDIUM set 26 laps old, safety car for 4 laps",
          advise_with_track(RaceState(**worn), [("SC", 4)]))