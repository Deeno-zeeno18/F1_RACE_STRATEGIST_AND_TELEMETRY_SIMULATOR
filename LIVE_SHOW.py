"""
LIVE_SHOW.py - connects the advisor (ADVISOR.py) to the real simulator
(TIRES_AND_CARS.py and PIT_STRATEGY.py).

You give it the state of the race right now, plus (optionally) the real lap
times of the last few laps and what the driver says about the tyre. It answers
PIT, STAY or ASK, with the numbers behind the answer.

How one call to advise() works:
  1. Replay the last few laps through the model          -> predicted laps
  2. Compare with the real laps                          -> residuals
  3. Forecast the rest of the race if the car STAYS OUT  -> model laps
  4. Add the residual trend to that forecast             -> corrected stay-out forecast
  5. Forecast the rest of the race for a stop on each compound (fresh tyre,
     so the old tyre's residual trend is NOT carried over)
  6. Ask ADVISOR.decide() whether the difference beats the forecast error
"""

import math
from dataclasses import dataclass, field, replace

from TIRES_AND_CARS import Car
import ADVISOR

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
START_FUEL = 100.0          # same starting fuel the simulator uses


# ------------------------------------------------------------------ race state
@dataclass
class RaceState:
    driver: str                 # "balanced", "smooth" or "aggressive"
    compound: str               # compound on the car now
    tyre_age: int               # age of this set for the NEXT lap to be driven
    lap: int                    # laps completed so far
    race_laps: int              # total laps in the race
    compounds_used: set = field(default_factory=set)   # compounds already used (two-compound rule)
    recent_laps: list = field(default_factory=list)    # real lap times, oldest first, same set of tyres
    started_on_pit: bool = False                       # True if this set was fitted on a pit lap

    def __post_init__(self):
        self.compound = self.compound.upper()
        self.compounds_used = {c.upper() for c in self.compounds_used} | {self.compound}

    @property
    def laps_remaining(self):
        return self.race_laps - self.lap


def fuel_at(state, lap_index):
    """Fuel on board at the start of lap number `lap_index` (0 = first lap)."""
    return START_FUEL * (1 - lap_index / state.race_laps)


def make_car(state, tyre_age, lap_index, compound=None, care_mult=1.0):
    """
    A Car in the situation described. care_mult scales the driver's tyre_care,
    which is how a 'tyre going off' feel report enters the model (effective
    age = age x tyre_care x care_mult).
    """
    car = Car(compound or state.compound, tyre_age, fuel_at(state, lap_index),
              lap_index, driver=state.driver)
    if care_mult != 1.0:
        car.driver = dict(car.driver)          # do not touch the shared drivers dict
        car.driver["tyre_care"] *= care_mult
    return car


def _next_lap(car, race_laps):
    """The same three updates every loop in the simulator does after a lap."""
    car.increase_lap_number()
    car.fuel_unit_reduction(race_laps)
    car.tyre.increase_tyre_age()


# ------------------------------------------------------------ model forecasts
def model_laps_stay(state, care_mult=1.0):
    """Lap times for the rest of the race if the car never stops again."""
    car = make_car(state, state.tyre_age, state.lap, care_mult=care_mult)
    times = []
    for _ in range(state.laps_remaining):
        times.append(car.calculate_lap_time())
        _next_lap(car, state.race_laps)
    return times


def model_laps_pit(state, compound):
    """
    Lap times for the rest of the race if the car pits THIS lap for `compound`:
    the first lap carries the pit-stop penalty, then the new tyre runs to the flag.
    The feel multiplier is deliberately not applied: it described the OLD tyre.
    """
    car = make_car(state, state.tyre_age, state.lap)
    times = [car.pit_penalty(new_compound=compound, new_tyre_age=0)]
    _next_lap(car, state.race_laps)
    for _ in range(state.laps_remaining - 1):
        times.append(car.calculate_lap_time())
        _next_lap(car, state.race_laps)
    return times


def predicted_recent(state, care_mult=1.0):
    """What the model says the last len(recent_laps) laps should have taken."""
    m = len(state.recent_laps)
    predicted = []
    for j in range(m):
        age = state.tyre_age - m + j            # age of the set on that lap
        lap_index = state.lap - m + j
        car = make_car(state, age, lap_index, care_mult=care_mult)
        predicted.append(car.calculate_lap_time())
    return predicted


# ------------------------------------------------------------ calibration
# The engineer's way of using your lap times: find the tyre-wear multiplier that makes
# the MODEL match the REAL laps, then let the tyre model carry that multiplier through
# the rest of the stint (cliff included). It is dropped at a pit stop, because it
# describes the old tyre.
MULT_MIN, MULT_MAX, MULT_STEP = 0.70, 1.60, 0.01
FIT_PRIOR = 0.05        # pulls the fit toward 1.0 when the data cannot tell (a young tyre barely wears)
FIT_WINDOW = 6          # use at most the last 6 laps on this set
MIN_FIT_LAPS = 3        # fewer than this and we only trust the driver's report
MIN_SENSITIVITY = 0.30  # s: if wear x0.8 vs x1.3 moves the laps less than this, the laps cannot tell the wear rate


def check_recent(state):
    n_recent = len(state.recent_laps)
    usable = state.tyre_age - (1 if state.started_on_pit else 0)
    if n_recent > usable:
        raise ValueError(f"{n_recent} recent laps given, but only {usable} laps on this set "
                         f"are usable (a pit lap carries the 22 s stop and must be left out)")


def _rms(values, dof):
    return math.sqrt(sum(v * v for v in values) / max(dof, 1))


def fit_wear_multiplier(state, window=FIT_WINDOW):
    """
    Grid search for the multiplier (0.70 .. 1.60) with the smallest squared error between
    real and modelled laps, plus a tiny pull toward 1.0. Returns None with fewer than 3 laps.
    """
    check_recent(state)
    laps = state.recent_laps[-window:]
    k = len(laps)
    if k < MIN_FIT_LAPS:
        return None
    sub = replace(state, recent_laps=laps)

    steps = int(round((MULT_MAX - MULT_MIN) / MULT_STEP))
    best_m, best_obj = None, None
    for i in range(steps + 1):
        m = MULT_MIN + i * MULT_STEP
        sse = sum((a - p) ** 2 for a, p in zip(laps, predicted_recent(sub, m)))
        obj = sse + FIT_PRIOR * k * (m - 1.0) ** 2
        if best_obj is None or obj < best_obj:
            best_m, best_obj = m, obj

    at_bound = best_m <= MULT_MIN + 1e-9 or best_m >= MULT_MAX - 1e-9

    # can these laps tell the wear rate at all? A young tyre barely wears, so no.
    lo, hi = predicted_recent(sub, 0.8), predicted_recent(sub, 1.3)
    sensitivity = _rms([a - b for a, b in zip(lo, hi)], k)
    return {"multiplier": round(best_m, 2), "laps_used": k, "at_bound": at_bound,
            "sensitivity": sensitivity, "laps": laps, "sub": sub}


def calibrate(state, feel=None, laps_since_feel=0):
    """
    Decide which wear multiplier the whole-race plan should use, and how far to trust it.
      3+ real laps  -> fitted from the laps (the stopwatch beats the feeling)
      otherwise     -> the driver's report (faded), or 1.0 if there is none
    """
    prior = 1.0 if feel is None else ADVISOR.wear_multiplier_from_feel(feel, laps_since_feel)
    fit = fit_wear_multiplier(state)

    if fit is None or fit["sensitivity"] < MIN_SENSITIVITY:
        note = None
        if fit is not None:
            note = ("This tyre is too young for the lap times to show how it wears, "
                    "so the engineer is using the model" + (" and the driver's report." if feel else "."))
        return {"multiplier": prior, "source": "driver report" if feel else "model only",
                "prior": prior, "laps_used": len(state.recent_laps),
                "rms_before": None, "rms_after": None, "error_per_lap": ADVISOR.DEFAULT_ERROR,
                "warning": None, "note": note}

    m, sub, laps = fit["multiplier"], fit["sub"], fit["laps"]
    before = [a - p for a, p in zip(laps, predicted_recent(sub, prior))]
    after = [a - p for a, p in zip(laps, predicted_recent(sub, m))]
    rms_after = _rms(after, len(laps) - 1)
    warning = None
    if fit["at_bound"]:
        warning = ("The lap times are further from the model than any tyre-wear setting can explain "
                   "(traffic, a mistake, or a different pace?). Treat the answer with care.")
    return {"multiplier": m, "source": "lap times", "prior": prior, "laps_used": len(laps),
            "rms_before": round(_rms(before, len(laps)), 3), "rms_after": round(rms_after, 3),
            "error_per_lap": max(rms_after, ADVISOR.NOISE_FLOOR), "warning": warning, "note": None}


# -------------------------------------------------------------------- advise
def advise(state, feel=None, laps_since_feel=0, max_horizon=8, window=4):
    """
    state:            a RaceState
    feel:             None, or "fine" / "going off" / "badly gone" (see ADVISOR.FEEL_WEAR)
    laps_since_feel:  how many laps ago the driver said it (the effect fades)
    max_horizon:      pit-now vs stay-out is only trusted this close to the flag
    Returns a dict with the decision and the numbers behind it.
    """
    if state.laps_remaining < 1:
        raise ValueError("the race is over: no laps remaining")

    check_recent(state)
    n_recent = len(state.recent_laps)

    notes = []
    multiplier = 1.0 if feel is None else ADVISOR.wear_multiplier_from_feel(feel, laps_since_feel)

    # 1-2: residuals against the model that already includes the feel report,
    # so the feel and the lap times do not count the same wear twice.
    resid = []
    if n_recent:
        resid = ADVISOR.residuals(predicted_recent(state, multiplier), state.recent_laps)

    # 3-4: stay-out forecast, corrected by the residual trend
    stay_model = model_laps_stay(state, multiplier)
    slope = 0.0
    if len(resid) >= 2:
        fc = ADVISOR.corrected_forecast(stay_model, resid, window=window, max_horizon=max_horizon)
        stay_laps, slope = fc["forecast"], fc["slope"]
    else:
        stay_laps = stay_model
        notes.append("fewer than 2 real laps given: the stay-out forecast is the model alone")
    horizon_ok = state.laps_remaining <= max_horizon
    err = ADVISOR.forecast_error(resid, window=window)
    stay_total = sum(stay_laps)

    # 5: pit options (two-compound rule: if only one compound has been used so far,
    # the second set must be a different one)
    only_one_used = len(state.compounds_used) == 1
    options = [c for c in COMPOUNDS if not (only_one_used and c in state.compounds_used)]
    pit_totals = {c: sum(model_laps_pit(state, c)) for c in options}
    best_compound = min(pit_totals, key=pit_totals.get)

    result = {
        "laps_remaining": state.laps_remaining,
        "feel": feel, "wear_multiplier": round(multiplier, 4),
        "residuals": [round(r, 3) for r in resid],
        "residual_slope": round(slope, 3),
        "error_per_lap": round(err, 3),
        "stay_total": round(stay_total, 2),
        "pit_totals": {c: round(t, 2) for c, t in pit_totals.items()},
        "best_pit_compound": best_compound,
        "notes": notes,
    }

    # the rule overrides everything: finishing on one compound is not allowed
    if only_one_used:
        result.update({"decision": "PIT", "diff": None, "margin": None,
                       "why": "only one compound used so far: a stop is compulsory"})
        return result

    # 6: decide
    verdict = ADVISOR.decide(pit_time=pit_totals[best_compound], stay_time=stay_total,
                             laps_remaining=state.laps_remaining, err_per_lap=err,
                             confident=horizon_ok)
    if not horizon_ok:
        verdict["why"] = (f"{state.laps_remaining} laps remain: pit-now vs stay-out-to-the-flag "
                          f"is only reliable close to the end (max_horizon={max_horizon}); "
                          f"use the strategy optimiser for the whole race")
    result.update(verdict)
    return result


# ---------------------------------------------------------------------- tests
def _fake_real_laps(state, m, truth_mult, noise):
    """Stand-in for real timing data: laps from a car whose tyre wears truth_mult x faster."""
    laps = []
    for j in range(m):
        age = state.tyre_age - m + j
        lap_index = state.lap - m + j
        laps.append(make_car(state, age, lap_index, care_mult=truth_mult).calculate_lap_time() + noise[j])
    return laps


def _show(title, r):
    print(f"\n{title}")
    for key in ("decision", "why", "diff", "margin", "stay_total", "pit_totals",
                "best_pit_compound", "residuals", "residual_slope", "error_per_lap",
                "wear_multiplier", "notes"):
        print(f"   {key}: {r[key]}")


if __name__ == "__main__":
    from PIT_STRATEGY import simulate_a_race, time_if_pit_now

    # CHECK 1: do my per-lap forecasts add up to your own totals?
    state = RaceState("balanced", "HARD", tyre_age=40, lap=62, race_laps=65,
                      compounds_used={"MEDIUM", "HARD"})
    car = Car("HARD", 40, fuel_at(state, 62), 62)
    print("CHECK 1: forecasts vs your simulate_a_race / time_if_pit_now (3 laps left)")
    print("   stay :", round(sum(model_laps_stay(state)), 6), "vs", round(simulate_a_race(car, 3, 65), 6))
    for c in COMPOUNDS:
        print(f"   {c:6}:", round(sum(model_laps_pit(state, c)), 6), "vs",
              round(time_if_pit_now(car, 3, 65, c), 6))

    # CHECK 2: HARD set (fitted on lap 27), 7 laps to go, tyre exactly as the model expects
    state = RaceState("balanced", "HARD", tyre_age=31, lap=58, race_laps=65,
                      compounds_used={"MEDIUM", "HARD"}, started_on_pit=True)
    noise = [0.03, -0.04, 0.02, -0.01]
    state.recent_laps = _fake_real_laps(state, 4, 1.0, noise)
    _show("CHECK 2: tyre behaving as modelled", advise(state))

    # CHECK 3: same race, but the real laps show the tyre wearing 1.25x faster
    state.recent_laps = _fake_real_laps(state, 4, 1.25, noise)
    _show("CHECK 3: real laps slower than the model says", advise(state))

    # CHECK 4: driver says 'going off', laps still look as modelled
    state.recent_laps = _fake_real_laps(state, 4, 1.0, noise)
    _show("CHECK 4: driver says 'going off', laps as modelled", advise(state, feel="going off"))

    # CHECK 4b: HARD age 40, 3 laps to the flag, no real laps given. Staying out is about
    # 2.3 s faster than the best stop, which beats the 0.52 s margin, so the answer is STAY.
    close = RaceState("balanced", "HARD", tyre_age=40, lap=62, race_laps=65,
                      compounds_used={"MEDIUM", "HARD"})
    _show("CHECK 4b: 3 laps to go, pit and stay nearly equal", advise(close))

    # CHECK 5: too early to compare pit-now with stay-out
    early = RaceState("balanced", "MEDIUM", tyre_age=20, lap=30, race_laps=65,
                      compounds_used={"MEDIUM", "HARD"})
    _show("CHECK 5: 35 laps remain", advise(early))

    # CHECK 6: only one compound used so far
    one = RaceState("balanced", "MEDIUM", tyre_age=30, lap=40, race_laps=65)
    _show("CHECK 6: one compound used so far", advise(one))
    # CHECK 7: calibrate() on laps made with a KNOWN wear multiplier must recover it
    cal_state = RaceState("balanced", "HARD", tyre_age=31, lap=58, race_laps=65,
                          compounds_used={"MEDIUM", "HARD"}, started_on_pit=True)
    cal_state.recent_laps = _fake_real_laps(cal_state, 4, 1.25, noise)
    cal = calibrate(cal_state)
    print("\nCHECK 7: calibrate() on laps made with a true wear multiplier of 1.25")
    print("   fitted multiplier:", cal["multiplier"], "| source:", cal["source"],
          "| rms before/after:", cal["rms_before"], "/", cal["rms_after"])
    assert cal["source"] == "lap times" and abs(cal["multiplier"] - 1.25) <= 0.02, "calibrate did not recover the planted multiplier"
    young = RaceState("balanced", "MEDIUM", tyre_age=6, lap=26, race_laps=65,
                      compounds_used={"SOFT", "MEDIUM"}, started_on_pit=True)
    young.recent_laps = _fake_real_laps(young, 4, 1.25, noise)
    cal = calibrate(young)
    print("   young tyre (age 6): source:", cal["source"], "-> the laps cannot show the wear, so the model is used")
    assert cal["source"] == "model only"
