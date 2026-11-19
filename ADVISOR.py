"""
ADVISOR.py - live layer of the F1 simulator.

Step 1: residuals            how wrong was the model, lap by lap?
Step 2: fit_line             what is the recent trend in those errors?
Step 3: corrected_forecast   model forecast + extended error trend
Step 4: feel inputs          turn "tyre going off" into numbers (and let them fade)
Step 5: forecast_error       how big is the typical error of the forecast?
Step 6: decide               PIT / STAY / ASK
"""

import math


# ---------------------------------------------------------------- Step 1
def residuals(predicted, actual):
    """actual - predicted, lap by lap. Positive = slower than the model said."""
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have the same length")
    return [a - p for p, a in zip(predicted, actual)]


# ---------------------------------------------------------------- Step 2
def fit_line(ys):
    """
    Least-squares straight line through points (0, ys[0]), (1, ys[1]), ...
    Returns (slope, intercept). Needs at least 2 points.
    """
    n = len(ys)
    if n < 2:
        raise ValueError("need at least 2 points to fit a line")
    xs = range(n)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    return slope, intercept


# ---------------------------------------------------------------- Step 3
def corrected_forecast(model_forecast, resid, window=4, max_horizon=5):
    """
    model_forecast: the model's lap times for the NEXT laps.
    resid:          residuals so far (oldest first).
    window:         only the last `window` residuals are used for the trend.
    max_horizon:    how many laps ahead we trust a straight-line extension.

    Returns a dict:
      forecast  - corrected lap times (model + extended residual trend)
      slope     - how fast the model error is growing per lap
      confident - False if we had too little data or looked too far ahead
    """
    recent = resid[-window:]

    # Not enough data for a trend: return the model unchanged and say so.
    if len(recent) < 2:
        return {"forecast": list(model_forecast), "slope": 0.0, "confident": False}

    slope, intercept = fit_line(recent)
    n = len(recent)

    corrected = []
    for k, model_time in enumerate(model_forecast, start=1):
        # x = n - 1 is the last known lap, so lap k ahead sits at x = n - 1 + k
        predicted_resid = intercept + slope * (n - 1 + k)
        corrected.append(model_time + predicted_resid)

    confident = len(model_forecast) <= max_horizon
    return {"forecast": corrected, "slope": slope, "confident": confident}


# ---------------------------------------------------------------- Step 4
# PLACEHOLDER values - to be calibrated later against real (FastF1) data.
FEEL_WEAR = {          # multiplier on the tyre's effective age
    "fine": 1.00,
    "going off": 1.15,
    "badly gone": 1.30,
}
FEEL_RIVAL_PACE = {    # seconds added to the rival's lap time (negative = faster)
    "slower": +0.3,
    "same": 0.0,
    "faster": -0.3,
}


def faded_multiplier(m0, laps_since, half_life=3.0):
    """
    A feel report should not last forever. The EXTRA wear (m0 - 1) halves
    every `half_life` laps, so the multiplier drifts back toward 1.0
    unless the driver reports again.
    """
    return 1.0 + (m0 - 1.0) * 0.5 ** (laps_since / half_life)


def wear_multiplier_from_feel(feel, laps_since_report=0, half_life=3.0):
    if feel not in FEEL_WEAR:
        raise ValueError(f"unknown feel '{feel}', choose from {list(FEEL_WEAR)}")
    return faded_multiplier(FEEL_WEAR[feel], laps_since_report, half_life)


def rival_pace_from_feel(feel):
    if feel not in FEEL_RIVAL_PACE:
        raise ValueError(f"unknown rival feel '{feel}', choose from {list(FEEL_RIVAL_PACE)}")
    return FEEL_RIVAL_PACE[feel]


# ---------------------------------------------------------------- Step 5
NOISE_FLOOR = 0.20     # s/lap: real lap times never repeat exactly (placeholder)
DEFAULT_ERROR = 0.30   # s/lap: used when there is too little data (placeholder)


def forecast_error(resid, window=4):
    """
    Typical size (seconds per lap) of the forecast error: how far the recent
    residuals sit from their own fitted line. With fewer than 3 residuals we
    cannot tell, so we return a cautious default.
    """
    recent = resid[-window:]
    n = len(recent)
    if n < 3:
        return DEFAULT_ERROR
    slope, intercept = fit_line(recent)
    sq = [(y - (intercept + slope * x)) ** 2 for x, y in enumerate(recent)]
    rms = math.sqrt(sum(sq) / (n - 2))      # n - 2 because the line used 2 numbers
    return max(rms, NOISE_FLOOR)


# ---------------------------------------------------------------- Step 6
def decide(pit_time, stay_time, laps_remaining, err_per_lap, confident=True, k=1.0):
    """
    pit_time / stay_time: predicted total time to the flag for each option.
    The advantage of pitting is diff = stay_time - pit_time (positive = pit wins).
    The advice is only trusted if |diff| is bigger than a margin built from the
    forecast error: margin = k * err_per_lap * sqrt(laps_remaining).
    """
    diff = stay_time - pit_time
    margin = k * err_per_lap * math.sqrt(laps_remaining)

    if not confident:
        decision, why = "ASK", "forecast not confident (too few laps of data, or too far ahead)"
    elif abs(diff) < margin:
        decision, why = "ASK", "difference is smaller than the forecast error: ask the driver"
    elif diff > 0:
        decision, why = "PIT", "pitting is faster by more than the margin"
    else:
        decision, why = "STAY", "staying out is faster by more than the margin"

    return {"decision": decision, "diff": round(diff, 3),
            "margin": round(margin, 3), "why": why}


if __name__ == "__main__":
    predicted = [95.4, 95.6, 95.8]
    actual = [95.9, 96.3, 96.8]
    r = residuals(predicted, actual)
    print("residuals:", [round(x, 3) for x in r])

    out = corrected_forecast([96.0, 96.2, 96.4], r)
    print("corrected:", [round(x, 3) for x in out["forecast"]], "confident:", out["confident"])

    print("\n--- feel inputs ---")
    for laps in (0, 3, 6):
        m = wear_multiplier_from_feel("going off", laps)
        print(f"'going off' reported {laps} laps ago -> multiplier {m:.4f}")
    print("effective age of a 12-lap-old tyre:", round(12 * wear_multiplier_from_feel("going off"), 2))
    print("rival 'faster' ->", rival_pace_from_feel("faster"), "s per lap")

    print("\n--- forecast error ---")
    print("error with 3 residuals:", round(forecast_error(r), 3))
    print("error with 2 residuals:", round(forecast_error(r[:2]), 3))

    print("\n--- decisions (3 laps left) ---")
    err = forecast_error(r)
    print("gap 15 s case :", decide(pit_time=303.31, stay_time=303.48, laps_remaining=3, err_per_lap=err))
    print("clear-cut case:", decide(pit_time=295.00, stay_time=303.48, laps_remaining=3, err_per_lap=err))
    print("not confident :", decide(pit_time=295.00, stay_time=303.48, laps_remaining=3, err_per_lap=err, confident=False))