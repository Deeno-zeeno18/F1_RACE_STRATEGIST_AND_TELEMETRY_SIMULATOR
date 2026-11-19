from flask import Flask, jsonify, request, render_template
from PIT_STRATEGY import (best_strategy_any_stops, simulate_strategy,
                          simulate_strategy_laps, evaluate_strategy, compounds, drivers)
from LIVE_SHOW import RaceState, calibrate
from SAFETY_CAR_VSC import advise_with_track, TRACK_STATES
from DUEL import duel

app = Flask(__name__)


def parse_strategy(text):
    """
    TURNS 'MEDIUM:31,HARD:34' INTO [('MEDIUM', 31), ('HARD', 34)].
    Raises ValueError with a readable message if the text is wrong.
    """
    strategy = []
    for part in text.split(","):
        try:
            compound, laps = part.split(":")
            laps = int(laps)
        except ValueError:
            raise ValueError("Could not read the strategy. Use e.g. strategy=MEDIUM:31,HARD:34")
        compound = compound.strip().upper()
        if compound not in compounds:
            raise ValueError(f"Unknown compound '{compound}'. Use SOFT, MEDIUM or HARD.")
        strategy.append((compound, laps))
    return strategy


def read_driver():
    """READS ?driver= FROM THE ADDRESS (default balanced) AND CHECKS IT EXISTS."""
    driver = request.args.get("driver", "balanced")
    if driver not in drivers:
        raise ValueError(f"Unknown driver '{driver}'. Use one of: {', '.join(drivers)}.")
    return driver


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/best")
def api_best():
    """BEST STRATEGY FOR A RACE LENGTH AND DRIVER, E.G.  /api/best?laps=65&driver=smooth"""
    race_laps = request.args.get("laps", 65, type=int)
    try:
        driver = read_driver()
    except ValueError as problem:
        return jsonify({"error": str(problem)}), 400

    best = best_strategy_any_stops(race_laps, driver=driver)
    if best is None:
        return jsonify({"error": "Race too short: the optimizer needs at least 10 laps."}), 400

    return jsonify({"race_laps": race_laps,
                    "driver": driver,
                    "strategy": best,
                    "stops": len(best) - 1,
                    "total": round(simulate_strategy(best, race_laps, driver=driver), 2)})


@app.route("/api/evaluate")
def api_evaluate():
    """COMPARES A SUGGESTED STRATEGY WITH THE BEST ONE, E.G.  /api/evaluate?laps=65&strategy=MEDIUM:31,HARD:34&driver=smooth"""
    race_laps = request.args.get("laps", 65, type=int)
    try:
        driver = read_driver()
        strategy = parse_strategy(request.args.get("strategy", ""))
    except ValueError as problem:
        return jsonify({"error": str(problem)}), 400

    return jsonify(evaluate_strategy(strategy, race_laps, driver=driver))


@app.route("/api/laps")
def api_laps():
    """ONE RECORD PER LAP FOR A STRATEGY, E.G.  /api/laps?laps=65&strategy=MEDIUM:20,SOFT:15,SOFT:15,SOFT:15&driver=smooth"""
    race_laps = request.args.get("laps", 65, type=int)
    try:
        driver = read_driver()
        strategy = parse_strategy(request.args.get("strategy", ""))
        records = simulate_strategy_laps(strategy, race_laps, driver=driver)
    except ValueError as problem:
        return jsonify({"error": str(problem)}), 400

    rounded = [{key: round(value, 3) if isinstance(value, float) else value
                for key, value in record.items()} for record in records]
    return jsonify({"race_laps": race_laps, "driver": driver, "laps": rounded})


# ----------------------------------------------------------------- race engineer
def parse_track_plan(text):
    """
    TURNS 'SC:4,VSC:2' INTO [('SC', 4), ('VSC', 2)]. Empty text means green flag.
    """
    plan = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            name, laps = part.split(":")
            laps = int(laps)
        except ValueError:
            raise ValueError("Could not read the track plan. Use e.g. SC:4 or VSC:2,SC:3")
        name = name.strip().upper()
        if name not in TRACK_STATES:
            raise ValueError(f"Unknown track state '{name}'. Use SC, VSC or GREEN.")
        if laps < 1:
            raise ValueError("Each part of the track plan needs at least 1 lap.")
        plan.append((name, laps))
    return plan


def read_number(data, key, kind=float, minimum=None):
    """READS ONE NUMBER FROM THE JSON BODY WITH A READABLE ERROR IF IT IS MISSING OR WRONG."""
    try:
        value = kind(data[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"'{key}' is missing or is not a number.")
    if minimum is not None and value < minimum:
        raise ValueError(f"'{key}' must be at least {minimum}.")
    return value


def read_state(data):
    """
    READS THE RACE STATE FROM A JSON BODY. Returns (state, plan, feel, laps_since_feel).
    Raises ValueError with a readable message for anything that does not make sense.
    """
    driver = data.get("driver", "balanced")
    if driver not in drivers:
        raise ValueError(f"Unknown driver '{driver}'. Use one of: {', '.join(drivers)}.")
    compound = str(data.get("compound", "")).upper()
    if compound not in compounds:
        raise ValueError("Unknown compound. Use SOFT, MEDIUM or HARD.")
    used = [str(c).upper() for c in data.get("compounds_used", [])]
    if any(c not in compounds for c in used):
        raise ValueError("Unknown compound in compounds_used.")

    race_laps = read_number(data, "race_laps", int, 10)
    lap = read_number(data, "lap", int, 0)
    tyre_age = read_number(data, "tyre_age", int, 0)
    if lap >= race_laps:
        raise ValueError("The race is already over: 'lap' must be less than 'race_laps'.")
    if tyre_age > lap:
        raise ValueError("The tyre cannot have run more laps than the race has had.")

    raw_laps = data.get("recent_laps") or []
    try:
        recent = [float(x) for x in raw_laps]
    except (TypeError, ValueError):
        raise ValueError("'recent_laps' must be a list of lap times in seconds.")
    if any(not 60 < x < 200 for x in recent):
        raise ValueError("Lap times should be in seconds (for example 95.4).")

    feel = data.get("feel") or None
    laps_since_feel = read_number(data, "laps_since_feel", int, 0) if feel else 0
    plan = parse_track_plan(data.get("track", ""))

    state = RaceState(driver=driver, compound=compound, tyre_age=tyre_age, lap=lap,
                      race_laps=race_laps, compounds_used=set(used),
                      recent_laps=recent, started_on_pit=bool(data.get("on_pit_set")))
    return state, plan, feel, laps_since_feel


def read_rival(data, me):
    """READS THE RIVAL (rival_driver, rival_compound, rival_tyre_age, rival_compounds_used) AND THE GAP."""
    driver = data.get("rival_driver", "balanced")
    if driver not in drivers:
        raise ValueError(f"Unknown rival driver '{driver}'. Use one of: {', '.join(drivers)}.")
    compound = str(data.get("rival_compound", "")).upper()
    if compound not in compounds:
        raise ValueError("Unknown rival compound. Use SOFT, MEDIUM or HARD.")
    used = [str(c).upper() for c in data.get("rival_compounds_used", [])]
    if any(c not in compounds for c in used):
        raise ValueError("Unknown compound in rival_compounds_used.")
    age = read_number(data, "rival_tyre_age", int, 0)
    if age > me.lap:
        raise ValueError("The rival's tyre cannot have run more laps than the race has had.")
    gap = read_number(data, "gap", float)
    if abs(gap) > 120:
        raise ValueError("The gap should be in seconds (for example 3.5).")
    rival = RaceState(driver=driver, compound=compound, tyre_age=age, lap=me.lap,
                      race_laps=me.race_laps, compounds_used=set(used))
    return rival, gap


def calibration_summary(cal):
    """WHAT THE PAGE SHOWS ABOUT HOW THE ENGINEER USED THE LAP TIMES AND THE DRIVER'S REPORT."""
    return {key: cal[key] for key in ("multiplier", "source", "prior", "laps_used", "rms_before",
                                      "rms_after", "error_per_lap", "warning", "note")}


def ask_questions(decision, state, feel):
    """WHAT THE ENGINEER NEEDS FROM THE DRIVER WHEN THE CALL IS TOO CLOSE."""
    if decision != "ASK":
        return []
    questions = []
    if not feel:
        questions.append({"kind": "feel", "text": "How does the tyre feel?"})
    if len(state.recent_laps) < 3:
        questions.append({"kind": "laps",
                          "text": "Send your last 3 or 4 lap times on this set so I can check the model against the car."})
    if not questions:
        questions.append({"kind": "decide",
                          "text": "The options are within the forecast error. Your call."})
    return questions


@app.route("/api/advise", methods=["POST"])
def api_advise():
    """
    THE RACE ENGINEER. POST a JSON body like
      {"driver": "balanced", "compound": "MEDIUM", "tyre_age": 20, "lap": 20, "race_laps": 65,
       "compounds_used": ["MEDIUM"], "recent_laps": [95.1, 95.4, 95.9], "on_pit_set": false,
       "feel": "going off", "laps_since_feel": 0, "track": "SC:4"}
    Answers PIT, STAY or ASK with the numbers behind it. The real lap times are used to
    calibrate how fast this tyre wears, and that carries through the whole rest of the race.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send the race state as a JSON body."}), 400

    try:
        state, plan, feel, laps_since_feel = read_state(data)
        cal = calibrate(state, feel, laps_since_feel)
        track = advise_with_track(state, plan, wear_mult=cal["multiplier"], err_per_lap=cal["error_per_lap"])
    except ValueError as problem:
        return jsonify({"error": str(problem)}), 400
    except Exception as problem:                       # e.g. a tyre age the Tyre class refuses
        return jsonify({"error": f"The engineer could not run this situation: {problem}"}), 400

    if track.get("advice"):                            # green flag: the stop window's own wording
        advice = track["advice"]
    elif track["decision"] == "PIT":
        advice = track["best_now"]["how"]
    elif track["decision"] == "STAY":
        advice = track["best_later"]["how"] if track["best_later"] else "stay out"
    else:
        advice = "too close to call"

    headline = {"decision": track["decision"], "advice": advice, "why": track["why"],
                "diff": track["diff"], "margin": track["margin"]}
    return jsonify({"headline": headline, "track": track, "calibration": calibration_summary(cal),
                    "questions": ask_questions(track["decision"], state, feel),
                    "laps_remaining": state.laps_remaining})


@app.route("/api/duel", methods=["POST"])
def api_duel():
    """
    YOU AGAINST ONE RIVAL. The same JSON body as /api/advise, plus
      "gap": 3.5   (seconds the rival is BEHIND you; negative = he is ahead),
      "rival_driver", "rival_compound", "rival_tyre_age", "rival_compounds_used".
    Every choice of yours is scored against every choice of his, and the answer holds up
    whatever he does. A safety car wipes out the gap (the field closes up); a VSC does not.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send the race state as a JSON body."}), 400

    try:
        state, plan, feel, laps_since_feel = read_state(data)
        rival, gap = read_rival(data, state)
        cal = calibrate(state, feel, laps_since_feel)
        result = duel(state, rival, gap, plan, my_wear=cal["multiplier"], err_per_lap=cal["error_per_lap"])
    except ValueError as problem:
        return jsonify({"error": str(problem)}), 400
    except Exception as problem:
        return jsonify({"error": f"The engineer could not run this situation: {problem}"}), 400

    if result["decision"] == "PIT":
        advice = result["my_options"]["pit"]["how"]
    elif result["decision"] == "STAY":
        advice = result["my_options"]["wait"]["how"]
    else:
        advice = "too close to call"

    headline = {"decision": result["decision"], "advice": advice, "why": result["why"],
                "diff": result["diff"], "margin": result["margin"]}
    return jsonify({"headline": headline, "duel": result, "calibration": calibration_summary(cal),
                    "questions": ask_questions(result["decision"], state, feel),
                    "laps_remaining": state.laps_remaining})


if __name__ == "__main__":
    app.run(debug=True)