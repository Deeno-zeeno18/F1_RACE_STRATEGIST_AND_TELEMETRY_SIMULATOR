from TIRES_AND_CARS import Car, Tyre, compound_pace_offset, pit_stop_penalty, drivers
import copy
compounds = ["SOFT", "MEDIUM", "HARD"]
min_stint = 5   # no tyre set is used for fewer than 5 laps


"""
THIS FUNCTION PRINTS THE TYRE COMPOUND BEFORE THE PIT CHANGES SO
THAT THE PIT LAP GETS A TIME WITH THOSE TYRES BEFORE THE TIME
PENALTY IS ADDED.
"""
def check_tyre(car, label):
    t = car.tyre
    print(f"{label}: compound={t.compound}, age={t.tyre_age}, "
          f"caution={t.specify_tyre_caution_points()}")

"""
THIS FUNCTION TESTS PITTING MECHANISM. 

"""

def testing_pitting_mechanism():
    our_car = Car(compound = "MEDIUM", tyre_age = 0, fuel_load = 100, current_lap_number = 0)
    no_laps = 65

    print(
        f"----------------LAP RESULTS AFTER A SIMULATION OVER A NUMBER OF LAPS IF A TYRE WERE TO PIT ANY LAP----------------")
    lap_results = {}

    for l in range(no_laps):

        laps_remaining = no_laps - our_car.current_lap_number

        # ------------------------------------------------------------
        # PIT DECISION: should_the_car_pit() says WHEN (pit window and
        # tyre life vs laps remaining). On a yes, compare staying out
        # with a stop on each compound, print the totals, then pit on
        # the compound with the lowest total.
        # ------------------------------------------------------------
        if should_the_car_pit(our_car, laps_remaining):
            stay_out = simulate_a_race(our_car, laps_remaining, no_laps)
            chosen, totals = best_compound_to_pit_on(our_car, laps_remaining, no_laps)

            print(f"---- LAP {l + 1} DECISION POINT ----")
            print(f"     stay out : {stay_out:.2f}s")
            for compound, t in totals.items():
                print(f"     pit on {compound:6}: {t:.2f}s")
            print(f"     chosen: {chosen}")

            check_tyre(our_car, "BEFORE PIT")
            lap_time = our_car.pit_penalty(new_compound=chosen, new_tyre_age=0)
            check_tyre(our_car, "AFTER PIT")
        else:
            lap_time = our_car.calculate_lap_time()

        print(f"LAP {l + 1} ---------------- {lap_time:.2f}seconds")

        lap_results[l + 1] = round(lap_time, 2)

        our_car.increase_lap_number()
        our_car.fuel_unit_reduction(no_laps)
        our_car.tyre.increase_tyre_age()

    renamed_lap_results = {}

    for key in lap_results.keys():
        new_key = "LAP {}".format(key)
        renamed_lap_results[new_key] = lap_results[key]
    print(f"--------------LAP RESULTS-------------------")
    print(f"{renamed_lap_results}")

"""

PHASE 1 OF PITTING STRATEGY
CALCULATING WHEN A CAR SHOULD PIT BASED ON THE CAUTION POINT FOR EACH TYRE.
PIT WINDOW WILL OPEN IF THE CAR LAPS ARE TWO BEHIND THE CAUTION POINT AND 
THE DRIVER WOULD BE FORCED TO PIT IF THE LAPS ARE TWO AFTER THE PIT CAUTION POINT.
IT USES THE TECHNIQUE WHEREBY THE CURRENT TYRE IS CALLED AND THEN TWO CONDITIONS ARE REQUIRED.
1. THE LAPS LEFT WHEN THE TYRE REACHES ITS CAUTION POINT SHOULD BE MORE THAN THE AGE OF THE TYRE
2. THE PIT WINDOW WILL OPEN IF THE TYRES AGE IS GREATER THAN OR EQUAL TO THE EARLIEST PIT LAP RECOMMENDED.
"""
def should_the_car_pit(car,laps_remaining):
    critical_allowable_margin = 2
    early_pit_margin = 2
    tyre = car.tyre
    caution_point = tyre.specify_tyre_caution_points()

    lives_left = (caution_point + critical_allowable_margin) - tyre.tyre_age

    in_trouble_cannot_end = laps_remaining > lives_left
    in_pit_window = tyre.tyre_age >= (caution_point - early_pit_margin)

    return in_trouble_cannot_end and in_pit_window


"""
TESTING THE FUNCTION WITH
INTERESTING USE CASES.
INT THE TUPLE FIRST NUMBER IS THE TYRE AGE NEXT IS LAPS REMAINING
NAD THE THIRD IS EXPECTED VALUE
"""
def test_should_the_car_pit():
    total_laps = 60
    cases = [(26, 11, True),(28, 3, True),(27, 1, False),(15, 40, False)]
    for the_age, remaining, expected in cases:
        car = Car(compound="MEDIUM", tyre_age=the_age, fuel_load=100,
                  current_lap_number=total_laps - remaining)
        result = should_the_car_pit(car, remaining)
        status = "ALRIGHT" if result == expected else "CHECK"
        print(f"age={the_age}, remaining={remaining}, expected={expected}, got={result} -> {status}")


"""
RUNNING A LAP SIMULATION
ON ONE CAR ON A NUMBER OF RACES. We will import
the copy module to help us simulate many cars and 
CALCULATE TOTAL LAP TIME FOR A RACE. AT THE END OF EVERY RACE 
THE LAP INCREASES, FUEL DECREASES BASED ON THE CURRENT NUMBER OF LAPS LEFT AND 
"""
def simulate_a_race(car, number_of_laps, race_laps):
    simulated_cars = copy.deepcopy(car)
    total_stored_time = 0

    for lap in range(number_of_laps):
        lap_time = simulated_cars.calculate_lap_time()
        total_stored_time += lap_time
        simulated_cars.increase_lap_number()
        simulated_cars.fuel_unit_reduction(race_laps)
        simulated_cars.tyre.increase_tyre_age()



    return total_stored_time



"""
CREATING A FUNCTION THAT GIVES YOU THE AMOUNT OF TIME LEFT
AFTER YOU PITTED AND IT WORKS ON THE COPIED PITTED CAR.
THIS FUNCTION WILL GET THE LAP TIME WHEN THE CAR WAS IN PIT AND THEN ADD UP TO 
THE LAP TIME FOR THE REST OF THE RACE.
"""
def time_if_pit_now(car, laps_remaining, race_laps, new_compound):
    pit_car = copy.deepcopy(car)

    # this lap: new tyre fitted, 22s added
    pit_lap_time = pit_car.pit_penalty(new_compound=new_compound, new_tyre_age=0)

    # the same three updates your loop does after every lap
    pit_car.increase_lap_number()
    pit_car.fuel_unit_reduction(race_laps)
    pit_car.tyre.increase_tyre_age()

    # the rest of the race: one lap fewer, because the pit lap was one of them
    rest_of_race = simulate_a_race(pit_car, laps_remaining - 1, race_laps)

    return pit_lap_time + rest_of_race

"""
    PICKS THE COMPOUND THAT GIVES THE FASTEST FINISH IF THE CAR PITS NOW.
    For each compound it simulates a stop (22s penalty, new tyre at age 0)
    followed by the rest of the race, using time_if_pit_now().
    Returns (best_compound, {compound: total_time}).
    LIMIT: it assumes the new tyre runs to the flag with no further stops,
    so it is only reliable when few laps remain.
"""
def best_compound_to_pit_on(car, laps_remaining, race_laps):
    results = {}
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        results[compound] = time_if_pit_now(car, laps_remaining, race_laps, compound)

    best = min(results, key=results.get)
    return best, results


"""
PLAYS A WHOLE RACE FROM A STRATEGY AND RETURNS THE TOTAL TIME.
strategy = list of (compound, laps on that tyre set), in race order.
The lap a new set is fitted counts as its first lap (22s penalty added).
THIS ALLOWS FOR A PARTICULAR STRATEGY TO BE WORKED OUT BEFORE SELCETING THE BEST STRATEGY BY COMPARING FINISHING TIMES.
"""
def simulate_strategy(strategy, race_laps, start_fuel=100, driver="balanced"):
    records = simulate_strategy_laps(strategy, race_laps, start_fuel, driver)
    return sum(record["lap_time"] for record in records)


def simulate_strategy_laps(strategy, race_laps, start_fuel=100, driver="balanced"):
    """
    PLAYS A WHOLE RACE FROM A STRATEGY AND RETURNS ONE RECORD (DICT) PER LAP:
    lap number, compound, tyre age, whether it was a pit lap, the parts of the
    lap time, and the lap time itself. Full precision; round only for display.
    """
    planned_laps = sum(stint_laps for compound, stint_laps in strategy)
    if planned_laps != race_laps:
        raise ValueError(f"Strategy covers {planned_laps} laps, race is {race_laps}")

    car = Car(compound=strategy[0][0], tyre_age=0, fuel_load=start_fuel, current_lap_number=0, driver=driver)
    records = []
    lap_number = 0

    for stint_number in range(len(strategy)):
        compound, stint_laps = strategy[stint_number]

        for lap_in_stint in range(stint_laps):
            is_pit_lap = stint_number > 0 and lap_in_stint == 0

            if is_pit_lap:
                lap_time = car.pit_penalty(new_compound=compound, new_tyre_age=0)
            else:
                lap_time = car.calculate_lap_time()

            parts = car._raw_lap_parts()
            lap_number += 1
            records.append({
                "lap": lap_number,
                "compound": compound,
                "tyre_age": car.tyre.tyre_age,
                "pit_lap": is_pit_lap,
                "base": parts["base time"],
                "compound_pace": parts["compound pace"],
                "driver_pace": parts["driver pace"],
                "wear": parts["wear degradation"],
                "fuel": parts["fuel degradation"],
                "pit_stop": round(lap_time - sum(parts.values()), 6),
                "lap_time": lap_time,
            })

            car.increase_lap_number()
            car.fuel_unit_reduction(race_laps)
            car.tyre.increase_tyre_age()

    return records




"""
TIME COST OF ONE STINT THAT DEPENDS ON THE TYRE: compound pace plus
wear, summed over the tyre's ages 0 .. laps-1. Fuel, the 92s base and
the pit stop penalty are not included.
THIS FUNCTION CALCULATES THE COST IN WEAR PENALTY DEPENDING ON THE TYRE CHOSEN.
"""
def stint_cost(compound, laps):

    cost = 0
    for age in range(laps):
        tyre = Tyre(compound, age)
        cost += compound_pace_offset[compound] + tyre.get_wear_penalty()
    return cost



"""
FINDS THE FASTEST STRATEGY FOR ANY NUMBER OF STOPS (dynamic programming).
best[n] holds the cheapest way to cover exactly n laps, kept separately
for each set of compounds used so far, so the two-compound rule can be
checked at the end. Costs cover tyre-dependent time and pit stops only;
fuel and base time are the same for every strategy.
cost_table[compound][L] = tyre-dependent cost of a stint of L laps
"""
def best_strategy_any_stops(race_laps, pit_loss=pit_stop_penalty, driver="balanced"):
    profile = drivers[driver]
    cost_table = {}
    for compound in compounds:
        costs = [0.0]
        for age in range(race_laps):
            try:
                tyre = Tyre(compound, age)
            except Exception:
                break                      # the tyre is not allowed to get this old
            costs.append(costs[-1] + compound_pace_offset[compound] + profile["pace"]
                         + tyre.get_wear_penalty(profile["tyre_care"]))
        cost_table[compound] = costs
    best = [dict() for _ in range(race_laps + 1)]
    best[0][frozenset()] = (0.0, [])          # 0 laps covered, nothing used, cost 0

    for n in range(min_stint, race_laps + 1):
        for last in range(min_stint, n + 1):          # length of the last stint
            before = n - last
            if not best[before]:
                continue                               # 'before' laps can't be covered
            stop_cost = pit_loss if before > 0 else 0.0

            for compound in compounds:
                if last >= len(cost_table[compound]):
                    continue                   # this compound cannot last that long

                for used, (cost, strategy) in best[before].items():
                    new_used = used | {compound}
                    new_cost = cost + cost_table[compound][last] + stop_cost
                    if new_used not in best[n] or new_cost < best[n][new_used][0]:
                        best[n][new_used] = (new_cost, strategy + [(compound, last)])

    # keep only plans that used at least two different compounds
    candidates = [(cost, strategy) for used, (cost, strategy) in best[race_laps].items()
                  if len(used) >= 2]
    if not candidates:
        return None            # race too short for two tyre sets of min_stint laps
    best_cost, best_strategy = min(candidates)
    return best_strategy



def max_stint_laps(compound, race_laps):
    """
    HOW MANY LAPS THE Tyre CLASS ALLOWS ON ONE SET OF THIS COMPOUND
    (the same limit that best_strategy_any_stops respects).
    """
    laps = 0
    for age in range(race_laps):
        try:
            Tyre(compound, age)
        except Exception:
            break
        laps += 1
    return laps


def evaluate_strategy(user_strategy, race_laps, driver="balanced"):
    """
    COMPARES A USER-SUGGESTED STRATEGY WITH THE BEST STRATEGY FOUND FOR THAT DRIVER.
    Checks the two-compound rule, the lap count and the tyre age limit first.
    Returns a dict, so the dashboard can use it later.
    """
    compounds_used = set(compound for compound, laps in user_strategy)
    if len(compounds_used) < 2:
        return {"valid": False, "reason": "A dry race needs at least two different compounds."}

    planned_laps = sum(laps for compound, laps in user_strategy)
    if planned_laps != race_laps:
        return {"valid": False, "reason": f"Stints cover {planned_laps} laps, race is {race_laps}."}

    for compound, stint_laps in user_strategy:
        limit = max_stint_laps(compound, race_laps)
        if stint_laps > limit:
            return {"valid": False,
                    "reason": f"A {compound} set can only run {limit} laps; {stint_laps} were planned."}

    best = best_strategy_any_stops(race_laps, driver=driver)

    if best is None:
        return {"valid": False,
                "reason": f"The optimizer needs a race of at least {2 * min_stint} laps."}

    your_total = simulate_strategy(user_strategy, race_laps, driver=driver)
    best_total = simulate_strategy(best, race_laps, driver=driver)

    return {"valid": True,
            "driver": driver,
            "your_strategy": user_strategy,
            "your_total": round(your_total, 2),
            "best_strategy": best,
            "best_total": round(best_total, 2),
            "time_lost": round(your_total - best_total, 2)}


def simulate_duel(my_car, rival_car, gap, laps_remaining, race_laps,
                  my_decision="stay", rival_edge=0.0):
    """
    TWO CARS, ONE GAP, THE LAPS THAT ARE LEFT.
    gap = seconds the rival is behind me (negative = he is ahead).
    my_decision = "stay" or a compound name to pit on right now.
    rival_edge = extra seconds per lap the rival is faster than his car's lap time says.
    The rival stays out. A pass happens when the gap drops to zero or below.
    Both cars are copied, so the real cars are untouched.
    """
    me = copy.deepcopy(my_car)
    rival = copy.deepcopy(rival_car)
    pass_lap = None
    my_total = 0.0

    for lap in range(1, laps_remaining + 1):
        if lap == 1 and my_decision != "stay":
            my_time = me.pit_penalty(new_compound=my_decision, new_tyre_age=0)
        else:
            my_time = me.calculate_lap_time()
        rival_time = rival.calculate_lap_time() - rival_edge

        was_ahead = gap > 0
        gap += rival_time - my_time          # if I am slower, the gap shrinks
        if was_ahead and gap <= 0 and pass_lap is None:
            pass_lap = lap

        my_total += my_time
        for car in (me, rival):
            car.increase_lap_number()
            car.fuel_unit_reduction(race_laps)
            car.tyre.increase_tyre_age()

    return {"decision": my_decision,
            "final_gap": round(gap, 2),
            "still_ahead": gap > 0,
            "pass_lap": pass_lap,
            "my_time": round(my_total, 2)}



def best_one_stop(race_laps):
    """
    TRIES EVERY ONE-STOP STRATEGY AND RETURNS THE FASTEST.
    first compound x second compound x lap of the stop.
    The two compounds must differ (F1 dry-race rule).
    """
    best_strategy = None
    best_total = None

    for first in compounds:
        for second in compounds:
            if first == second:
                continue

            for first_stint in range(min_stint, race_laps - min_stint + 1):
                strategy = [(first, first_stint), (second, race_laps - first_stint)]
                total = simulate_strategy(strategy, race_laps)

                if best_total is None or total < best_total:
                    best_total = total
                    best_strategy = strategy

    return best_strategy, best_total


def best_two_stop(race_laps, step=3):
    """
    TRIES EVERY TWO-STOP STRATEGY ON A GRID AND RETURNS THE FASTEST.
    Three tyre sets, each used for at least MIN_STINT laps. At least two
    different compounds are required. 'step' skips laps to keep it fast.
    """
    best_strategy = None
    best_total = None

    for first in compounds:
        for second in compounds:
            for third in compounds:
                if first == second == third:
                    continue

                for a in range(min_stint, race_laps - 2 * min_stint + 1, step):
                    for b in range(min_stint, race_laps - a - min_stint + 1, step):
                        c = race_laps - a - b
                        strategy = [(first, a), (second, b), (third, c)]
                        total = simulate_strategy(strategy, race_laps)

                        if best_total is None or total < best_total:
                            best_total = total
                            best_strategy = strategy

    return best_strategy, best_total

if __name__ == "__main__":
    test_should_the_car_pit()
    testing_pitting_mechanism()

    test_car = Car(compound="MEDIUM", tyre_age=0, fuel_load=100, current_lap_number=0)
    total_time_covered = simulate_a_race(test_car, 26, 65)
    print(f"Simulated 26 laps: {total_time_covered:.2f}s")
    print(f"Original car untouched: age={test_car.tyre.tyre_age}, lap={test_car.current_lap_number}")

    """ ------------------------------------------------------------
     CHECK 1: HOW MUCH OF A LAP IS TYRE WEAR?
     Same MEDIUM car at age 0 and age 26 (full tank in both), so any
     change in the numbers is caused by tyre age alone.
     Expect: base and fuel identical, wear larger at age 26,
    and total at age 0 equal to lap 1 of the race (94.52).
     ------------------------------------------------------------"""
    for age in (0, 26):
        c = Car(compound="MEDIUM", tyre_age=age, fuel_load=100, current_lap_number=0)
        print(f"age={age}:", c.lap_time_breakdown())


    """------------------------------------------------------------
     CHECK 2: DO THE COMPOUNDS DIFFER IN PACE?
     Fresh tyres (age 0) with the same fuel, one car per compound.
     Any gap in the totals is down to the compound itself.
     Expect: SOFT fastest, HARD slowest. If all three totals match,
     the lap time model has no per-compound pace yet.
     ------------------------------------------------------------"""
    for compound in ("SOFT", "MEDIUM", "HARD"):
        c = Car(compound=compound, tyre_age=0, fuel_load=100, current_lap_number=0)
        print(f"{compound}:", c.lap_time_breakdown())


    """ ------------------------------------------------------------
     CHECK 3: DOES A STRATEGY RECIPE REPRODUCE THE RACE WE RAN?
     The first recipe is the race printed above, so its total should
     equal the sum of all 65 printed laps. The second is a one-stop
     alternative for comparison.
     ------------------------------------------------------------"""

    two_stop = [("MEDIUM", 26), ("HARD", 31), ("SOFT", 8)]
    one_stop = [("MEDIUM", 26), ("HARD", 39)]
    print(f"Two stop: {simulate_strategy(two_stop, 65):.2f}s")
    print(f"One stop: {simulate_strategy(one_stop, 65):.2f}s")


    best_strategy, best_total = best_one_stop(65)
    print(f"Best one-stop: {best_strategy} -----> {best_total:.2f}s")


    print(round(simulate_strategy([("MEDIUM", 31), ("HARD", 34)], 65), 3))
    print(round(simulate_strategy([("HARD", 34), ("MEDIUM", 31)], 65), 3))

    two_best, two_total = best_two_stop(65)
    print(f"Best two-stop: {two_best} -> {two_total:.2f}s")

    # ------------------------------------------------------------
    # CHECK 4: WOULD A THIRD STOP PAY?
    # The three recipes swap more and more MEDIUM laps for short SOFT
    # stints. Compare each total with the best two-stop (6121.26s).
    # ------------------------------------------------------------
    for strat in ([("SOFT", 16), ("SOFT", 16), ("SOFT", 16), ("MEDIUM", 17)],
                  [("SOFT", 16), ("SOFT", 16), ("MEDIUM", 16), ("MEDIUM", 17)],
                  [("SOFT", 16), ("MEDIUM", 16), ("MEDIUM", 16), ("MEDIUM", 17)]):
        print(strat, f"{simulate_strategy(strat, 65):.2f}s")


    """------------------------------------------------------------
     CHECK 5: DOES stint_cost AGREE WITH THE FULL SIMULATION?
     Fuel and base time are identical in any two 65-lap strategies, so
     the gap between them = difference in stint costs + 22s per extra stop.
     ------------------------------------------------------------
    """
    three_stop = [("SOFT", 16), ("SOFT", 16), ("SOFT", 16), ("MEDIUM", 17)]
    two_stop_best = [("MEDIUM", 23), ("MEDIUM", 26), ("SOFT", 16)]

    gap_simulated = simulate_strategy(three_stop, 65) - simulate_strategy(two_stop_best, 65)

    cost_three = sum(stint_cost(c, n) for c, n in three_stop) + pit_stop_penalty * 3
    cost_two = sum(stint_cost(c, n) for c, n in two_stop_best) + pit_stop_penalty * 2
    gap_formula = cost_three - cost_two

    print(f"gap by simulation: {gap_simulated:.4f}s | gap by stint_cost: {gap_formula:.4f}s")

    """ ------------------------------------------------------------
    CHECK 6: BEST STRATEGY FOR ANY NUMBER OF STOPS
    Must be at least as fast as the three-stop recipe (6118.44s),
    because that recipe is one of the strategies being searched.
    ------------------------------------------------------------"""

    dp_best = best_strategy_any_stops(65)
    print(f"Best any-stop: {dp_best}")
    print(f"   stops: {len(dp_best) - 1} | total: {simulate_strategy(dp_best, 65):.2f}s")


    # ------------------------------------------------------------
    # CHECK 7: USER SUGGESTS A STRATEGY, THE OPTIMIZER COMPARES
    # First is your usual medium-then-hard one-stop; second breaks the rule.
    # ------------------------------------------------------------
    print(evaluate_strategy([("MEDIUM", 31), ("HARD", 34)], 65))
    print(evaluate_strategy([("HARD", 65)], 65))
    result = evaluate_strategy([("MEDIUM", 31), ("HARD", 34)], 65)
    for key, value in result.items():
        print(f"  {key}: {value}")


    # ------------------------------------------------------------
    # CHECK 8: PER-LAP RECORDS FOR THE DASHBOARD
    # Lap 1 is a normal lap; lap 21 is the first stop of the best strategy.
    # ------------------------------------------------------------
    laps = simulate_strategy_laps(dp_best, 65)
    print(len(laps), "laps recorded")
    print(laps[0])
    print(laps[20])
    print(f"Sum of lap times: {sum(r['lap_time'] for r in laps):.2f}s")


    print(evaluate_strategy([("SOFT", 4), ("MEDIUM", 4)], 8))
    print(evaluate_strategy([("MEDIUM", 50), ("HARD", 15)], 65))
    for c in compounds:
        print(c, "max laps on one set:", max_stint_laps(c, 65))


    # ------------------------------------------------------------
    # CHECK 9: HOW DOES THE BEST STRATEGY CHANGE WITH THE PIT STOP PENALTY?
    # Only the search's pit cost changes here, so strategies are printed, not totals.
    # ------------------------------------------------------------
    for loss in (14, 18, 22, 26, 30, 34):
        best = best_strategy_any_stops(65, pit_loss=loss)
        print(f"pit loss {loss}s -> {len(best) - 1} stops: {best}")


    # ------------------------------------------------------------
    # CHECK 9: DOES THE DRIVER CHANGE THE WEAR?
    # SOFT at age 14, the last lap of the best strategy's SOFT stints.
    # ------------------------------------------------------------
    for name in ("balanced", "smooth", "aggressive"):
        c = Car(compound="SOFT", tyre_age=14, fuel_load=100, current_lap_number=0, driver=name)
        print(name, c.lap_time_breakdown())


    # ------------------------------------------------------------
    # CHECK 10: DOES THE BEST STRATEGY CHANGE WITH THE DRIVER?
    # ------------------------------------------------------------
    for name in ("balanced", "smooth", "aggressive"):
        best = best_strategy_any_stops(65, driver=name)
        print(name, f"{len(best) - 1} stops:", best,
              f"{simulate_strategy(best, 65, driver=name):.2f}s")


    # ------------------------------------------------------------
    # CHECK 11: DUEL. Leader on HARD age 40, rival on SOFT age 8, 3 laps to the flag.
    # Both cars have the same fuel, so fuel cancels out of the gap.
    # ------------------------------------------------------------
    for gap in (15.0, 25.0):
        for decision in ("stay", "SOFT"):
            me = Car(compound="HARD", tyre_age=40, fuel_load=100, current_lap_number=62)
            rival = Car(compound="SOFT", tyre_age=8, fuel_load=100, current_lap_number=62)
            print(f"gap {gap}s:", simulate_duel(me, rival, gap, 3, 65, decision))