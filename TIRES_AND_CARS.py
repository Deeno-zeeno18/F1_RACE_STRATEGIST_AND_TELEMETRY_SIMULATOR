import math

"""
THIS FILE IS ONE OF THE MOST IMPORTANT FILES. 
1. 
"""

"""
Seconds per lap relative to MEDIUM (negative = faster). Placeholder values;
calibrate against FastF1 stint data later.
TUNED against real teams' choices (BACKTEST.py --sweep, 8 races of 2024): with these values the model's next stop
is within 3 laps of the real one 53% of the time (was 29%) and picks the same next compound 68% (was 20%).
SOFT anywhere from -0.3 to +0.1 scored the same, so -0.3 is the faster-tyre end of what the data allows.
"""

compound_pace_offset = {"SOFT": -0.3, "MEDIUM": 0.0, "HARD": 0.0}


"""STEADY TYRE WEAR, MEASURED: seconds lost per lap of tyre age, from 41 dry races of 2023-24 (the fit gave
# SOFT 0.098 / 0.098, MEDIUM 0.052 / 0.068, HARD 0.057 / 0.057 for the two years). Real tyres lose time from
# the first lap; the cliff below is only for the end of a stint.
"""

linear_wear = {"SOFT": 0.098, "MEDIUM": 0.062, "HARD": 0.057}

# MEASURED: median green-flag stop loss, 41 dry races of 2023-24 (954 stops; 22.6 and 22.7 in the two years)
pit_stop_penalty = 22.7


"""
pace: seconds per lap vs an average driver (negative = faster)
tyre_care: multiplies tyre age in the wear calculation (below 1 = gentler on tyres)
Placeholder values, to calibrate against real data later.
"""
drivers = {"balanced":   {"pace": 0.0,   "tyre_care": 1.0},
           "smooth":     {"pace": 0.15,  "tyre_care": 0.85},
           "aggressive": {"pace": -0.15, "tyre_care": 1.15}}

""" REAL DRIVERS, MEASURED from 21 dry races of 2024 (OpenF1). Pace is seconds per lap against the field's median
driver, and it includes the CAR (a Red Bull driver is fast partly because of his car). tyre_care is
shrunk toward 1.0 and is noisy (about 1000 laps per driver); treat it as a hint, not a fact."""

real_drivers = {
    "VER": {"pace": -1.125, "tyre_care": 1.037},
    "NOR": {"pace": -1.032, "tyre_care": 0.924},
    "LEC": {"pace": -0.905, "tyre_care": 0.874},
    "PIA": {"pace": -0.87, "tyre_care": 1.019},
    "SAI": {"pace": -0.789, "tyre_care": 0.807},
    "PER": {"pace": -0.637, "tyre_care": 1.07},
    "HAM": {"pace": -0.581, "tyre_care": 0.924},
    "RUS": {"pace": -0.566, "tyre_care": 0.838},
    "ALO": {"pace": -0.01, "tyre_care": 0.975},
    "HUL": {"pace": 0.063, "tyre_care": 1.006},
    "COL": {"pace": 0.194, "tyre_care": 0.911},
    "STR": {"pace": 0.224, "tyre_care": 0.994},
    "GAS": {"pace": 0.241, "tyre_care": 1.026},
    "MAG": {"pace": 0.318, "tyre_care": 0.921},
    "ALB": {"pace": 0.333, "tyre_care": 0.918},
    "TSU": {"pace": 0.37, "tyre_care": 0.862},
    "RIC": {"pace": 0.392, "tyre_care": 0.861},
    "LAW": {"pace": 0.427, "tyre_care": 0.968},
    "OCO": {"pace": 0.446, "tyre_care": 0.977},
    "BOT": {"pace": 0.544, "tyre_care": 1.116},
    "ZHO": {"pace": 0.58, "tyre_care": 1.077},
    "SAR": {"pace": 0.589, "tyre_care": 1.147},
}
drivers.update(real_drivers)


"""
THE TYRE CLASS IS CREATED.
A RANDOM TYRE AGE CANNOT BE JUST PASSED IN SO I CREATED A PRE TEST FUNCTION TO MAKE SURE THAT
THE TYRE AGE IS NOTHING OTHER THAN A POSTIVE REASONABLE WHOLE NUMBER.
"""
class Tyre:
    def __init__(self, compound, tyre_age):
        self.compound = compound.upper()
        if validate_tyre_age(tyre_age):
            self.assign_tyre_age(tyre_age)
        else:
            raise Exception('Invalid tyre age')

    """
    ASSIGNS THE TYRE AGE TO THAT OBJECT WHEN CREATED AFTER THE INPUT HAS BEEN CHECKED AND VERIFIED.
    """
    def assign_tyre_age(self, tyre_age):
        self.tyre_age = tyre_age

    """
    INCREASES TYRE AGE BY ONE AFTER EVERY LAP IS COMPLETED
    """
    def increase_tyre_age(self):
        self.tyre_age += 1
        return self.tyre_age

    """
    RETURNS A CLIFF POINT FOR A PARTICULAR TYRE COMPOUND 
    """
    def specify_tyre_caution_points(self):
        caution_point = create_caution_points()
        tyre_caution_point = caution_point[self.compound]
        return tyre_caution_point


    """
    RETURNS A PROBABILITY OR PERCENTAGE OF THE CLOSENESS OF A TYRE TO WEAR I.E ITS SEVERITY. 
    IT USES A CUSTOM LOGISTIC FUNCTION TO CALCULATE THE COEFFICIENT AND SENSITIVITY.
    """
    def predict_wear_severity(self, age=None):
        if age is None:
            age = self.tyre_age
        caution_point = self.specify_tyre_caution_points()
        reaction = solve_for_the_reactiveness(base_line_reaction=0.01, cliff_point=caution_point, tyre_age=5)
        z = reaction * (age - caution_point)
        p_cliff_point = 1 / (1 + math.exp(-z))
        return p_cliff_point

    """
    CALCULATION OF WEAR TIME PENALTY:
    FIRST OF ALL, WE CAP THE MAX PENALTY AT 5 SECONDS. BUT, UPON REALIZING THAT 
    PENALTIES COULD INCUR MORE DEGRADATION, I DECIDED TO ADD A QUADRATIC FUNCTION THAT
    EXTENDS THE WEAR PENALTY DUE TO THE TYRE USAGE.
    """
    def get_wear_penalty(self, tyre_care=1.0):
        base_wear_penalty = 5.0
        cliff_growth = 0.035

        age = self.tyre_age * tyre_care            # effective age
        caution_point = self.specify_tyre_caution_points()
        penalty = base_wear_penalty * self.predict_wear_severity(age)
        penalty += linear_wear[self.compound] * age          # steady wear, measured

        if age > caution_point:
            laps_past = age - caution_point
            penalty += cliff_growth * laps_past ** 2

        return penalty




"""
CALCULATES THE REACTIVENESS OR THE SENSITIVITY COEFFICIENT FOR THE  FUNCTION
"""
def solve_for_the_reactiveness(base_line_reaction, cliff_point, tyre_age):
    reaction = (-math.log((1 - base_line_reaction) / base_line_reaction)) / (tyre_age - cliff_point)
    return reaction

"""
RETURNS A MESSAGE VALIDATING WHETHER THE TYRE AGE IS SUITABLE FOR USE
"""
def validate_tyre_age(tyre_age):
    if 0 <= tyre_age <= 50:
        return True
    else:
        return False


"""
RETURNS A DICTIONARY OF THE CAUTION POINTS IN LAPS FOR THE VARIOUS TYRE COMPOUNDS
"""
def create_caution_points():
    caution_points = {"SOFT": 20, "MEDIUM": 30, "HARD": 36}
    return caution_points



"""
THE CAR CLASS IS CREATED. IT HOSTS EVERYTHING ABOUT THE CAR. NEW ADDITIONS APPEAR;
CURRENT LAP NUMBER, DRIVER'S WAY OF DRIVING, FUEL LOAD OF THE CAR.
NB: THE TYRE OBJECT IS CREATED AS SOON AS THE CAR OBJECT IS DONE NOT INDEPENDENT.
"""
class Car:
    def __init__(self, compound, tyre_age, fuel_load, current_lap_number, driver = "balanced"):
        self.tyre =Tyre(compound, tyre_age)
        self.fuel_load = fuel_load
        self.current_lap_number = current_lap_number
        self.driver = drivers[driver]


    """
    INCREASES LAP NUMBER AFTER EVERY LAP HAS RUN
    """
    def increase_lap_number(self):
        self.current_lap_number += 1
        return self.current_lap_number

    """
    CALCULATES THE NUMBER OF LAPS THAT HAVE BEEN RUN FOR AT A PARTICULAR POINT
    """
    def laps_used(self, initial_lap_number):
        if initial_lap_number == 0:
            laps_run = self.current_lap_number
            return laps_run
        elif 0 < initial_lap_number < 80:
            laps_run = self.current_lap_number - initial_lap_number
            return laps_run
        else:
            raise Exception('Invalid initial lap number')


    """
    REDUCES THE FUEL LOAD AFTER EVERY LAP IS COMPLETED
    """
    def fuel_unit_reduction(self,no_of_laps):
        fuel_reduction = 100 / no_of_laps
        self.fuel_load -= fuel_reduction
        return self.fuel_load



    """
    CALCULATING FOR THE FUEL PENALTY
    """
    def calculate_fuel_penalty(self):
        sensitivity = solve_for_sensitivity()
        penalty_by_fuel = sensitivity * (self.fuel_load/ 100)
        return penalty_by_fuel

    """
    FULL-PRECISION PARTS OF ONE LAP TIME. NOTHING IS ROUNDED HERE, SO SIMULATIONS
    ADD UP EXACT VALUES. ROUNDING IS ONLY FOR DISPLAY.
    """
    def _raw_lap_parts(self):
        return {"base time": 92.0,
                "compound pace": compound_pace_offset[self.tyre.compound],
                "driver pace": self.driver["pace"],
                "wear degradation": self.tyre.get_wear_penalty(self.driver["tyre_care"]),
                "fuel degradation": self.calculate_fuel_penalty()}

    """
    BREAKS DOWN THE LAP TIMES INTO THE RESPECTIVE CAUSES BY TYRE DEGRADATION, FUEL BURNING, 
    """
    def lap_time_breakdown(self):
        parts = self._raw_lap_parts()
        shown = {name: round(value, 3) for name, value in parts.items()}
        shown["total"] = round(sum(parts.values()), 2)
        return shown

    def calculate_lap_time(self):
        return sum(self._raw_lap_parts().values())

    """
    PREPARING FOR PIT.
    THIS FUNCTION REPLACES THE OLD TYRES WITH THE NEW ONES ACQUIRED DURING PITS AND SETS A NEW TYR AGE AS WELL.
    """
    def replace_tyre_pit(self, new_compound, new_tyre_age):
        self.tyre = Tyre(new_compound, new_tyre_age)
        return self.tyre

    """
    ADD PENALTY DUE TO THE PIT DELAYS TO THE LAP TIMES
    """
    def pit_penalty(self, new_compound, new_tyre_age=0):
        self.replace_tyre_pit(new_compound, new_tyre_age)
        lap_time_real = self.calculate_lap_time() + pit_stop_penalty
        return lap_time_real


"""
CALCULATE THE FUEL LOAD PENALTY
WHEN FUEL BURNS, IT CAUSES A BIT OF CHANGE IN LAP TIME WHICH I CALCULATED FOR.
"""
def solve_for_sensitivity():
    max_fuel = 1
    max_penalty = 3.5            # MEASURED: 3.3 s (2023) and 3.8 s (2024) for a full tank; includes some track evolution
    return max_penalty / max_fuel












if __name__ == "__main__":
    """
    CREATING ACTUAL TYRE OBJECTS
    """
    first_car = Car(compound = "SOFT", tyre_age = 0, fuel_load = 90, current_lap_number = 5)
    #second_car = Car(compound = "MEDIUM", tyre_age = 1, fuel_load = 100, current_lap_number = 10)
    #third_car = Car(compound = "HARD", tyre_age = 2, fuel_load = 100, current_lap_number = 20)

    no_laps = 65

    print(f"----------------LAP RESULTS AFTER A SIMULATION OVER A NUMBER OF LAPS IF A TYRE WERE NEVER CHANGED----------------")
    lap_results = {}

    for l in range(no_laps):
        lap_time = first_car.calculate_lap_time()
        print(f"LAP {l + 1} ---------------- {lap_time:.2f}seconds")

        lap_results[l + 1] = round(lap_time, 2)
        first_car.increase_lap_number()
        first_car.fuel_unit_reduction(no_laps)
        first_car.tyre.increase_tyre_age()

        """
            MAKING THE RESULTS READABLE
            """
        renamed_lap_results = {}

        for key in lap_results.keys():
            new_key = "LAP {}".format(key)
            renamed_lap_results[new_key] = lap_results[key]
        print(f"--------------LAP RESULTS-------------------")
        print(f"{renamed_lap_results}")