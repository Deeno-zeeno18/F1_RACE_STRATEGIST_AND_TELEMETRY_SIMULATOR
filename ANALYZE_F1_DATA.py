"""
ANALYZE F1 DATA IS THE FIRST FILE THAT WAS CREATED. IT MAKES USE OF THE FASTF1 MODULE
IN THIS FILE,
1. IT TAKES THE LAP TIMES FOR A COUPLE OF RACES WITH SOME SPECIFIC FAST PACED RACERS
2. IT GIVES A LIST OF RACERS THAT TOOK PART IN THE RACE.
3. IT GIVES TYRE LIFE AND THE COMPOUND UTILIZED AS WELL.
4. IT GIVES INFORMATION ABOUT PIT CHANGE LAPS FOR THAT PARTICULAR RACER,
    WHETHER THERE WERE SAFETY CAR OR VIRTUAL SAFETY CAR SESSIONS.
"""

import fastf1

import os

os.makedirs('cache', exist_ok=True)

fastf1.Cache.enable_cache('cache')

session = fastf1.get_session(2023, 'British', 'R')
session.load()


#Retrieves F1 data


laps = session.laps

# Checking unique Drivers

print(f" THE LIST OF F1 DRIVERS ARE: {laps['Driver'].unique()}")
print(laps[['Driver', 'LapNumber', 'LapTime', 'Compound', 'TyreLife']].head(60))


#Checking messages
print(session.race_control_messages)
