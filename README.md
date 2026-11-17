# GTA VI launch-day Twitch viewership forecaster

This program forecasts how many people watch Grand Theft Auto VI live on
Twitch in each hour of its release day, after the first hour. For each
case it is given, it writes one forecast file holding, for hours 2
through 24, five quantiles of that hour's concurrent viewer count.

## How it works

The forecast is anchored on the first hour's viewer count. The shape of
the rest of the day — how viewership typically rises and falls hour by
hour relative to the first hour, and how uncertain that shape is — is
borrowed from the comparable past game launches supplied with each case.
Each comparable contributes its own release-day shape, expressed as log
ratios to its first hour. Comparables are weighted by similarity to the
target launch: how closely their pre-launch trajectory over the previous
day resembles the target's, and how close their first-hour scale is.
Launches that unlocked in the same rolling midnight-local pattern and at
a similar scale therefore count most, which captures both the staggered
unlock waves across time zones and the daily rhythm of viewing. For each
hour, the five forecast quantiles are the weighted quantiles of the
comparable shapes, with a minimum spread that grows over the day so the
bands stay honest when the comparables happen to agree closely. The
program is deterministic: repeated runs on the same case produce
byte-identical output.

## How to run it

The program needs Python 3.13 and only the standard library. It reads
every case file it finds in the input directory and writes one forecast
file per case to the output directory:

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

When the environment variables are unset, it reads from `/input/` and
writes to `/output/`, matching the grading sandbox. Each output file is
a single JSON object with one key, `hours`, holding one row per hour
with the hour number and the five quantile values.

## Files

- `main.py` — the forecaster, the repository's entrypoint.
- `LICENSE` — the MIT licence.
