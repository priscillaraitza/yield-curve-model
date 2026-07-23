"""
Yield Curve Nelson-Siegel Analysis
====================================
Fits the Nelson-Siegel model to US Treasury yields over time, decomposes
the curve into Level / Slope / Curvature, and checks whether yield curve
inversions have historically led NBER-dated recessions.

Data source: FRED (Federal Reserve Bank of St. Louis)
Author: Priscilla Raitza
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------

# US Treasury yields at 5 maturities: 3-month, 2-year, 5-year, 10-year, 30-year
YIELD_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO,DGS2,DGS5,DGS10,DGS30"
df = pd.read_csv(YIELD_URL, parse_dates=["observation_date"])
df = df.rename(columns={"observation_date": "date"}).set_index("date")

# Restrict to a window with enough historical recessions to compare against
START_DATE = "1990-01-01"
df_recent = df.loc[START_DATE:]

# Maturities in years, matching the column order above
MATURITIES = np.array([0.25, 2, 5, 10, 30])

# NBER recession indicator (1 = recession, 0 = expansion), monthly
RECESSION_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"
recession_df = pd.read_csv(RECESSION_URL, parse_dates=["observation_date"])
recession_df = recession_df.rename(
    columns={"observation_date": "date", "USREC": "recession"}
).set_index("date")
recession_df_monthly = recession_df.resample("MS").last()

# Recession start dates = months where recession flips from 0 to 1
recession_starts = recession_df_monthly[
    (recession_df_monthly["recession"] == 1)
    & (recession_df_monthly["recession"].shift(1) == 0)
].index


# ---------------------------------------------------------------------------
# 2. NELSON-SIEGEL MODEL
# ---------------------------------------------------------------------------

def nelson_siegel(tau, beta0, beta1, beta2, lam):
    """
    Nelson-Siegel yield curve formula.
    beta0 = level, beta1 = slope, beta2 = curvature, lam = decay parameter.
    """
    term1 = beta1 * (1 - np.exp(-tau / lam)) / (tau / lam)
    term2 = beta2 * ((1 - np.exp(-tau / lam)) / (tau / lam) - np.exp(-tau / lam))
    return beta0 + term1 + term2


# Lambda is fixed (not fitted) using the Diebold-Li (2006) convention.
# With only 5 maturities, letting lambda float freely is poorly identified
# (it tends to run to an arbitrary extreme, or sit at whatever bound is set).
# Fixing it keeps beta0/beta1/beta2 comparable across every day in the sample.
LAMBDA_FIXED = 0.7308  # places max curvature loading around a ~2.5yr maturity


def ns_fixed_lambda(tau, beta0, beta1, beta2):
    return nelson_siegel(tau, beta0, beta1, beta2, LAMBDA_FIXED)


# ---------------------------------------------------------------------------
# 3. FIT THE MODEL FOR EVERY TRADING DAY
# ---------------------------------------------------------------------------

results = []
for date, row in df_recent.iterrows():
    yields_row = row.values
    if np.any(pd.isna(yields_row)):
        continue
    try:
        params, _ = curve_fit(
            ns_fixed_lambda, MATURITIES, yields_row, p0=[4, -1, 1]
        )
        results.append(
            {"date": date, "level": params[0], "slope": params[1], "curvature": params[2]}
        )
    except RuntimeError:
        # Fit failed to converge for this particular day; skip it.
        continue

ns_timeseries = pd.DataFrame(results).set_index("date")


# ---------------------------------------------------------------------------
# 4. DETECT INVERSION EPISODES (monthly, to filter out day-to-day noise)
# ---------------------------------------------------------------------------

slope_monthly = ns_timeseries["slope"].resample("ME").last()
is_positive = slope_monthly > 0

episodes_list = []
current_start = slope_monthly.index[0]
current_sign = is_positive.iloc[0]
prev_date = current_start

for date, sign in is_positive.items():
    if sign != current_sign:
        duration = (prev_date.to_period("M") - current_start.to_period("M")).n + 1
        episodes_list.append(
            {"is_positive": current_sign, "start": current_start, "end": prev_date, "duration_months": duration}
        )
        current_start = date
        current_sign = sign
    prev_date = date

# Close out the final episode
duration = (prev_date.to_period("M") - current_start.to_period("M")).n + 1
episodes_list.append(
    {"is_positive": current_sign, "start": current_start, "end": prev_date, "duration_months": duration}
)

episodes_m = pd.DataFrame(episodes_list)
inversion_episodes_m = episodes_m[episodes_m["is_positive"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. MEASURE LEAD TIME: INVERSION START -> NEXT RECESSION START
# ---------------------------------------------------------------------------

LOOKBACK_MONTHS = 30  # how far back to search for the relevant inversion episode

print("Inversion episodes (monthly):")
print(inversion_episodes_m[["start", "end", "duration_months"]])
print()

lead_time_results = []
for rec_start in recession_starts:
    prior_episodes = inversion_episodes_m[inversion_episodes_m["end"] < rec_start]
    if len(prior_episodes) == 0:
        print("Recession", rec_start.date(), ": no prior inversion found (before data start or genuinely none)")
        continue

    lookback = prior_episodes[prior_episodes["start"] >= rec_start - pd.DateOffset(months=LOOKBACK_MONTHS)]
    if len(lookback) == 0:
        print("Recession", rec_start.date(), ": no inversion within", LOOKBACK_MONTHS, "months prior")
        continue

    main_episode = lookback.loc[lookback["duration_months"].idxmax()]
    months_lead = (rec_start - main_episode["start"]).days / 30.44
    lead_time_results.append(
        {"recession": rec_start, "inversion_start": main_episode["start"], "lead_months": round(months_lead, 1)}
    )
    print(
        "Recession", rec_start.date(),
        ": main inversion started", main_episode["start"].date(),
        "(lasted", main_episode["duration_months"], "months), lead time", round(months_lead, 1), "months",
    )

lead_time_df = pd.DataFrame(lead_time_results)


# ---------------------------------------------------------------------------
# 6. FINAL CHART
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(ns_timeseries.index, ns_timeseries["slope"], color="darkblue", linewidth=1, label="Slope (beta1)")
ax.axhline(0, color="gray", linestyle="--", alpha=0.6, linewidth=1)

# Grey bands: official NBER recessions
first_label_done = False
for rec_start in recession_starts:
    future = recession_df_monthly[recession_df_monthly.index > rec_start]
    end_candidates = future[future["recession"] == 0]
    rec_end = end_candidates.index[0] if len(end_candidates) > 0 else ns_timeseries.index[-1]
    label = "NBER recession" if not first_label_done else ""
    ax.axvspan(rec_start, rec_end, color="gray", alpha=0.3, label=label)
    first_label_done = True

# Red bands: our detected inversion episodes (3+ months, to filter noise)
for _, ep in inversion_episodes_m.iterrows():
    if ep["duration_months"] >= 3:
        ax.axvspan(ep["start"], ep["end"], color="red", alpha=0.15)

ax.set_xlabel("Date")
ax.set_ylabel("Slope (beta1)")
ax.set_title("Yield Curve Inversion vs. NBER Recessions (" + START_DATE[:4] + "-2026)")
ax.set_xlim(ns_timeseries.index.min(), ns_timeseries.index.max())
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("yield_curve_inversions.png", dpi=150)
plt.show()

