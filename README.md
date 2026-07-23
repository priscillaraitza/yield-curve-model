# Yield Curve Model - Does Inversion Predict Recessions?

Independent research project fitting the Nelson-Siegel model to US Treasury 
yields (1990–2026) to test whether yield curve inversions have historically 
led NBER-dated recessions, and by how much.

# Method
- Data: FRED (3mo, 2y, 5y, 10y, 30y Treasury yields; NBER recession dates)
- Nelson-Siegel decomposition into Level / Slope / Curvature
- Lambda fixed using the Diebold-Li (2006) convention for cross-time comparability
- Inversion "episodes" detected on monthly-aggregated data to filter day-to-day noise

# Key finding
Inversions preceded the 2001 and 2007–09 recessions by 8 and 17 months 
respectively. The 2019 inversion preceded the 2020 recession by 9 months, 
though that recession was pandemic-driven rather than monetary-policy-driven. 
The 2022–2024 inversion (25 months, the longest on record) has not been 
followed by a recession (an ongoing outlier relative to the historical pattern.)

![Chart](yield_curve_chart.png)

[Full write-up with methodology and limitations] (in progress)
