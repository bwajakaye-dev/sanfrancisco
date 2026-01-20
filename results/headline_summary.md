# Headline findings (auto-generated)

## Clustering: top EV-adoption cluster
- Most recent year analyzed: **2024**
- Top cluster (highest mean EV%): **cluster 0**

Cluster summary (K-Means):

| cluster | n_zip | mean_ev_pct | mean_growth | mean_total |
| --- | --- | --- | --- | --- |
| 0.000 | 6.000 | 14.21 | 2.333 | 12136.83 |
| 3.000 | 14.00 | 7.948 | 1.683 | 11058.14 |
| 1.000 | 6.000 | 6.695 | 1.452 | 31079.50 |
| 2.000 | 2.000 | 4.250 | -0.810 | 966.50 |

ZIP codes in the top EV cluster:

| zip_code | area | ev_% | ev_growth_pp | total_vehicles |
| --- | --- | --- | --- | --- |
| 94107 | Potrero Hill / Mission Bay / SoMa | 18.03 | 0.520 | 17,064 |
| 94105 | Financial District / South Beach / Mission Bay | 16.91 | 4.340 | 5,943 |
| 94158 | Mission Bay / Potrero Hill (Dogpatch area) | 15.16 | 2.560 | 3,575 |
| 94114 | Castro / Noe Valley / Twin Peaks | 12.23 | 2.260 | 16,523 |
| 94131 | Noe Valley / Glen Park / Bernal Heights | 11.55 | 1.980 | 16,324 |
| 94127 | West Portal / Twin Peaks | 11.36 | 2.340 | 13,392 |

## Direct rankings (most recent year)
Top 5 ZIP codes by EV percentage:

| zip_code | area | ev_% | ev_growth_pp | total_vehicles |
| --- | --- | --- | --- | --- |
| 94107 | Potrero Hill / Mission Bay / SoMa | 18.03 | 0.520 | 17,064 |
| 94105 | Financial District / South Beach / Mission Bay | 16.91 | 4.340 | 5,943 |
| 94158 | Mission Bay / Potrero Hill (Dogpatch area) | 15.16 | 2.560 | 3,575 |
| 94114 | Castro / Noe Valley / Twin Peaks | 12.23 | 2.260 | 16,523 |
| 94131 | Noe Valley / Glen Park / Bernal Heights | 11.55 | 1.980 | 16,324 |

Top 5 ZIP codes by EV growth (percentage-point change):

| zip_code | area | ev_growth_pp | ev_% | total_vehicles |
| --- | --- | --- | --- | --- |
| 94105 | Financial District / South Beach / Mission Bay | 4.340 | 16.91 | 5,943 |
| 94123 | Marina / Cow Hollow | 2.580 | 9.440 | 13,305 |
| 94158 | Mission Bay / Potrero Hill (Dogpatch area) | 2.560 | 15.16 | 3,575 |
| 94127 | West Portal / Twin Peaks | 2.340 | 11.36 | 13,392 |
| 94114 | Castro / Noe Valley / Twin Peaks | 2.260 | 12.23 | 16,523 |

Bottom 5 ZIP codes by EV percentage:

| zip_code | area | ev_% | ev_growth_pp | total_vehicles |
| --- | --- | --- | --- | --- |
| 94128 | SFO (San Francisco International Airport) | 0.880 | 0.400 | 569.00 |
| 94124 | Bayview–Hunters Point | 5.140 | 1.020 | 31,346 |
| 94134 | Visitacion Valley / Excelsior / Portola | 5.550 | 1.420 | 24,624 |
| 94112 | Excelsior / Ingleside–Oceanview | 5.810 | 1.350 | 44,448 |
| 94130 | Treasure Island / Yerba Buena Island | 5.940 | 1.460 | 893.00 |

## Association rules: top 5 by lift
- Interpretation: lift > 1 suggests the consequent is more likely when the antecedent holds.
- False discovery control: rules include a permutation-test p-value and BH-FDR q-value (when computed).

| antecedent | consequent | support | confidence | lift | q_value_bh |
| --- | --- | --- | --- | --- | --- |
| ev_share_mid | make_TOYOTA_high | 0.214 | 0.667 | 1.556 | 0.020 |
| ev_share_mid, make_OTHER/UNK_high | make_TOYOTA_high | 0.214 | 0.667 | 1.556 | 0.020 |
| ev_share_mid | make_OTHER/UNK_high, make_TOYOTA_high | 0.214 | 0.667 | 1.556 | 0.020 |
| duty_light_majority, ev_share_mid | make_TOYOTA_high | 0.214 | 0.667 | 1.556 | 0.020 |
| ev_share_mid | duty_light_majority, make_TOYOTA_high | 0.214 | 0.667 | 1.556 | 0.020 |

## Anomaly detection: ZIP codes flagged
- Flagged anomalies: **5**

| zip_code | area | anomaly_score | ev_percentage | ev_growth | make_hhi | top_make_share | total_vehicles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 94105 | Financial District / South Beach / Mission Bay | 0.599 | 16.91 | 4.340 | 0.591 | 0.759 | 5,943 |
| 94128 | SFO (San Francisco International Airport) | 0.593 | 0.880 | 0.400 | 0.848 | 0.917 | 569.00 |
| 94112 | Excelsior / Ingleside–Oceanview | 0.574 | 5.810 | 1.350 | 0.198 | 0.292 | 44,448 |
| 94104 | Financial District / Chinatown | 0.571 | 7.620 | -2.020 | 0.371 | 0.578 | 1,364 |
| 94130 | Treasure Island / Yerba Buena Island | 0.548 | 5.940 | 1.460 | 1.000 | 1.000 | 893.00 |

## Predictive modeling: performance snapshot
- Best regression model (by RMSE): **random_forest**, RMSE=1.882, MAE=1.228, R²=0.717
- Best classifier (by F1): **logistic_regression**, F1=0.933, accuracy=0.893, AUC=0.980, threshold≈6.65%

## Caveat about ZIP → neighborhood names
ZIP codes overlap multiple neighborhoods. The area labels are short human-readable summaries based on DataSF's ZIP↔Analysis Neighborhood crosswalk; treat them as approximate.
