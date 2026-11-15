# PPI vs famotidine and detected ICU delirium

Extraction and analysis code for the article "Detected delirium after starting a proton pump inhibitor versus famotidine
in critically ill adults: a multi-database target trial emulation". The study compared adults in the intensive care unit
(ICU) who started a proton pump inhibitor (PPI) with those who started famotidine. The primary outcome was the first
positive delirium screen within 7 days of the first dose in MIMIC-IV and eICU-CRD, with ICU discharge and death as
competing events. Two further analyses compared a daily famotidine dose of 20 mg or less with 40 mg or more in famotidine
starters with an eGFR below 60 mL/min/1.73 m², and examined the first antipsychotic within 7 days as an exploratory
outcome in five databases (MIMIC-IV, eICU-CRD, NWICU, INSPIRE and MOVER).

The analyses used MIMIC-IV 3.1 (with MIMIC-IV-ED 2.2 and MIMIC-IV-Note 2.2), eICU-CRD 2.0, NWICU 0.1.0 and INSPIRE
1.4.2, which are available to credentialed researchers through PhysioNet, and MOVER, which is available through the
University of California Irvine. The results are reported in the article and its Additional file 1.

## Contents

| Folder | What it holds |
|---|---|
| `code/` | extraction and analysis scripts |
| `code/sql/` | SQL that defines eligibility, time zero and the treatment arms: one file for each comparison and database (`ppi_*`, PPI versus famotidine; `famdose_*`, famotidine dose) and one for each variant definition of R7 and R8 (`r7_*`, `r8_*`) |
| `code/tools/helpers/` | helper modules: Parquet tables as DuckDB views (`mimic_local.py`), Bayesian reanalysis (`bayes_reanalysis.py`), reading the MOVER archive as a stream (`extract_mover.py`) |
| `environment/` | package versions |

## Scripts

`asd.py` holds the main analyses. The other analyses are labelled E1, E2 and R7 to R18 in the order in which they were
added, and their scripts are named after these labels (`explore_ppi.py` is E1).

| Script | What it does |
|---|---|
| `asd.py` | Main analyses: cohorts, covariates, follow-up and outcomes (primary and negative control) in MIMIC-IV and eICU-CRD; stage 1 design diagnostics; PPI vs famotidine and famotidine dose, with bootstrap, pooling and the Bayesian reanalysis |
| `stats_core.py` | Statistical methods: ridge logistic regression, overlap and inverse probability weights, censoring models, weighted Aalen-Johansen cumulative incidence, pooling, two-stage least squares, masking of counts |
| `explore_ppi.py` | E1, why MIMIC-IV and eICU-CRD differ: delirium assessment; strata of sedation, ventilation and screening; standardisation to the sedation mix of the other database |
| `e2_burden.py` | E2: days alive and free of delirium or coma, and days with delirium, within 7 days; E-values |
| `e2_mover.py` | E2: first antipsychotic in MOVER |
| `r7.py` | Flow of patients; added adjustment (diabetes, insulin, current-admission codes); first antipsychotic in MIMIC-IV and eICU-CRD; three variants of the definitions (MIMIC-IV bleeding exclusion without the principal discharge diagnosis; MIMIC-IV time zero and arm from orders; eICU-CRD dose classification) |
| `r8.py` | MIMIC-IV defined as in eICU-CRD (no exclusion for home medication; time zero and arm from orders) |
| `r9.py` | First antipsychotic in NWICU and INSPIRE |
| `r10.py` | Flexible models: gradient-boosted propensity score and doubly robust estimation |
| `r11.py` | Screening frequency and the interval-censored onset model; covariate groups; post-surgical patients; calendar periods; further adjustment (corticosteroids, types of sedative); flexible models in NWICU |
| `r12.py` | eICU-CRD without hospital indicators, with hospital-level and patient-level bootstrap; hospital-level description |
| `r14.py`, `r14_checks.py` | eICU-CRD model with information recorded before time zero and ventilation strata from time-stamped records; record and timing checks |
| `r15.py` | Negative control within the sedation strata |
| `r16.py`, `r16_fmt.py` | Further analyses (Tables S13-S22, Figures S7-S10) and their values to two decimals |
| `r16_a16.py`, `r16_a17.py` | First antipsychotic without the exclusion for an earlier positive screen; orders and administrations in MIMIC-IV (Table S23) |
| `r16b.py` | MIMIC-IV model with reasons for admission, added after the results of `r16.py` were seen |
| `r16_dxcheck.py` | Check of the eICU-CRD diagnosis groups behind Table S20 (both arms together, no outcomes) |
| `r17_mult.py` | Holm and Bonferroni correction of the 25 stratum estimates (Table S24) |
| `r18.py`, `r18b.py`, `r18w.py` | eICU-CRD intervals and tests from patient-level resampling (square brackets in Additional file 1) |
| `fig_values.py`, `fig_values_r9.py`, `fig_curves.py` | Unrounded values and curves for the figures (no new analysis) |

## Where the numbers come from

| Table or figure | Scripts |
|---|---|
| Tables 1 and S12: target trial and its emulation | `sql/*.sql`, `asd.py`, `r9.py`, `e2_mover.py` |
| Table 2: characteristics, MIMIC-IV and eICU-CRD | `asd.py` |
| Figures 1 and S5: flow of patients | `r7.py` |
| Figure 2: cumulative incidence | `fig_curves.py`, `asd.py`, `explore_ppi.py`, `r15.py` |
| Figure 3, Tables S3 and S9: delirium, famotidine dose | `asd.py`, `r7.py`, `fig_values.py`, `r10.py`, `r11.py`, `e2_burden.py` |
| Figure 4, Table S24: sedation and ventilation strata, Holm P values | `explore_ppi.py`, `r17_mult.py` |
| Figure 5, Table S6: first antipsychotic | `r7.py`, `fig_values.py`, `r9.py`, `fig_values_r9.py`, `e2_mover.py` |
| Table S1, Figures S2 and S3: NWICU, INSPIRE and MOVER | `r9.py`, `e2_mover.py` |
| Table S2: delirium assessment and sedation | `explore_ppi.py`, `r11.py` |
| Table S4: the difference between the databases | `r15.py`, `r11.py`, `explore_ppi.py`, `r7.py`, `r8.py` |
| Tables S5, S7 and S8 | `r11.py`, `r9.py` |
| Table S10: eICU-CRD hospital indicators | `r12.py` |
| Table S11: eICU-CRD records and timing | `r14_checks.py`, `r14.py`, `r7.py`, `r9.py`, `e2_mover.py`, `r11.py` |
| Tables S13-S22, Figures S7-S10 | `r16.py`, `r16_fmt.py`, `r16b.py`, `r16_dxcheck.py` |
| Table S23 | `r16_a16.py`, `r16_a17.py` |
| Figure S4 | `r7.py`, `fig_values.py`, `r10.py` |
| Figure S6 | `e2_burden.py` |
| eICU-CRD intervals and tests from patient-level resampling (square brackets in Additional file 1) | `r18.py`, `r18b.py`, `r18w.py` |

## Licence

The code is released under the MIT License (see `LICENSE`).
