icu1 AS (
  SELECT s.subject_id, s.hadm_id, s.stay_id, s.intime, s.outtime, p.gender, p.anchor_year_group,
         p.anchor_age + (year(s.intime) - p.anchor_year) AS age
  FROM icu.icustays s JOIN hosp.patients p USING (subject_id)
  QUALIFY row_number() OVER (PARTITION BY s.subject_id ORDER BY s.intime, s.stay_id) = 1
),
asadm AS (
  SELECT e.hadm_id, e.charttime AS t,
         CASE WHEN regexp_matches(lower(e.medication), 'famotidine|pepcid') THEN 'FAM'
              WHEN regexp_matches(lower(e.medication), 'ranitidine|zantac|cimetidine|nizatidine') THEN 'H2O' ELSE 'PPI' END AS drug
  FROM hosp.emar e
  WHERE regexp_matches(lower(e.medication), 'famotidine|pepcid|pantoprazole|protonix|omeprazole|prilosec|esomeprazole|nexium|lansoprazole|prevacid|ranitidine|zantac|cimetidine|nizatidine|rabeprazole|dexlansoprazole')
    AND regexp_matches(lower(coalesce(e.event_txt, '')), 'administ|started')
    AND NOT regexp_matches(lower(coalesce(e.event_txt, '')), 'not |hold|stopp')
  UNION ALL
  SELECT s.hadm_id, x.starttime,
         CASE WHEN x.itemid = 225907 THEN 'FAM' WHEN x.itemid IN (225911, 222190) THEN 'H2O' ELSE 'PPI' END
  FROM icu.inputevents x JOIN icu.icustays s USING (stay_id)
  WHERE x.itemid IN (225907, 225910, 227694, 229615, 225911, 222190)
    AND coalesce(x.statusdescription, '') <> 'Rewritten' AND coalesce(x.amount, 0) > 0
),
first_t AS (
  SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, i.gender, i.anchor_year_group, i.age, min(a.t) AS t0
  FROM icu1 i JOIN asadm a USING (hadm_id)
  WHERE a.t BETWEEN i.intime AND i.outtime AND i.age >= 18
  GROUP BY ALL
),
first_as AS (

  SELECT f.*, CASE WHEN count(DISTINCT a.drug) = 1 THEN min(a.drug) END AS drug
  FROM first_t f JOIN asadm a ON a.hadm_id = f.hadm_id AND a.t = f.t0
  GROUP BY ALL
),
home AS (
  SELECT DISTINCT e.hadm_id FROM ed.medrecon m JOIN ed.edstays e USING (stay_id)
  WHERE e.hadm_id IS NOT NULL
    AND regexp_matches(lower(m.name), 'famotidine|pepcid|pantoprazole|protonix|omeprazole|prilosec|esomeprazole|nexium|lansoprazole|prevacid|ranitidine|zantac|cimetidine|nizatidine|rabeprazole|dexlansoprazole')
  UNION
  SELECT DISTINCT hadm_id FROM note.discharge
  WHERE hadm_id IS NOT NULL
    AND regexp_matches(regexp_extract(lower(text), '(?s)medications on admission:(.*?)(discharge medications:|discharge disposition:)', 1),
                       'famotidine|pepcid|pantoprazole|protonix|omeprazole|prilosec|esomeprazole|nexium|lansoprazole|prevacid|ranitidine|zantac|cimetidine|nizatidine|rabeprazole|dexlansoprazole')
),
cr AS (SELECT subject_id, charttime, valuenum FROM hosp.labevents WHERE itemid = 50912 AND valuenum > 0),
cam AS (SELECT stay_id, charttime FROM icu.chartevents WHERE itemid = 228332 AND value = 'Positive'),
cand AS (
  SELECT f.*,
         (SELECT c.valuenum FROM cr c WHERE c.subject_id = f.subject_id AND c.charttime BETWEEN f.t0 - INTERVAL 24 HOUR AND f.t0
          ORDER BY c.charttime DESC LIMIT 1) AS scr
  FROM first_as f
  WHERE f.drug = 'FAM' AND f.anchor_year_group <> '2008 - 2010'
    AND NOT EXISTS (SELECT 1 FROM asadm a WHERE a.hadm_id = f.hadm_id AND a.t < f.t0 AND a.t >= f.t0 - INTERVAL 24 HOUR)
    AND f.hadm_id NOT IN (SELECT hadm_id FROM home)
    AND NOT EXISTS (SELECT 1 FROM cam c WHERE c.stay_id = f.stay_id AND c.charttime <= f.t0)
),
cohort AS (
  SELECT *, 142 * power(least(scr / CASE WHEN gender = 'F' THEN 0.7 ELSE 0.9 END, 1), CASE WHEN gender = 'F' THEN -0.241 ELSE -0.302 END)
            * power(greatest(scr / CASE WHEN gender = 'F' THEN 0.7 ELSE 0.9 END, 1), -1.2) * power(0.9938, age)
            * CASE WHEN gender = 'F' THEN 1.012 ELSE 1 END AS egfr
  FROM cand WHERE scr IS NOT NULL
),
elig AS (SELECT subject_id, anchor_year_group AS era FROM cohort WHERE egfr < 60),
famrx AS (
  SELECT r.hadm_id, r.starttime, r.stoptime, TRY_CAST(regexp_extract(r.dose_val_rx, '^[0-9.]+') AS DOUBLE) * r.doses_per_24_hrs AS daily
  FROM hosp.prescriptions r WHERE regexp_matches(lower(r.drug), 'famotidine|pepcid')
),
act AS (
  SELECT c.subject_id, f.daily, rank() OVER (PARTITION BY c.subject_id ORDER BY f.starttime DESC) AS rk
  FROM cohort c JOIN famrx f ON f.hadm_id = c.hadm_id AND f.starttime <= c.t0 + INTERVAL 1 HOUR AND coalesce(f.stoptime, c.t0 + INTERVAL 2 HOUR) > c.t0
  WHERE c.subject_id IN (SELECT subject_id FROM elig)
),
ord AS (

  SELECT e.subject_id,
         (SELECT CASE WHEN count(*) > 0 AND count(*) = count(a.daily) AND min(a.daily) = max(a.daily) THEN min(a.daily) END
          FROM act a WHERE a.subject_id = e.subject_id AND a.rk = 1) AS daily
  FROM elig e
),
expo AS (
  SELECT c.subject_id, c.t0,
         CASE WHEN o.daily <= 20 THEN 'A' WHEN o.daily >= 40 THEN 'B' END AS arm
  FROM cohort c JOIN ord o USING (subject_id)
),
fu AS (
  SELECT x.subject_id, x.t0, c.stay_id, c.hadm_id, least(c.outtime, x.t0 + INTERVAL 168 HOUR) AS t_end
  FROM expo x JOIN cohort c USING (subject_id)
),
glu AS (
  SELECT subject_id, charttime FROM hosp.labevents WHERE itemid IN (50931, 50809) AND valuenum > 0 AND valuenum < 70
  UNION ALL
  SELECT subject_id, charttime FROM icu.chartevents WHERE itemid IN (225664, 220621, 226537) AND valuenum > 0 AND valuenum < 70
),
outc AS (
  SELECT f.subject_id, 'delirium_7d' AS outcome, date_diff('minute', f.t0, min(c.charttime)) / 60.0 AS event_h
  FROM fu f JOIN cam c ON c.stay_id = f.stay_id AND c.charttime > f.t0 AND c.charttime <= f.t_end GROUP BY f.subject_id, f.t0
  UNION ALL
  SELECT f.subject_id, 'delirium_72h', date_diff('minute', f.t0, min(c.charttime)) / 60.0
  FROM fu f JOIN cam c ON c.stay_id = f.stay_id AND c.charttime > f.t0 AND c.charttime <= least(f.t_end, f.t0 + INTERVAL 72 HOUR) GROUP BY f.subject_id, f.t0
  UNION ALL
  SELECT f.subject_id, 'negctl_hypoglycemia_7d', date_diff('minute', f.t0, min(g.charttime)) / 60.0
  FROM fu f JOIN glu g ON g.subject_id = f.subject_id AND g.charttime > f.t0 AND g.charttime <= f.t_end GROUP BY f.subject_id, f.t0
  UNION ALL
  SELECT f.subject_id, 'death_in_hospital', greatest(0, date_diff('minute', f.t0, a.deathtime)) / 60.0
  FROM fu f JOIN hosp.admissions a USING (hadm_id) WHERE a.deathtime IS NOT NULL
),
rass AS (SELECT stay_id, charttime FROM icu.chartevents WHERE itemid = 228096 AND NULLIF(value, '') IS NOT NULL),
conf AS (
  SELECT x.subject_id, 'sofa_t0' AS item,
         EXISTS (SELECT 1 FROM derived.sofa s WHERE s.stay_id = c.stay_id AND x.t0 >= s.starttime AND x.t0 < s.endtime) AS available
  FROM expo x JOIN cohort c USING (subject_id)
  UNION ALL SELECT x.subject_id, 'creat_t0', c.scr IS NOT NULL FROM expo x JOIN cohort c USING (subject_id)
  UNION ALL
  SELECT x.subject_id, 'rass_t0', EXISTS (SELECT 1 FROM rass r WHERE r.stay_id = c.stay_id AND r.charttime BETWEEN x.t0 - INTERVAL 12 HOUR AND x.t0)
  FROM expo x JOIN cohort c USING (subject_id)
  UNION ALL SELECT x.subject_id, 't0_exact', x.t0 IS NOT NULL FROM expo x
)
