pt AS (
  SELECT p.patientunitstayid, p.uniquepid, p.hospitalid, p.gender, p.hospitaldischargeyear, p.unitdischargeoffset, p.apacheadmissiondx,
         p.hospitaldischargeoffset, p.hospitaldischargestatus, TRY_CAST(IF(p.age = '> 89', '90', p.age) AS BIGINT) AS age
  FROM eicu.patient p
  WHERE TRY_CAST(IF(p.age = '> 89', '90', p.age) AS BIGINT) >= 18
  QUALIFY ROW_NUMBER() OVER (PARTITION BY p.uniquepid ORDER BY p.hospitaldischargeyear, p.patienthealthsystemstayid, p.unitvisitnumber, p.patientunitstayid) = 1
),
dsc AS (
  SELECT patientunitstayid, nursingchartoffset AS t,
         (upper(nursingchartvalue) = 'YES' OR TRY_CAST(nursingchartvalue AS INTEGER) >= 4) AS pos
  FROM eicu.nursecharting
  WHERE nursingchartcelltypevallabel = 'Delirium Scale/Score' AND nursingchartcelltypevalname = 'Delirium Score'
    AND NULLIF(nursingchartvalue, '') IS NOT NULL
),
dl_hosp AS (
  SELECT hospitalid FROM pt GROUP BY 1
  HAVING count(*) >= 50 AND count(*) FILTER (WHERE patientunitstayid IN (SELECT patientunitstayid FROM dsc)) >= 0.5 * count(*)
),
med0 AS (
  SELECT m.patientunitstayid, m.drugstartoffset, m.drugstopoffset, lower(coalesce(m.dosage, '')) AS dosage, lower(coalesce(m.frequency, '')) AS freq,
         CASE WHEN m.drughiclseqno IN (4521, 35085, 35107, 4851) OR regexp_matches(lower(coalesce(m.drugname, '')), 'famotidine|pepcid') THEN 'FAM'
              WHEN regexp_matches(lower(coalesce(m.drugname, '')), 'ranitidine|zantac|cimetidine') THEN 'H2O' ELSE 'PPI' END AS drug
  FROM eicu.medication m
  WHERE coalesce(m.drugordercancelled, 'No') <> 'Yes'
    AND (m.drughiclseqno IN (4521, 35085, 35107, 4851, 22008, 4673, 25968)
         OR regexp_matches(lower(coalesce(m.drugname, '')), 'famotidine|pepcid|pantoprazole|protonix|omeprazole|prilosec|esomeprazole|nexium|lansoprazole|prevacid|ranitidine|zantac|cimetidine|rabeprazole'))
),
med AS (
  SELECT *, CASE
           WHEN TRY_CAST(regexp_extract(dosage, '([0-9.]+)', 1) AS DOUBLE) IS NULL THEN NULL
           WHEN regexp_matches(freq, 'q12|bid|twice|every 12|2 times') THEN TRY_CAST(regexp_extract(dosage, '([0-9.]+)', 1) AS DOUBLE) * 2
           WHEN regexp_matches(freq, 'q24|daily|qd|once|every 24|qhs|bedtime|nightly|1 time|1xdaily') THEN TRY_CAST(regexp_extract(dosage, '([0-9.]+)', 1) AS DOUBLE) END AS daily
  FROM med0
),
first_off AS (SELECT patientunitstayid, min(drugstartoffset) AS t0 FROM med WHERE drugstartoffset >= 0 GROUP BY 1),
first_as AS (

  SELECT f.patientunitstayid, f.t0 AS drugstartoffset,
         CASE WHEN count(DISTINCT m.drug) = 1 THEN min(m.drug) END AS drug,
         CASE WHEN count(*) = count(m.daily) AND min(m.daily) = max(m.daily) THEN min(m.daily) END AS daily
  FROM first_off f JOIN med m ON m.patientunitstayid = f.patientunitstayid AND m.drugstartoffset = f.t0
  GROUP BY 1, 2
),
home AS (
  SELECT DISTINCT patientunitstayid FROM eicu.admissiondrug
  WHERE drughiclseqno IN (4521, 35085, 35107, 4851, 22008, 4673, 25968)
     OR regexp_matches(lower(coalesce(drugname, '')), 'famotidine|pepcid|pantoprazole|protonix|omeprazole|prilosec|esomeprazole|nexium|lansoprazole|prevacid|ranitidine|zantac|cimetidine|rabeprazole')
),
gib AS (
  SELECT patientunitstayid, diagnosisoffset AS t FROM eicu.diagnosis
  WHERE regexp_matches(lower(diagnosisstring), 'gi bleeding|gastrointestinal bleed|upper gi bleed|hematemesis|melena|peptic ulcer|variceal|ulcer disease')
),
ppi_inf AS (
  SELECT patientunitstayid, infusionoffset AS t FROM eicu.infusiondrug
  WHERE regexp_matches(lower(coalesce(drugname, '')), 'pantoprazole|protonix|esomeprazole|nexium|omeprazole')
),
cr AS (SELECT patientunitstayid, labresultoffset, labresult FROM eicu.lab WHERE labname = 'creatinine' AND labresult > 0),
cand AS (
  SELECT p.*, f.drugstartoffset AS t0, f.drug, f.daily,
         (SELECT c.labresult FROM cr c WHERE c.patientunitstayid = p.patientunitstayid AND c.labresultoffset BETWEEN f.drugstartoffset - 1440 AND f.drugstartoffset
          ORDER BY c.labresultoffset DESC LIMIT 1) AS scr
  FROM pt p JOIN first_as f USING (patientunitstayid)
  WHERE p.hospitalid IN (SELECT hospitalid FROM dl_hosp) AND f.drug IN ('FAM', 'PPI') AND f.drugstartoffset < p.unitdischargeoffset
    AND NOT regexp_matches(lower(coalesce(p.apacheadmissiondx, '')), 'bleeding, upper gi|bleeding, gi|gi bleeding|bleeding, lower gi|ulcer disease|varices|peptic')
    AND NOT EXISTS (SELECT 1 FROM gib g WHERE g.patientunitstayid = p.patientunitstayid AND g.t <= f.drugstartoffset)
    AND NOT EXISTS (SELECT 1 FROM ppi_inf i WHERE i.patientunitstayid = p.patientunitstayid AND i.t BETWEEN f.drugstartoffset - 1440 AND f.drugstartoffset + 60)
    AND NOT EXISTS (SELECT 1 FROM med a WHERE a.patientunitstayid = p.patientunitstayid AND a.drugstartoffset < f.drugstartoffset
                    AND a.drugstartoffset > f.drugstartoffset - 10080
                    AND (a.drugstopoffset > f.drugstartoffset - 1440 OR a.drugstopoffset <= a.drugstartoffset))
    AND p.patientunitstayid NOT IN (SELECT patientunitstayid FROM home)
    AND NOT EXISTS (SELECT 1 FROM dsc d WHERE d.patientunitstayid = p.patientunitstayid AND d.pos AND d.t <= f.drugstartoffset)
),
cohort AS (
  SELECT *, 142 * power(least(scr / CASE WHEN gender = 'Female' THEN 0.7 ELSE 0.9 END, 1), CASE WHEN gender = 'Female' THEN -0.241 ELSE -0.302 END)
            * power(greatest(scr / CASE WHEN gender = 'Female' THEN 0.7 ELSE 0.9 END, 1), -1.2) * power(0.9938, age)
            * CASE WHEN gender = 'Female' THEN 1.012 ELSE 1 END AS egfr
  FROM cand
),
elig AS (SELECT patientunitstayid AS subject_id, hospitaldischargeyear AS era FROM cohort),
site AS (SELECT patientunitstayid AS subject_id, hospitalid AS site FROM cohort),
expo AS (
  SELECT c.patientunitstayid AS subject_id, c.t0, CASE WHEN c.drug = 'PPI' THEN 'A' WHEN c.drug = 'FAM' THEN 'B' END AS arm FROM cohort c
),
fu AS (
  SELECT x.subject_id, x.t0, least(c.unitdischargeoffset, x.t0 + 10080) AS t_end, c.hospitaldischargeoffset, c.hospitaldischargestatus
  FROM expo x JOIN cohort c ON c.patientunitstayid = x.subject_id
),
glu AS (SELECT patientunitstayid, labresultoffset AS t FROM eicu.lab WHERE labname IN ('glucose', 'bedside glucose') AND labresult > 0 AND labresult < 70),
outc AS (
  SELECT f.subject_id, 'delirium_7d' AS outcome, (min(d.t) - f.t0) / 60.0 AS event_h
  FROM fu f JOIN dsc d ON d.patientunitstayid = f.subject_id AND d.pos AND d.t > f.t0 AND d.t <= f.t_end GROUP BY f.subject_id, f.t0
  UNION ALL
  SELECT f.subject_id, 'delirium_72h', (min(d.t) - f.t0) / 60.0
  FROM fu f JOIN dsc d ON d.patientunitstayid = f.subject_id AND d.pos AND d.t > f.t0 AND d.t <= least(f.t_end, f.t0 + 4320) GROUP BY f.subject_id, f.t0
  UNION ALL
  SELECT f.subject_id, 'negctl_hypoglycemia_7d', (min(g.t) - f.t0) / 60.0
  FROM fu f JOIN glu g ON g.patientunitstayid = f.subject_id AND g.t > f.t0 AND g.t <= f.t_end GROUP BY f.subject_id, f.t0
  UNION ALL
  SELECT f.subject_id, 'death_in_hospital', greatest(0, f.hospitaldischargeoffset - f.t0) / 60.0 FROM fu f WHERE f.hospitaldischargestatus = 'Expired'
),
sed AS (
  SELECT DISTINCT n.patientunitstayid, n.nursingchartoffset AS t FROM eicu.nursecharting n
  WHERE (n.nursingchartcelltypevallabel = 'RASS' OR (n.nursingchartcelltypevallabel = 'Sedation Scale/Score/Goal' AND n.nursingchartcelltypevalname = 'Sedation Score'))
    AND NULLIF(n.nursingchartvalue, '') IS NOT NULL
),
apache AS (SELECT DISTINCT patientunitstayid FROM eicu.apachepatientresult WHERE apachescore > 0),
conf AS (
  SELECT x.subject_id, 'apache_firstday' AS item, x.subject_id IN (SELECT patientunitstayid FROM apache) AS available FROM expo x
  UNION ALL SELECT x.subject_id, 'creat_t0', c.scr IS NOT NULL FROM expo x JOIN cohort c ON c.patientunitstayid = x.subject_id
  UNION ALL SELECT x.subject_id, 'rass_t0', EXISTS (SELECT 1 FROM sed s WHERE s.patientunitstayid = x.subject_id AND s.t BETWEEN x.t0 - 720 AND x.t0) FROM expo x
  UNION ALL SELECT x.subject_id, 't0_exact', false FROM expo x
)
