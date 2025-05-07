WITH admit_reason AS (
    SELECT patientunitstayid,
        admitdxname,
        admitdxtext
    FROM `physionet-data.eicu_crd.admissiondx`
),
base_table AS (
    SELECT patientunitstayid,
        hospitalid,
        unittype,
        unitadmitsource,
        gender,
        age,
        ethnicity,
        apacheadmissiondx,
        unitdischargeoffset,
        unitdischargestatus
    FROM `physionet-data.eicu_crd.patient`
    WHERE apacheadmissiondx LIKE '%Sepsis%'
        AND unitdischargeoffset > 1440 -- At least 24 hours
        -- AND gender LIKE '%ale%' -- Ignore empty
        -- AND ethnicity LIKE '%n%' -- Ignore empty
        -- AND unitadmitsource NOT LIKE "" -- Ignore empty
),
predictors AS (
    SELECT patientunitstayid,
        intubated,
        dialysis,
        vent,
        heartrate,
        temperature,
        sodium,
        meanbp,
        hematocrit,
        creatinine,
        glucose,
        albumin,
        bilirubin,
        wbc,
        bun
    FROM `physionet-data.eicu_crd.apacheapsvar`
),
troponin_lab AS (
    SELECT patientunitstayid,
        labresultoffset,
        labname,
        labresulttext AS labresult,
        labmeasurenamesystem
    FROM `physionet-data.eicu_crd.lab`
    WHERE labname = 'troponin - T'
),
max_troponin AS (
    SELECT patientunitstayid,
        MAX(labresult) AS max_troponin
    FROM troponin_lab
    GROUP BY patientunitstayid
),
prep_table AS (
    SELECT base_table.*,
        admit_reason.admitdxname,
        admit_reason.admitdxtext,
        max_troponin.max_troponin,
        CASE
            WHEN admit_reason.admitdxname NOT LIKE '%Cardi%'
            AND base_table.apacheadmissiondx NOT LIKE '%Cardi%' THEN 1
            ELSE 0
        END AS non_cardiac_patient,
        CASE
            WHEN base_table.unitdischargestatus = 'Expired' THEN 1
            ELSE 0
        END AS expired,
        ROUND(base_table.unitdischargeoffset / 60, 2) AS unitdischargeoffset_hours,
        predictors.intubated,
        predictors.dialysis,
        predictors.vent,
        predictors.heartrate,
        predictors.temperature,
        predictors.sodium,
        predictors.meanbp,
        predictors.hematocrit,
        predictors.creatinine,
        predictors.glucose,
        predictors.albumin,
        predictors.bilirubin,
        predictors.wbc,
        predictors.bun
    FROM base_table
        LEFT JOIN admit_reason ON base_table.patientunitstayid = admit_reason.patientunitstayid
        LEFT JOIN max_troponin ON base_table.patientunitstayid = max_troponin.patientunitstayid
        LEFT JOIN predictors ON base_table.patientunitstayid = predictors.patientunitstayid
),
qualified_table AS (
    SELECT patientunitstayid,
        hospitalid,
        unittype,
        unitadmitsource,
        gender,
        ethnicity,
        unitdischargeoffset_hours,
        max_troponin,
        non_cardiac_patient,
        expired,
        intubated,
        dialysis,
        vent,
        heartrate,
        temperature,
        sodium,
        meanbp,
        hematocrit,
        creatinine,
        glucose,
        albumin,
        bilirubin,
        wbc,
        bun,
        CASE
            WHEN prep_table.age LIKE '%>%'
            OR prep_table.age LIKE '%<%' THEN -1
            ELSE CAST(SUBSTRING(prep_table.age, 1, 2) AS INT64)
        END AS age
    FROM prep_table
),
final_table AS (
    SELECT *
    FROM qualified_table
    WHERE age > 0
    ORDER BY patientunitstayid
)
SELECT *
FROM final_table