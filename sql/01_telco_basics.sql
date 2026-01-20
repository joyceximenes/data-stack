SELECT *
FROM telco_churn_raw
LIMIT 5;


SELECT "customerID", "Contract", "MonthlyCharges", "Churn"
FROM telco_churn_raw
WHERE "Churn" = 'Yes'
LIMIT 10;

SELECT "customerID", "Contract", "Churn"
FROM telco_churn_raw
WHERE "Contract" = 'Month-to-month'
LIMIT 10;


SELECT "Churn", COUNT(*) AS qtd
FROM telco_churn_raw
GROUP BY "Churn";

SELECT "Contract", COUNT(*) AS qtd
FROM telco_churn_raw
GROUP BY "Contract"
ORDER BY qtd DESC;


