# Data setup

Raw datasets are excluded from this public portfolio to respect source licensing and avoid redistributing third-party data.

## Customer churn

Download the **Telco Customer Churn** dataset from its [Kaggle dataset page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save the CSV as:

```text
data/telco_customer_churn.csv
```

The training script expects the standard 21-column schema, including `customerID`, service and contract features, account charges, and the `Churn` target.

## Bitcoin prices

Provide a daily Bitcoin price CSV as:

```text
data/bitcoin_daily_prices.csv
```

It must contain:

- `Date` formatted as `MM/DD/YYYY`
- `High` containing the daily high price in USD

The included results were produced from daily observations beginning in 2019, with training through 2022 and later observations reserved for chronological testing.

## Optional inference sample

For the Lambda workflow, provide a CSV with the same predictor columns as the churn dataset but without requiring a `Churn` target. Upload it to the configured S3 bucket and pass its object key in the API request.
