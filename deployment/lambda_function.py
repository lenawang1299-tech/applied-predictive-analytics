import json
import os
import boto3
import numpy as np
import io
import csv

S3_BUCKET = os.environ["MODEL_BUCKET"]
MODEL_KEY = os.environ.get(
    "MODEL_KEY", "models/telco_churn_light_artifact.json"
)

def load_artifact(s3_client):
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
    artifact_wrapper = json.loads(response["Body"].read().decode("utf-8"))
    return artifact_wrapper["artifact"]

def preprocess(row, artifact):
    # Numeric features
    numeric_features = artifact["numeric_features"]
    medians = artifact["numeric_imputer_medians"]
    means = artifact["numeric_means"]
    scales = artifact["numeric_scales"]

    numeric_vec = []
    for i, col in enumerate(numeric_features):
        val = row.get(col, None)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = medians[i]
        if val != val:  # NaN check
            val = medians[i]
        val = (val - means[i]) / scales[i]
        numeric_vec.append(val)

    # Categorical features
    categorical_features = artifact["categorical_features"]
    cat_imputer_values = artifact["categorical_imputer_values"]
    categories = artifact["categories"]

    cat_vec = []
    for i, col in enumerate(categorical_features):
        val = row.get(col, None)
        if val is None or val == "":
            val = cat_imputer_values[i]
        cats = categories[i]
        ohe = [1.0 if val == c else 0.0 for c in cats]
        cat_vec.extend(ohe)

    return np.array(numeric_vec + cat_vec, dtype="float32")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict(feature_vec, artifact):
    coef = np.array(artifact["coef"][0], dtype="float32")
    intercept = float(artifact["intercept"][0])
    score = np.dot(coef, feature_vec) + intercept
    prob = sigmoid(score)
    label = 1 if prob >= artifact["threshold"] else 0
    return float(prob), label

def lambda_handler(event, context):
    s3 = boto3.client("s3", region_name="us-east-1")

    # Load artifact
    artifact = load_artifact(s3)

    # Handle both direct Lambda invocation and API Gateway invocation
    if "body" in event and event["body"]:
        body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        input_key = body.get("input_key", "input/telco_churn_full.csv")
    else:
        input_key = event.get("input_key", "input/telco_churn_full.csv")

    # Read input CSV from S3
    response = s3.get_object(Bucket=S3_BUCKET, Key=input_key)
    content = response["Body"].read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))

    results = []
    for row in reader:
        # Drop customerID if present
        row.pop("customerID", None)
        # Fix TotalCharges
        if "TotalCharges" in row:
            try:
                row["TotalCharges"] = float(row["TotalCharges"])
            except (ValueError, TypeError):
                row["TotalCharges"] = None
        features = preprocess(row, artifact)
        prob, label = predict(features, artifact)
        results.append({
            "churn_probability": round(prob, 4),
            "churn_prediction": label
        })

    # Write results to S3 output folder
    output_key = "output/predictions.json"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=output_key,
        Body=json.dumps(results, indent=2).encode("utf-8"),
        ContentType="application/json"
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": f"Processed {len(results)} rows",
            "output_location": f"s3://{S3_BUCKET}/{output_key}",
            "preview": results[:3]
        })
    }
