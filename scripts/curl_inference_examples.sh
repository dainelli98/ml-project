# Example curl requests for MLflow model inference
# The model expects the following features:
# ["price", "income", "advertising", "population", "age", "education", "shelve_loc", "urban", "us", "comp_price"]
#
# The server is running at http://127.0.0.1:7072

# 1. DataFrame records orientation (recommended for pandas DataFrame input)
curl -X POST http://127.0.0.1:7072/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_records": [
      {"price": 120, "income": 60, "advertising": 10, "population": 300, "age": 40, "education": 16, "shelve_loc": 3, "urban": 1, "us": 1, "comp_price": 110},
      {"price": 90, "income": 45, "advertising": 5, "population": 200, "age": 35, "education": 14, "shelve_loc": 2, "urban": 0, "us": 0, "comp_price": 95}
    ]
  }'

# 2. DataFrame split orientation
curl -X POST http://127.0.0.1:7072/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_split": {
      "columns": ["price", "income", "advertising", "population", "age", "education", "shelve_loc", "urban", "us", "comp_price"],
      "data": [
        [120, 60, 10, 300, 40, 16, 3, 1, 1, 110],
        [90, 45, 5, 200, 35, 14, 2, 0, 0, 95]
      ]
    }
  }'

# 3. Inputs (list of lists, order must match features)
curl -X POST http://127.0.0.1:7072/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      [120, 60, 10, 300, 40, 16, 3, 1, 1, 110],
      [90, 45, 5, 200, 35, 14, 2, 0, 0, 95]
    ]
  }'

# 4. Instances (list of dicts, similar to records)
curl -X POST http://127.0.0.1:7072/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "instances": [
      {"price": 120, "income": 60, "advertising": 10, "population": 300, "age": 40, "education": 16, "shelve_loc": 3, "urban": 1, "us": 1, "comp_price": 110},
      {"price": 90, "income": 45, "advertising": 5, "population": 200, "age": 35, "education": 14, "shelve_loc": 2, "urban": 0, "us": 0, "comp_price": 95}
    ]
  }'

# Replace values as needed for your use case.
