# COVID_Taxonomy_Classification

## Running with FastAPI

Clone this repository, populate the models directory, and then run 

```
pip install -r requirements.txt
uvicorn application:app --host 0.0.0.0 --port 8081
```

Example request:
```
curl -X POST 'http://localhost:8081/covid/categorize' -H 'Content-Type: application/json' -d '{"text":"eat alkaline foods to prevent covid"}'
```
