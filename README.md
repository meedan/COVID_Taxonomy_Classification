# COVID_Taxonomy_Classification

## Running Flask locally for development

Clone this repository, populate the models directory, and then run 

```
pip install -r requirements.txt
FLASK_APP=application.py python -m flask run
```

Example request:
```
curl -X POST 'http://localhost:5000/covid/categorize' -d '{"text":"eat alkaline foods to prevent covid"}'
```
