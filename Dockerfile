FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./application.py /code/
COPY ./models /code/models/

# 
CMD ["uvicorn", "application:app", "--host", "0.0.0.0", "--port", "8081"]
