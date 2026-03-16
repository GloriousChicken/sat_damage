FROM tensorflow/tensorflow:2.16.2

COPY satdamage /satdamage
COPY requirements_prod.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn satdamage.api.fast:app --host 0.0.0.0 --port $PORT
