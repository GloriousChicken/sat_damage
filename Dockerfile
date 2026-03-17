FROM tensorflow/tensorflow:2.16.2

RUN pip install --upgrade pip

COPY requirements_prod.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY satdamage /satdamage

CMD uvicorn satdamage.api.fast:app --host 0.0.0.0 --port $PORT
