FROM python:3.11.7

COPY ["main.py", \
        "utils.py", \
        "requirements.txt", \
        "feature_scaler_PAstro.joblib", \
        "LSTMpredLC__PAstro.joblib", \
        "target_scaler_PAstro.joblib", \
        "/app/"]
WORKDIR /app
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
