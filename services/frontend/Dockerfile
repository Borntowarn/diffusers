FROM python:3.11

WORKDIR /app

COPY ./services/frontend/requirements.txt  ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./services/frontend ./frontend
COPY ./configs ./configs

ENV LOGURU_COLORIZE "true"

CMD ["python", "-m", "frontend"]