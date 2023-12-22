FROM python:3.8

ENV TZ=Asia/Hong_Kong
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY load_transformer.py .
RUN python load_transformer.py

COPY . .

EXPOSE 8080

# ENTRYPOINT [ "python", "nlp_pipeline.py" ]