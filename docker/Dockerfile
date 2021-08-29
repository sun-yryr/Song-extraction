FROM python:3.9-buster

# Install ffmpeg
RUN apt update \
    && apt install -y ffmpeg

# Install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH=$PATH:/root/.poetry/bin
RUN poetry config virtualenvs.create false

WORKDIR /app

# COPY poetry.lock pyproject.toml ./
COPY . .

RUN poetry install

ENTRYPOINT [ "python", "-u", "cut-video.py" ]
