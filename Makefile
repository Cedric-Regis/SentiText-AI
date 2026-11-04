.PHONY: build data classical lstm train evaluate app test all

build:
	docker compose build

data:
	docker compose run --rm prepare

classical:
	docker compose run --rm train-classical

lstm:
	docker compose run --rm train-lstm

train: data classical lstm

evaluate:
	docker compose run --rm train-classical python -m src.evaluate

app:
	docker compose up app

test:
	docker compose run --rm app pytest tests

all: build train evaluate app
