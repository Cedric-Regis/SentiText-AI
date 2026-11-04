.PHONY: build data classical lstm lstm-quick train evaluate app test all

build:
	docker compose build

data:
	docker compose run --rm prepare

classical:
	docker compose run --rm train-classical

lstm:
	docker compose run --rm train-lstm

# Test rapide du LSTM : n'écrit rien dans models/ ni results/
lstm-quick:
	docker compose run --rm train-lstm python -m src.train_lstm --quick

train: data classical lstm

evaluate:
	docker compose run --rm app python -m src.evaluate

app:
	docker compose up app

test:
	docker compose run --rm app python -m pytest tests -v

all: build train evaluate app
