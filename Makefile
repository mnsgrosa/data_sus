NETWORK_NAME := data_sus_network

.PHONY: all network build run stop clean

all: network build run

network:
	@echo "Checking for network: $(NETWORK_NAME)..."
	@docker network inspect $(NETWORK_NAME) >/dev/null 2>&1 || \
	(echo "Creating network $(NETWORK_NAME)..." && docker network create $(NETWORK_NAME))

build:
	@echo "Building docker compose services..."
	docker compose build --no-cache

run:
	@echo "Starting services..."
	docker compose up -d

stop:
	docker compose down

clean: stop
	@echo "Removing network..."
	docker network rm $(NETWORK_NAME) || true
