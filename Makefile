NAME := fedPlay
PYTHON_VERSION := 3.7

create-nodes:
	docker-compose up --build

run-docker:
	docker run --rm -d -p 50052:50052 --name $(NAME) $(NAME)

kill-docker:
	docker kill $(NAME)
