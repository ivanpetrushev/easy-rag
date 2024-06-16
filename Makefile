RETRIEVER_CMD = docker run
IMG_NAME = local-rag

build-images:
	docker build -t ${IMG_NAME} image/ --no-cache

build-images-quick:
	docker build -t ${IMG_NAME} image/

run-pdfloader: build-images-quick
	docker run -it --env-file .env ${IMG_NAME}:latest python3 pdfloader.py

run-tools: build-images-quick
	docker run -it --env-file .env ${IMG_NAME}:latest python3 tool-usage.py

pip-freeze:
	docker run -it --env-file .env ${IMG_NAME}:latest pip freeze