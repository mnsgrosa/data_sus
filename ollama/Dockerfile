FROM ollama/ollama:latest

ARG MODEL_NAME=llama3.1:8b

RUN ollama serve & \
    sleep 5 && \
    ollama pull ${MODEL_NAME} && \
    pkill ollama