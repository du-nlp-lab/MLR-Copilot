docker run -it --gpus all -v $(pwd):/app --env-file .env --user root 'tortcode/nlp-coresearcher:latest' bash
