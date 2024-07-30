docker run -it --gpus all -v $(pwd):/app --env-file .env --user root 'tortcode/autoresearch:5.0' bash
