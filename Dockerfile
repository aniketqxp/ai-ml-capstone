FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    CAPSTONE_DATA_ROOT=/home/user/workdata \
    CAPSTONE_EVAL_RESULTS=/home/user/workdata/evaluation_results \
    CAPSTONE_FRONTEND_PUBLIC=/home/user/workdata/frontend_public

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user \
    && mkdir -p /home/user/workdata/na_testset \
                /home/user/workdata/evaluation_results \
                /home/user/workdata/frontend_public \
                /home/user/.cache/huggingface \
    && chown -R user:user /home/user/workdata /home/user/.cache

WORKDIR /home/user/app

COPY backend/requirements.txt backend/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r backend/requirements.txt

COPY backend/ backend/
COPY ml-services/ ml-services/
COPY --chown=user:user data/na_testset/manifest.json /home/user/workdata/na_testset/manifest.json

USER user
WORKDIR /home/user/app/backend

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
