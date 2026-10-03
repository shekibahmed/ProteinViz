# ProteinViz web app for self-hosting (any CPU host). The free hosted demo uses Streamlit Community Cloud.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROTEINVIZ_CACHE_DIR=/tmp/proteinviz-cache

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md LICENSE DATA_LICENSES.md ./
COPY --chown=user src ./src
COPY --chown=user .streamlit ./.streamlit
RUN pip install ".[app]"

USER user
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')"
CMD ["proteinviz", "app", "--host", "0.0.0.0", "--port", "7860"]
