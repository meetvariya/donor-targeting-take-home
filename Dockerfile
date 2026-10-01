FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.6.13 /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY pyproject.toml uv.lock README.md ./
COPY donor_targeting ./donor_targeting
RUN uv sync --frozen --no-dev
EXPOSE 8000
CMD ["/app/.venv/bin/python", "-m", "donor_targeting.service"]