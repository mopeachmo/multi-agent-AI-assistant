# Base
FROM python:3.11-slim

# .pyc（バイトコード）ファイルを作らない → コンテナ内をクリーンに保つ
ENV PYTHONDONTWRITEBYTECODE=1 \ 
# 標準出力を即時出力にする（ログがすぐ見られる）
    PYTHONUNBUFFERED=1
# -> ログが遅延せず即時に出力されるため、デバッグやモニタリングに便利。

WORKDIR /app

# システムレベルの依存パッケージをインストール
# ca-certificates: TLS, curl: healthcheck, tini: proper PID 1
RUN apt-get update && apt-get install -y --no-install-recommends \
 && rm -rf /var/lib/apt/lists/*

# Python deps 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code 
COPY . /app

# rootユーザーではなく安全な一般ユーザーで実行
RUN useradd -m appuser
USER appuser

# Streamlit(UI) config 
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

# Healthcheck 
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Runtime
CMD ["streamlit", "run", "main.py"]
