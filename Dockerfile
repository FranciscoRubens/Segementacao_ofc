# Imagem base com PyTorch 2.3 e CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Diretório de trabalho
WORKDIR /app

# Variáveis de ambiente
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copiar apenas o requirements.txt primeiro
COPY requirements.txt .

# Atualizar pip e instalar dependências Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código
COPY . .

# Expor porta (caso tenha API)
EXPOSE 8080

# Comando padrão
CMD ["python", "train_final.py"]