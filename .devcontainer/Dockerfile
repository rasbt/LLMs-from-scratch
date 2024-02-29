FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

RUN <<EOF
apt-get update
apt-get install -y --no-install-recommends git
rm -rf /var/lib/apt/lists/*
EOF

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
