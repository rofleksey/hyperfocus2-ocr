# hyperfocus2-ocr — survivor-name OCR microservice.
#
# The runtime base is glibc-based (Debian slim): onnxruntime and
# opencv-python-headless only publish manylinux (glibc) wheels, so a musl/Alpine
# base cannot install them.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OCR_HOST=0.0.0.0 \
    OCR_PORT=8081 \
    OCR_THREADS=1 \
    OCR_HYBRID=true \
    OCR_LOG_LEVEL=info

WORKDIR /opt/ocr

# libglib2.0-0 + libgomp1 are shared-lib deps of opencv/onnxruntime; libgl1 is a
# fallback for OpenCV. rapidocr_onnxruntime pulls the ONNX models via pip.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8081

HEALTHCHECK --interval=10s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        urllib.request.urlopen('http://localhost:${OCR_PORT}/healthz', timeout=5).read(); sys.exit(0)" \
    || exit 1

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8081"]
