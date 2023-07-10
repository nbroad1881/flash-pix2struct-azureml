FROM nvcr.io/nvidia/pytorch:23.04-py3

# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN apt-get update && apt-get -y install libfreetype6-dev
RUN pip uninstall -y pillow && \
    pip install --no-cache-dir pillow
