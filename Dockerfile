FROM python:3.14-slim

ARG INSTALL_FFMPEG=false
ENV PORT=5050

WORKDIR /app

# Install ffmpeg conditionally
RUN if [ "$INSTALL_FFMPEG" = "true" ]; then \
    apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*; \
    fi

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app directory
COPY app/ .

EXPOSE 5050

# Command to run the server
CMD ["python", "server.py"]
