FROM python:3.10

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy app code
COPY . .
RUN pip install --upgrade pip
# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Run your app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
