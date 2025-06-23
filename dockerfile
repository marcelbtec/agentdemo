FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY agentdemo.py .

# Expose port
EXPOSE 8080

# Run Streamlit with proper configuration
CMD streamlit run agentdemo.py \
    --server.port 8080 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.serverAddress 0.0.0.0 \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none