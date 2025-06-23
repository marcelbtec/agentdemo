FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY insurance_mcp_education.py .

# Expose Streamlit port
EXPOSE 8080

# Run Streamlit
CMD streamlit run insurance_mcp_education.py \
    --server.port 8080 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.serverAddress 0.0.0.0 \
    --browser.gatherUsageStats false