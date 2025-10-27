# Start from the provided base image which already contains a pipeline file
FROM agrigorev/zoomcamp-model:2025

# set working dir
WORKDIR /code

# copy your FastAPI app and pyproject so we have everything needed
COPY app.py .
COPY pyproject.toml .

# Install runtime dependencies we'll need (match what's in pyproject.toml)
# We explicitly install the common runtime deps so image is simple and reproducible.
# --no-cache-dir keeps the image smaller.
RUN pip install --no-cache-dir fastapi uvicorn[standard] pandas scikit-learn

# expose the port the app will run on
EXPOSE 8000

# run the app with uvicorn, listening on all interfaces so the container is reachable
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
