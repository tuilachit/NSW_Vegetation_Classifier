# Run Locally

## Backend

Use Python 3.11.

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Expected when assets are installed:

```json
{
  "ok": true,
  "models_loaded": true,
  "missing_assets": [],
  "soil_assets_ready": true,
  "missing_soil_assets": []
}
```

## Frontend

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8080
```

## API Endpoints

- `GET /health`
- `GET /model-info`
- `POST /predict-tile`
- `POST /predict-raw`

## Notes

The frontend defaults to `http://127.0.0.1:8000` for API calls. To use a different API URL, set `window.VEGEMAP_API_BASE` before loading `app.js`.
