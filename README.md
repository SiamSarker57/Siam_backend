<<<<<<< HEAD
# SCALA-Guard Backend v2.0

FastAPI-based threat intelligence backend for package security analysis.

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the ML model (first time only)
python train_model.py

# 3. (Optional) Set DeepSeek API key for real AI remediation
export DEEPSEEK_API_KEY="your-key-here"

# 4. Run the server
uvicorn main:app --reload --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/analyze` | Analyze uploaded package file |
| POST | `/analyze/name` | Analyze by package name |
| POST | `/analyze/text` | Analyze free text / IOC note |
| POST | `/analyze/batch` | Batch scan requirements.txt / package.json / csv |
| GET | `/history` | Scan history |
| GET | `/history/{scan_id}` | Get single scan by id |
| GET | `/stats` | Dashboard statistics |
| DELETE | `/history/{scan_id}` | Delete single scan by id |
| DELETE | `/history` | Clear history |

The `/analyze` endpoint accepts package archives, PDF, CSV, DOC, and DOCX files (max 20MB). For PDF/DOCX uploads, the backend tries best-effort text extraction and still accepts the file even if extraction is unavailable.

## Postman Collection

Use `SCALA-Guard-Backend.postman_collection.json` from this backend folder and import it into Postman. Set `baseUrl` to your running API address (default: `http://localhost:8000`).

## Example Usage

```bash
# Analyze by name
curl -X POST http://localhost:8000/analyze/name \
  -H "Content-Type: application/json" \
  -d '{"name": "requests", "ecosystem": "pypi"}'

# Analyze uploaded PDF or package archive
curl -X POST http://localhost:8000/analyze \
  -F "package_file=@sample.pdf"

# Analyze uploaded DOCX
curl -X POST http://localhost:8000/analyze \
  -F "package_file=@sample.docx"

# Analyze raw text
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "saw requests-fake trying to exfiltrate data", "ecosystem": "pypi"}'

# Batch scan
curl -X POST http://localhost:8000/analyze/batch \
  -F "package_file=@requirements.txt"

# Batch scan with CSV (column: package/name/dependency or first column)
curl -X POST http://localhost:8000/analyze/batch \
  -F "package_file=@dependencies.csv"
```

## Tech Stack
- **FastAPI** — REST API
- **Scikit-learn** — ML risk scoring (Random Forest)
- **DeepSeek API** — AI remediation (optional)
- **Docker/strace** — Sandbox (production deployment)
=======
# Siam_backend
>>>>>>> 549265781be5de0035cdf13b06d9d3a78991adc0
