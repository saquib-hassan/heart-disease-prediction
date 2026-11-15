# Heart Disease Prediction API

An educational end-to-end ML deployment project using scikit-learn, FastAPI, Docker, and Render.

> **Medical disclaimer:** This model is for learning and software demonstration only. It is not clinically validated and must not be used to diagnose, treat, or make decisions about a person's health.

## Project structure

```text
app/                 FastAPI application and Pydantic schema
model/               Trained Joblib model pipeline (created by train.py)
data/                Place the downloaded heart.csv here
train.py             Train and save the model
Dockerfile           Build the API container
docker-compose.yml   Run locally with Docker Compose
```

## 1. Get the dataset

Download the CSV from the assignment's Kaggle page:
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Place the file at `data/heart.csv`. This training script expects the UCI-style 14-column dataset with these columns:

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`

The target must be binary: `0` = absence and `1` = presence. If the downloaded file uses a different target column name or encoding, adjust `train.py` accordingly.

## 2. Train the model locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
python train.py
```

The script trains a StandardScaler + LogisticRegression pipeline and saves it to `model/heart_model.joblib`.

The scaler is included in the saved pipeline, so the API applies the same preprocessing used during training.

## 3. Run FastAPI without Docker (optional)

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs

## 4. Build and run with Docker

Make sure `model/heart_model.joblib` exists before building.

```bash
docker compose build
docker compose up
```

Open:
- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Model information: http://localhost:8000/info

Stop with `Ctrl+C`, then optionally run `docker compose down`.

## 5. Test the prediction endpoint

Example request body (illustrative values only):

```json
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":2}'
```

The response includes `heart_disease` (boolean), the numeric prediction, and the model's estimated positive-class probability. This probability is not a clinical risk estimate.

## 6. Deploy to Render

1. Push this project to a GitHub repository.
2. Ensure `model/heart_model.joblib` is included in the repository. Do not commit the Kaggle dataset unless you have the right to redistribute it.
3. In Render, choose **New → Web Service**, connect the repository, and select **Docker** as the runtime/environment.
4. Set the Docker build context to the repository root if the Dockerfile is in the root.
5. Deploy. Render should detect the Dockerfile and build the image.
6. Open the deployed URL and test `/health`, `/info`, and `/docs`.
7. Test `POST /predict` using Swagger UI or Postman.

The Docker image listens on `0.0.0.0:8000`. If your Render service configuration provides a custom port, update the Docker CMD to use Render's `PORT` environment variable, for example with a shell command:
`CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]`

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | Health check and model status |
| GET | `/info` | Model type and expected features |
| POST | `/predict` | Predict binary target |

## Notes

- The model is deliberately simple; the assignment prioritizes API, Docker, and deployment.
- The input feature names and order must match the dataset used for training.
- The API validates basic input types and ranges, but these checks do not establish clinical validity.
- Only load Joblib artifacts that you created or otherwise trust. Joblib uses Python pickle mechanisms and can execute code when loading an untrusted file.
