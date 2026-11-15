from pydantic import BaseModel, ConfigDict, Field

class HeartDiseaseInput(BaseModel):
    """Input features in the same order and units as the training dataset."""
    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="0=female, 1=male (dataset coding)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (dataset coding)")
    trestbps: int = Field(..., ge=50, le=300, description="Resting blood pressure")
    chol: int = Field(..., ge=50, le=1000, description="Serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result")
    thalach: int = Field(..., ge=30, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=20, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (dataset coding)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia category (dataset coding)")
