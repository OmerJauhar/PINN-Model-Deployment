# PINN Model Deployment

This project implements a Physics-Informed Neural Network (PINN) for structural engineering predictions, deployed as a FastAPI web service.

## Project Structure

```
PINN-Model-Deployment/
├── model.py          # PINN model architecture definition
├── main.py          # FastAPI server implementation
├── pinn_model.h5    # Trained model weights (not included in repo)
└── README.md        # This file
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- FastAPI
- Uvicorn
- NumPy
- Pydantic

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PINN-Model-Deployment.git
cd PINN-Model-Deployment
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture

The PINN model consists of:
- Input layer: 25 features
- 5 hidden blocks, each containing:
  - Dense layer
  - Batch Normalization
  - Dropout (except last block)
- Output layer: 1 prediction value

## API Endpoints

### GET /
- Returns a simple message indicating the model is ready
- Response: `{"message": "ML model is ready"}`

### POST /predict
- Accepts a JSON payload with 25 features
- Request body:
```json
{
    "features": [float, float, ...]  // 25 features
}
```
- Response:
```json
{
    "prediction": [float]  // Predicted value
}
```

## Running the Server

1. Make sure you have the trained model file (`pinn_model.h5`) in the project directory

2. Start the server:
```bash
python main.py
```

The server will start on `http://0.0.0.0:8000`

## Example Usage

Using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [0.1, 0.2, ...]}'  # Add all 25 features
```

Using Python requests:
```python
import requests
import json

url = "http://localhost:8000/predict"
data = {"features": [0.1, 0.2, ...]}  # Add all 25 features
response = requests.post(url, json=data)
prediction = response.json()["prediction"]
```

## Model Features

The model expects 25 input features in the following order:
1. Bridge Width (m)
2. Bridge Height (m)
3. Cross-Sectional Diameter (m)
4. Cross-Sectional Area (m²)
5. Cross-Sectional Moment of Inertia (m⁴)
6. Number of Strands
7. Number of Beams
8. Angle of Inclination (°)
9. Angle of Declination (°)
10. Young's Modulus (Pa)
11. Poisson's Ratio
12. Density (kg/m³)
13. Tensile Yield Strength (Pa)
14. Shear Modulus (Pa)
15. Mesh Elements
16. Mesh Density (elements/m³)
17. Max Equivalent Stress (Pa)
18. Max Principal Stress (Pa)
19. Min/Max Deformation (m)
20. Safety Factor
21. Strain Energy (J)
22. Work Done (J)
23. Energy Residual (J)
24. Yield Constraint Residual
25. Max Failure Load (N)

## Notes

- The model requires a trained weights file (`pinn_model.h5`) to be present in the project directory
- All input features should be properly scaled before sending to the API
- The server runs on CPU by default

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]
