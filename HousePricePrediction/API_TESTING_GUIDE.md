# House Price Prediction Flask API - Testing Guide

## Server Information
- **Base URL**: `http://127.0.0.1:5000`
- **Server**: Flask Development Server
- **Model**: Linear Regression (374 features)
- **Status**: ✅ Fully Operational

---

## Endpoints

### 1. Health Check Endpoint
**Endpoint**: `GET /`

**Purpose**: Verify API is running

**Response (200)**:
```json
{
  "status": "success",
  "message": "House Price Prediction API is running",
  "model_loaded": true,
  "timestamp": "2026-08-18T11:25:30.565499"
}
```

**Test with curl**:
```bash
curl http://127.0.0.1:5000/
```

---

### 2. API Information Endpoint
**Endpoint**: `GET /info`

**Purpose**: Get API documentation and required fields

**Response (200)**: Returns model info, required/optional fields, and example request

**Test with curl**:
```bash
curl http://127.0.0.1:5000/info
```

---

### 3. Prediction Endpoint
**Endpoint**: `POST /predict`

**Purpose**: Get house price prediction

#### Required Fields (Numerical):
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Area | float/int | Property area in square feet | 1200 |
| BHK | int | Number of bedrooms | 3 |
| Bathroom | float/int | Number of bathrooms | 2 |
| Parking | int | Number of parking spaces | 1 |
| Per_Sqft | float/int | Price per square foot | 8500 |

#### Optional Fields (Categorical):
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Furnishing | string | Furnishing type | Semi-Furnished, Unfurnished |
| Locality | string | Property locality/area | Dwarka, Dwarka Mor, etc. |
| Status | string | Property status | Ready_to_move, Under_Construction |
| Transaction | string | Transaction type | Resale, New_Property |
| Type | string | Property type | Apartment, Builder_Floor, Villa |

#### Example Request (with all fields):
```json
{
  "Area": 1500,
  "BHK": 3,
  "Bathroom": 2,
  "Parking": 1,
  "Per_Sqft": 9000,
  "Furnishing": "Semi-Furnished",
  "Locality": "Dwarka",
  "Status": "Ready_to_move",
  "Transaction": "Resale",
  "Type": "Apartment"
}
```

#### Example Response (200 - Success):
```json
{
  "status": "success",
  "predicted_price": 5292385.59,
  "predicted_price_formatted": "₹5,292,385.59",
  "input_summary": {
    "Area": 1500,
    "BHK": 3,
    "Bathroom": 2,
    "Parking": 1,
    "Per_Sqft": 9000
  }
}
```

#### Example Response (400 - Missing Field):
```json
{
  "status": "error",
  "message": "Missing required fields: Parking, Per_Sqft",
  "required_fields": ["Area", "BHK", "Bathroom", "Parking", "Per_Sqft"],
  "optional_fields": ["Furnishing", "Locality", "Status", "Transaction", "Type"]
}
```

#### Example Response (400 - Invalid Value):
```json
{
  "status": "error",
  "message": "Area must be a valid number"
}
```

---

## Testing Instructions

### Using curl

**Health Check**:
```bash
curl http://127.0.0.1:5000/
```

**Get API Info**:
```bash
curl http://127.0.0.1:5000/info
```

**Valid Prediction**:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Area": 1200,
    "BHK": 3,
    "Bathroom": 2,
    "Parking": 1,
    "Per_Sqft": 8500
  }'
```

**Invalid Request (Missing Field)**:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Area": 1200,
    "BHK": 3,
    "Bathroom": 2
  }'
```

---

### Using Python requests

```python
import requests
import json

base_url = 'http://127.0.0.1:5000'

# Health check
response = requests.get(f'{base_url}/')
print(response.json())

# Prediction
payload = {
    'Area': 1200,
    'BHK': 3,
    'Bathroom': 2,
    'Parking': 1,
    'Per_Sqft': 8500
}
response = requests.post(f'{base_url}/predict', json=payload)
print(response.json())
```

---

### Using Postman

1. **Import Collection** or create new requests:
   - **Method**: GET
   - **URL**: `http://127.0.0.1:5000/`
   - **Send**

2. **POST Prediction**:
   - **Method**: POST
   - **URL**: `http://127.0.0.1:5000/predict`
   - **Headers**: `Content-Type: application/json`
   - **Body** (raw JSON):
   ```json
   {
     "Area": 1200,
     "BHK": 3,
     "Bathroom": 2,
     "Parking": 1,
     "Per_Sqft": 8500
   }
   ```
   - **Send**

---

## Test Results Summary

### All Tests Passed ✅

1. **Health Check**: Returns 200 with correct status
2. **Valid Predictions**: Returns 200 with predicted prices
   - Test 1: ₹20,074,004.09
   - Test 2: ₹10,740,614.19
   - Test 3: ₹5,292,385.59 (with categorical features)
3. **Missing Fields**: Returns 400 with clear error message
4. **Invalid Values**: Returns 400 with validation error
5. **Categorical Features**: Properly handled and processed
6. **404 Errors**: Correctly identified invalid endpoints
7. **Error Handling**: All edge cases handled gracefully

---

## API Usage Notes

- All numerical fields must be valid numbers (int or float)
- Missing required fields result in HTTP 400
- Invalid numeric values result in HTTP 400
- Categorical features are optional
- Server runs in debug mode for development
- Feature names and order must match training data

---

## Running the Server

```bash
cd c:\Users\chand\Desktop\HousePricePrediction
python deployment/app.py
```

Server will start at: `http://127.0.0.1:5000`

---

## To Stop the Server

Press `CTRL+C` in the terminal running the Flask server.
