import requests
import json

base_url = 'http://127.0.0.1:5000'

print('='*60)
print('TESTING API WITH CATEGORICAL FEATURES')
print('='*60)

# Test with categorical features
print('\n1. Valid Prediction with Categorical Features')
print('-'*60)
try:
    payload = {
        'Area': 1500,
        'BHK': 3,
        'Bathroom': 2,
        'Parking': 1,
        'Per_Sqft': 9000,
        'Furnishing': 'Semi-Furnished',
        'Locality': 'Dwarka',
        'Status': 'Ready_to_move',
        'Transaction': 'Resale',
        'Type': 'Apartment'
    }
    print(f'Input: {json.dumps(payload, indent=2)}')
    response = requests.post(f'{base_url}/predict', json=payload, timeout=5)
    print(f'Status Code: {response.status_code}')
    resp_data = response.json()
    pred_price = resp_data.get('predicted_price_formatted', 'N/A')
    status = resp_data.get('status', 'N/A')
    print(f'Prediction: {pred_price}')
    print(f'Status: {status}')
except Exception as e:
    print(f'Error: {str(e)}')

# Test without categorical features
print('\n2. Prediction Without Categorical Features (optional)')
print('-'*60)
try:
    payload = {
        'Area': 1500,
        'BHK': 3,
        'Bathroom': 2,
        'Parking': 1,
        'Per_Sqft': 9000
    }
    print(f'Input (no categorical): {json.dumps(payload, indent=2)}')
    response = requests.post(f'{base_url}/predict', json=payload, timeout=5)
    print(f'Status Code: {response.status_code}')
    resp_data = response.json()
    pred_price = resp_data.get('predicted_price_formatted', 'N/A')
    status = resp_data.get('status', 'N/A')
    print(f'Prediction: {pred_price}')
    print(f'Status: {status}')
except Exception as e:
    print(f'Error: {str(e)}')

# Test 404 endpoint
print('\n3. Testing 404 - Nonexistent Endpoint')
print('-'*60)
try:
    response = requests.get(f'{base_url}/nonexistent', timeout=5)
    print(f'Status Code: {response.status_code}')
    print(f'Response: {json.dumps(response.json(), indent=2)}')
except Exception as e:
    print(f'Error: {str(e)}')

print('\n' + '='*60)
print('✓ CATEGORICAL FEATURE TESTS COMPLETED')
print('='*60)
