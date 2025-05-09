# Ecocity Traffic Image Analysis 🚗

## Overview 😎
- This subproject utilizes the **YOLOv8 model** for real-time traffic image analysis. 
- Fetches images from a **Firebase** database, processes them to detect vehicles, and updates congestion levels based on the number of detected vehicles. 

## Requirements ✅
1. Python 3.12 or above
2. Libraries:
 - `os`, `io`, `requests`, `ultralytics`, `PIL`,`firebase_admin`, `dotenv`

## Setup Instructions 📦
1. Clone the repository
```bash 
git clone "https://github.com/jiayin04/ecocity-traffic-prediction.git"
cd ecocity-traffic-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Environment variables
- Create `.env` file in root directory and add:
```bash
FIREBASE_CREDENTIAL_PATH=<fileName>.json
FIREBASE_DATABASE_URL= <firebase-database-url> 
```

## Run it 🏃‍♂️‍➡️
- Click to run the `main.py` file or
```bash
python main.py
```




