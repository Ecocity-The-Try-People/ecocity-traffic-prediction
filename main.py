from io import BytesIO
import requests
from ultralytics import YOLO
from PIL import Image
import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# === Firebase Init ===
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smartcity-8ef94-default-rtdb.asia-southeast1.firebasedatabase.app'
})
db = firestore.client()

# === Load YOLOv8 Model ===
model = YOLO('yolov8n.pt')

# === Load and Predict Image From Firebase ===
docs = db.collection("traffic_image").get()
img_urls = []
for doc in docs:
    data = doc.to_dict()
    img_url = data.get("traffic_img_url")
    location = data.get('location')
    if not img_url:
        continue

    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))

        results = model(img)[0]

        # === Filter Detected Vehicles ===
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']  # Names used in YOLOv8
        vehicle_count = sum(1 for c in results.names.values() if c in vehicle_classes for b in results.boxes.cls if results.names[int(b)] == c)

        # === Define Congestion Level ===
        if vehicle_count < 10:
            congestion = "Low"
            suggestion = "Continue on your current lane"
        elif vehicle_count < 25:
            congestion = "Medium"
            suggestion = "Will be congested soon, try to switch lane or route"
        else:
            congestion = "High"
            suggestion = "Please reroute, road is congested"

        # === Upload Image to Firebase Storage ===
        # filename = f"traffic_{datetime.datetime.now().isoformat()}.jpg"
        # blob = bucket.blob(f"traffic_images/{filename}")
        # blob.upload_from_filename(img_path)
        # image_url = blob.public_url

        # === Store Prediction in Firestore ===
        doc = {
            "vehicleNum": vehicle_count,
            "congestionLevel": congestion,
            "createdDateTime": datetime.datetime.now().isoformat(),
            "location": location,
            "suggestion": suggestion
        }
        db.collection("vehicle_data").add(doc)
        print("✅ YOLOv8 prediction uploaded to Firebase.")
        print(f"Vehicle Num:  {vehicle_count} \n Congestion Level: {congestion}")
    except Exception as e:
        print(f"The error existed: {e}")