import os
from io import BytesIO
import requests
from ultralytics import YOLO
from PIL import Image
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()
cred_path = os.getenv("FIREBASE_CREDENTIAL_PATH")
db_url = os.getenv("FIREBASE_DATABASE_URL")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'databaseURL': db_url
})
db = firestore.client()

# === Load YOLOv8 Model ===
model = YOLO('yolov8n.pt')


# === Function to Get Location Name from OpenStreetMap (Nominatim) ===
def get_location_name(lat, lon):
    try:
        if lat is None or lon is None:
            print("❌ Invalid latitude or longitude.")
            return "Unknown Location"

        print(f"📍 Resolving location name for coordinates: lat={lat}, lon={lon}")

        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {
            "User-Agent": "EcoCitySmartManagement/1.0"
        }

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code} error from Nominatim.")
            return "Unknown Location"

        data = response.json()
        address = data.get('address', {})
        road = address.get('road')
        suburb = address.get('suburb')
        city = address.get('city') or address.get('town') or address.get('village')

        location_name = road or suburb or city or "Unnamed Location"
        return location_name

    except Exception as e:
        print(f"⚠️ Error fetching location name: {e}")
        return "Unknown Location"


# === Load and Predict Image From Firebase ===
def main():
    docs = db.collection("traffic_image").get()
    img_urls = []
    for doc in docs:
        data = doc.to_dict()
        img_url = data.get("traffic_img_url")
        location_id = data.get('location_id')
        traffic_img_id = doc.id

        # Check if already processed
        if data.get("vehicleData_DocId"):
            print(f"🚫 Already processed: {traffic_img_id}")
            continue
        if not img_url:
            continue

        try:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))

            results = model(img)[0]

            # === Filter Detected Vehicles ===
            vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']  # Names used in YOLOv8
            vehicle_count = sum(1 for c in results.names.values() if c in vehicle_classes for b in results.boxes.cls if
                                results.names[int(b)] == c)

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

            doc_data = {
                "vehicleNum": vehicle_count,
                "congestionLevel": congestion,
                "createdDateTime": datetime.datetime.now().isoformat(),
                "location_id": location_id,
                "suggestion": suggestion
            }
            vehicle_data_ref = db.collection("vehicle_data").add(doc_data)
            vehicle_data_doc_id = vehicle_data_ref[1].id

            # === Update traffic_image with vehicle_data docID ===
            db.collection("traffic_image").document(traffic_img_id).update({
                "vehicleData_DocId": vehicle_data_doc_id
            })

            # === Create or Update Location in 'locations' collection ===
            location_ref = db.collection("locations").where("location_id", "==", location_id).get()
            lat_lon = location_id.split("_")
            lat = lat_lon[0]
            lon = lat_lon[1]

            if location_ref:
                for loc_doc in location_ref:
                    loc_doc_ref = db.collection("locations").document(loc_doc.id)
                    loc_doc_ref.update({
                        "lastUpdated": datetime.datetime.now().isoformat(),
                        "latest_trafficImage_DocId": traffic_img_id,
                        "latest_vehicleData_DocId": vehicle_data_doc_id,
                    })
                print(f"🚗 Location {location_id} updated.")

            else:
                print(f"📍 Creating new location for location_id: {location_id}")

                # Location does not exist, create new
                lat, lon = location_id.split("_")
                lat, lon = float(lat), float(lon)

                print(f"🧪 Parsed lat/lon: lat={lat}, lon={lon}")

                # Fetch the location name using Nominatim API
                name = get_location_name(lat, lon)

                lat_lon_doc_id = f"{lat}_{lon}"

                new_location_data = {
                    "location_id": location_id,
                    "lastUpdated": datetime.datetime.now().isoformat(),
                    "lat": lat,
                    "lon": lon,
                    "name": name,
                    "latest_trafficImage_DocId": traffic_img_id,
                    "latest_vehicleData_DocId": vehicle_data_doc_id,
                }
                try:
                    db.collection("locations").document(lat_lon_doc_id).set(new_location_data)
                    print(f"✅ New location {lat_lon_doc_id} added with name: {name}")
                except Exception as e:
                    print(f"❌ Failed to create location document: {e}")

            print("✅ YOLOv8 prediction uploaded to Firebase.")
            print(f"Vehicle Num:  {vehicle_count} \n Congestion Level: {congestion}")
        except Exception as e:
            print(f"The error existed: {e}")

if  __name__=="__main__":
    main()