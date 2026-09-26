import requests, numpy as np, cv2

URL = "http://172.16.101.165:8080/photoaf.jpg"

def take_photo():
    img_bytes = requests.get(URL, timeout=15).content
    return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

def process(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 50:
        print("Dark scene -> trigger action")

while True:
    input("Press Enter to take a photo (Ctrl+C to quit)...")
    frame = take_photo()
    if frame is None:
        print("Failed to get image")
        continue
    cv2.imwrite("latest.png", frame)
    print("Saved latest.png", frame.shape)
    process(frame)
