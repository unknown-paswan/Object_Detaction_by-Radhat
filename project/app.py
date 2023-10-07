from flask import Flask, render_template

app = Flask(__name__, template_folder='templates')


import cv2

app = Flask(__name__)

# Load the pre-trained model and label map
model_path = 'frozen_inference_graph.pb'
config_path = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
labelmap_path = 'labels.txt'

# Load the model and configuration for the detection
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
label_map = {}
with open(labelmap_path, 'r') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        class_name = line.strip()
        label_map[idx + 1] = class_name

cap = cv2.VideoCapture(0)

def detect_objects():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Preprocess the frame for the model
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

        net.setInput(blob)

        # Run inference
        detections = net.forward()

        # Visualization of the results
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Set a confidence threshold
                class_id = int(detections[0, 0, i, 1])
                score = confidence
                left = int(detections[0, 0, i, 3] * width)
                top = int(detections[0, 0, i, 4] * height)
                right = int(detections[0, 0, i, 5] * width)
                bottom = int(detections[0, 0, i, 6] * height)

                # Draw bounding box and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                class_name = label_map[class_id]
                cv2.putText(frame, f'{class_name} {score:.2f}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert frame to JPEG format
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
