import cv2
import os
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import cvzone

# Define the path to the video file
video_path = os.path.abspath("vid.mp4")
yolo_model_path = 'yolo11n-pose.pt'
xgb_model_path = 'trained_model.json'

def detect_shoplifting(video_path):
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return

    # Check if YOLO model exists
    if not os.path.exists(yolo_model_path):
        print(f"Error: YOLO model file {yolo_model_path} not found. Please download it or provide the correct path.")
        return

    # Check if XGBoost model exists
    if not os.path.exists(xgb_model_path):
        print(f"Error: XGBoost model file {xgb_model_path} not found. Ensure you have trained the model first.")
        return

    # Load YOLOv8 model
    try:
        model_yolo = YOLO(yolo_model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Load the trained XGBoost model
    try:
        model = xgb.Booster()
        model.load_model(xgb_model_path)
    except Exception as e:
        print(f"Error loading XGBoost model: {e}")
        return

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    print(f"Total Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_tot = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Finished processing video or frame could not be read.")
            break

        # Resize the frame for consistent processing
        frame = cv2.resize(frame, (1018, 600))

        # Run YOLOv8 on the frame
        results = model_yolo(frame, verbose=False)

        # Visualize the YOLO results on the frame
        annotated_frame = results[0].plot(boxes=False)

        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue
                
            bound_box = r.boxes.xyxy  # Bounding box coordinates
            conf = r.boxes.conf.tolist()  # Confidence levels
            keypoints = r.keypoints.xyn.tolist()  # Keypoints for human pose

            for index, box in enumerate(bound_box):
                if conf[index] > 0.55:  # Threshold for confidence score
                    x1, y1, x2, y2 = box.tolist()

                    # Prepare data for XGBoost prediction
                    data = {}
                    for j in range(len(keypoints[index])):
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    # Convert the data to a DataFrame
                    df_pred = pd.DataFrame(data, index=[0])

                    # Prepare data for XGBoost prediction
                    dmatrix = xgb.DMatrix(df_pred)

                    # Make prediction using the XGBoost model
                    sus = model.predict(dmatrix)
                    binary_predictions = (sus > 0.5).astype(int)

                    # Annotate the frame based on prediction (0 = Suspicious, 1 = Normal)
                    if binary_predictions == 0:  # Suspicious
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cvzone.putTextRect(annotated_frame, "Suspicious", (int(x1), int(y1)), 1, 1)      
                    else:  # Normal
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cvzone.putTextRect(annotated_frame, "Normal", (int(x1), int(y1) + 50), 1, 1)      

        # Show the annotated frame
        cv2.imshow('Shoplifting Detection', annotated_frame)
        frame_tot += 1

        # Press 'q' to stop early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_shoplifting(video_path)
