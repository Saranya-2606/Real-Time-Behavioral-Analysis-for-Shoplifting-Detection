import os
import cv2
from ultralytics import YOLO
import pandas as pd
from config import NORMAL_PATH, RAW_IMAGES_DIR, CSV_FILE_PATH

# Load your YOLO model
model = YOLO("yolo11s-pose.pt")

# Video path
cap = cv2.VideoCapture('nm1.mp4')

if not cap.isOpened():
    print("Error: Could not open video nm1.mp4")
else:
    # Get video properties
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    seconds = round(frames / fps) if fps > 0 else 0

    frame_total = 1000
    i = 0
    a = 0

    all_data = []

    while cap.isOpened():
        # Set the position in milliseconds
        if seconds > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, (i * ((seconds / frame_total) * 1000)))
        flag, frame = cap.read()

        if not flag:
            break

        # Save full frame image
        image_path = os.path.join(RAW_IMAGES_DIR, f'img_{i}.jpg')
        cv2.imwrite(image_path, frame)

        # Run YOLO detection
        results = model(frame, verbose=False)

        for r in results:
            bound_box = r.boxes.xyxy  # Get bounding boxes
            conf = r.boxes.conf.tolist()  # Confidence score
            keypoints = r.keypoints.xyn.tolist()  # Human keypoints

            for index, box in enumerate(bound_box):
                if conf[index] > 0.75:
                    x1, y1, x2, y2 = box.tolist()
                    cropped_person = frame[int(y1):int(y2), int(x1):int(x2)]
                    output_path = os.path.join(NORMAL_PATH, f'person_nn_{a}.jpg')

                    data = {'image_name': f'person_nn_{a}.jpg'}

                    # Save keypoint data
                    for j in range(len(keypoints[index])):
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    all_data.append(data)
                    cv2.imwrite(output_path, cropped_person)
                    a += 1

        i += 1

    print(f"Total frames processed: {i}, Total cropped images saved: {a}")
    cap.release()
    cv2.destroyAllWindows()

    # Convert to DataFrame
    if all_data:
        df = pd.DataFrame(all_data)

        # Check if the file exists to determine whether to append or create new
        if not os.path.isfile(CSV_FILE_PATH):
            df.to_csv(CSV_FILE_PATH, index=False)  # Create new file if it doesn't exist
        else:
            df.to_csv(CSV_FILE_PATH, mode='a', header=False, index=False)  # Append if it exists

        print(f"Keypoint data saved to {CSV_FILE_PATH}")
    else:
        print("No keypoints detected.")
