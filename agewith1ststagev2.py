import cv2
import mediapipe as mp
import statistics
import time

# Load MediaPipe face detection and face mesh models
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
agecolor = [(34, 35, 36), (34, 35, 36), (55, 56, 57), (55, 56, 57), (60, 66, 67), (60, 66, 67), (34, 35, 36), (34, 35, 36)]
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
ageNet = cv2.dnn.readNet(ageModel, ageProto)
age_list = [[0], [0], [0], [0]]

maximum_face = 4
# Initialize webcam
cap = cv2.VideoCapture(0)

# Define quadrant coordinates
quadrant_coords = [
    (0, 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2),
     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)),
    (0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2),
     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)),
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2),
     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2))
]

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
        mp_face_mesh.FaceMesh() as face_mesh:
    roundcount = 0
    # last_face_time = [time.time()] * maximum_face  # Initialize last_face_time for each face
    # start_processing = [False] * maximum_face  # Initialize start_processing for each face
    last_face_time = {}  #store the last detected time for each face. The keys are face indices (face_index), and the values are the times those faces were last detected.
    start_processing = {}  #store the processing state for each face. Again, the keys are face indices (face_index), and the values are boolean flags indicating whether processing has started for those faces.
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        results = face_detection.process(image_rgb)

        # Check if any faces are detected
        if results.detections:
            counter = 0
            bbox_center_x = 0
            for face_index, detection in enumerate(results.detections):
                face_detected_time = time.time()  # Update the face detected time for each face

                # Change condition to check if face_index is in dictionary and if delay has passed
                if face_index not in last_face_time or face_detected_time - last_face_time[face_index] >= 5:
                    start_processing[face_index] = True
                    last_face_time[face_index] = face_detected_time

                # Change condition to use get method with default value False
                if start_processing.get(face_index, False):
                    bbox_center_x = 0
                    if roundcount == 10:

                        roundcount = 0
                    else:
                        roundcount += 1
                        print(roundcount)

                    # Extract face landmarks
                    bbox_cordinates = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    bbox = int(bbox_cordinates.xmin * w), int(bbox_cordinates.ymin * h), \
                            int(bbox_cordinates.width * w), int(bbox_cordinates.height * h)
                    # Calculate the center of the face bounding box
                    bbox_center_x = bbox[0] + bbox[2] // 2
                    bbox_center_y = bbox[1] + bbox[3] // 2

                    # Determine the quadrant the face belongs to
                    face_quadrant = -1
                    for i, (x, y, w, h) in enumerate(quadrant_coords):
                        if x <= bbox_center_x < x + w and y <= bbox_center_y < y + h:
                            face_quadrant = i
                            stringface = str(i)
                            break

                    # Crop face region for face mesh
                    face_image = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

                    # Convert face image to RGB
                    imgsize = face_image.shape
                    if imgsize[0] > 0 and imgsize[1] > 0:
                        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                        # Perform face mesh on the face image
                        face_mesh_results = face_mesh.process(face_image_rgb)

                        # Check if face mesh is available
                        if face_mesh_results.multi_face_landmarks:
                            for face_landmarks in face_mesh_results.multi_face_landmarks:
                                # Calculate the average x-coordinate of all face landmarks
                                avg_x = sum([landmark.x for landmark in face_landmarks.landmark]) / len(
                                    face_landmarks.landmark)

                        # Estimate age based on the average x-coordinate
                        blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                        ageNet.setInput(blob)
                        agePreds = ageNet.forward()
                        age_index = agePreds[0].argmax()
                        age_list[face_index].append(age_index)
                        if len(age_list[face_index]) > 10:
                            age_list[face_index].pop()
                        new_age = statistics.mode(age_list[face_index])

                        # Draw bounding box and age on the image
                        mp_drawing.draw_detection(image, detection)
                        cv2.putText(image, f"face: {face_index} Age: {new_age}", (bbox[0], bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        if bbox_center_x >= 300:
                            counter -= 1
                        else:
                            counter += 1

                        if roundcount == 9:
                            if counter < 0:
                                print("moter_left_move")
                            elif counter > 0:
                                print("moter_right_move")
                            else:
                                print("both_motor_move")
                        last_face_time[face_index] = time.time()  # Update the last face time for each face
                else:
                    # No faces detected
                    if time.time() - last_face_time[face_index] > 10:
                        # If no faces are detected for 10 seconds, stop everything
                        print(f"No face {face_index} detected for 10 seconds. Stopping...")
                        break

        # Display the output image
       
        cv2.imshow('Age Detection', image)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
