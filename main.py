import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

closed_eye_timer = None
closed_eye_duration = 2  # Adjust this value based on how long you want to consider the eyes closed
attention_level_duration = 2  # Adjust this value based on how long you want to consider the inattentiveness

eye_aspect_ratio_threshold = 2.7

# Variables to track inattentive duration
inattentive_timer = None
inattentive_duration = 0

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make a copy of the image with writable flag set to False
    image_copy = np.copy(image)

    results = face_mesh.process(image)

    # Update the writable flag of the image back to True
    image_copy.flags.writeable = True

    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            img_h, img_w, img_c = image_copy.shape
            face_3d = []
            face_2d = []

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append(([x, y, lm.z]))

            # Check the number of points before solving PnP
            if len(face_3d) >= 4 and len(face_2d) >= 4:
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rmat, jac = cv2.Rodrigues(rot_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = np.degrees(angles[0])
                y = np.degrees(angles[1])
                z = np.degrees(angles[2])
                # Calculate the eye aspect ratios
                left_eye_top = (face_landmarks.landmark[159].x, face_landmarks.landmark[159].y)
                left_eye_bottom = (face_landmarks.landmark[145].x, face_landmarks.landmark[145].y)
                left_eye_inner = (face_landmarks.landmark[263].x, face_landmarks.landmark[263].y)
                left_eye_outer = (face_landmarks.landmark[373].x, face_landmarks.landmark[373].y)

                right_eye_top = (face_landmarks.landmark[386].x, face_landmarks.landmark[386].y)
                right_eye_bottom = (face_landmarks.landmark[374].x, face_landmarks.landmark[374].y)
                right_eye_inner = (face_landmarks.landmark[33].x, face_landmarks.landmark[33].y)
                right_eye_outer = (face_landmarks.landmark[7].x, face_landmarks.landmark[7].y)

                left_eye_height = np.linalg.norm(np.array(left_eye_top) - np.array(left_eye_bottom))
                left_eye_width = np.linalg.norm(np.array(left_eye_inner) - np.array(left_eye_outer))

                right_eye_height = np.linalg.norm(np.array(right_eye_top) - np.array(right_eye_bottom))
                right_eye_width = np.linalg.norm(np.array(right_eye_inner) - np.array(right_eye_outer))

                left_eye_aspect_ratio = left_eye_height / left_eye_width
                right_eye_aspect_ratio = right_eye_height / right_eye_width

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # Check if the person is looking left, right, up, or down
                if y < -10:
                    text = "Looking left"
                elif y > 10:
                    text = "Looking right"
                elif x < -1:
                    text = "Looking down"
                elif x > 10 or y > 10 or y < -10:
                    text = "Looking up"
                else:
                    text = "Looking Forward"

                if y < -10 or y > 10 or x < -1 or x > 10:

                    # If the person is inattentive, start the timer
                    if inattentive_timer is None:
                        inattentive_timer = time.time()
                    inattentive_duration = time.time() - inattentive_timer
                else:

                    # If the person is attentive, reset the inattentive timer and duration
                    inattentive_timer = None
                    inattentive_duration = 0

                # Check if the person has been inattentive for a certain duration
                if inattentive_duration >= attention_level_duration:
                    cv2.putText(image_copy, "Inattentive", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(image_copy, "Attentive", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Reset the closed_eye_timer when the eyes are open
                if left_eye_aspect_ratio < eye_aspect_ratio_threshold and right_eye_aspect_ratio < eye_aspect_ratio_threshold:
                    if closed_eye_timer is None:
                        closed_eye_timer = time.time()
                    elif time.time() - closed_eye_timer >= closed_eye_duration:
                        cv2.putText(image_copy, "Eyes Closed", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    closed_eye_timer = None
                    cv2.putText(image_copy, "Eyes Open", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Draw head pose lines and other text
                    # Draw head pose lines and other text
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                     dist_matrix)
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                    cv2.line(image, p1, p2, (255, 0, 0), 3)
                    cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                    cv2.putText(image, "x" + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2)
                    cv2.putText(image, "y" + str(np.round(x, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2)
                    cv2.putText(image, "z" + str(np.round(x, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2)
            else:
                # Handle the case when you don't have enough points
                print("Not enough 3D-2D point correspondences")

                # Display the total inattentive duration
                cv2.putText(image_copy, f'Inattentive Duration: {int(inattentive_duration)}', (20, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    print("FPS: ", fps)

    cv2.putText(image_copy, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image_copy,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    cv2.imshow('HEAD POSE ESTIMATION', image_copy)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
