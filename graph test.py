import pickle
import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
import time
import random
from threading import Thread
from queue import Queue,   Empty
import matplotlib.pyplot as plt
import mysql.connector

#Connect DB
db = mysql.connector.connect(host='localhost', user='root', password='', database='csrp')

if db.is_connected():
    print("conmnected")
#db.close()


# Initialize x_data and y_data outside the loop
x_data = []
y_data = []

# Initialize the graph
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('Status')
ax.set_title('Attentiveness Level')
line, = ax.plot([], [], 'b-', label='Attentiveness Level')
ax.legend()
#ax.set_ylim(0, 50)
plt.show()



# Global variable to store data
global_data = {
    "x": [],
    "y": []
}

db_update_time = time.time()
def update_graph(data_queue):
    x = 0
    y = 0

    while True:
        x += 1

        # Put the custom data in the queue for the graph thread to process
        data_queue.put((x, y))

        # Store data in the global variable
        global_data["x"].append(x)
        global_data["y"].append(y)

        # Wait for 1 second
        time.sleep(1)


data_queue = Queue()

# Create a thread for the graph function
graph_thread = Thread(target=update_graph, args=(data_queue,))
graph_thread.start()  # Start the thread

file_path = r'M:\Mediapipe model\attentiveness_detection_RF_6th_verstion_.pkl'
with open(file_path , 'rb') as f:
    model = pickle.load(f)

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                      )

            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array(
                    [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array(
                    [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Concate rows
                row = pose_row + face_row

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                print(body_language_class, body_language_prob)

                # Get status box
                cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                            , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                if body_language_class == "Attentive":
                    global_data["y"][-1] += 1
                else:
                    global_data["y"][-1] = max(global_data["y"][-1] - 0.1, 0)

                if time.time() - db_update_time >= 1:
                    try:
                        #Insert x and y values
                        cursor = db.cursor()
                        sql = ("INSERT INTO graph (x_coord, y_coord) VALUES (%s, %s)")
                        values = (global_data["x"][-1], global_data["y"][-1])
                        cursor.execute(sql, values)
                        db.commit()
                        print(cursor.rowcount, "Added")

                    except Exception as e:
                        print("Error:", e)

            except:
                pass

            # Update the graph with your custom data from the global variable
            line.set_data(global_data["x"], global_data["y"])
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

            print(global_data["x"])

            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()