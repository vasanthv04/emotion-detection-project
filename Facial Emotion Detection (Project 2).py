import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
faseCascade = cv2.CascadeClassifier("haar.xml")

while (True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faseCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Emotions Detection Starts
    if len(faces) > 0:
        emotions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        cv2.putText(frame, emotions[0]['dominant_emotion'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    # Emotion Detection Ends

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
    cv2.imshow('Face and Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
