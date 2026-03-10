import cv2
import pyttsx3

# initialize voice
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# load prebuilt haarcascade model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# start camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("GreenGase Vision System Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        speak("Person detected ahead")

    cv2.imshow("GreenGase Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()