import cv2, numpy, os

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
print('Training...')

(images, labels, names, id) = ([], [], {}, 0)
(width, height) = (130, 100)  # Ensure consistent image size for training

# Walk through dataset folders
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            img = cv2.imread(path, 0)
            if img is not None:
                img = cv2.resize(img, (width, height))  # ✅ Ensure uniform size
                images.append(img)
                labels.append(id)
        id += 1

# Convert to NumPy arrays
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# Train the model using FisherFaceRecognizer
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

# Initialize face detection and webcam
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
cnt = 0

# Start real-time face recognition
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1] < 800:
            name = names[prediction[0]]
            cv2.putText(im, f'{name} - {prediction[1]:.0f}', (x-10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 255, 255), 2)
            print(name)
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, 'Unknown', (x-10, y-10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("input.jpg", im)
                cnt = 0

    cv2.imshow('OpenCV', im)
    if cv2.waitKey(10) == 27:  # ESC to exit
        break

webcam.release()
cv2.destroyAllWindows()
