import cv2, os

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
datasets = 'datasets'  
sub_data = 'Rohit Sharma'     

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.makedirs(path)

(width, height) = (130, 100)   

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

count = 1
while count <= 100:
    print(f"Capturing image {count}")
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path, str(count).zfill(3)), face_resize)
        count += 1

    cv2.imshow('OpenCV', im)
    if cv2.waitKey(10) == 27:
        break

print("Dataset collection complete.")
webcam.release()
cv2.destroyAllWindows()
