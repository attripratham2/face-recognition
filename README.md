👤 Real-Time Face Recognition System using OpenCV
This project is a real-time face recognition system built with Python and OpenCV using the classical FisherFace algorithm for recognizing known faces via webcam. It's a beginner-friendly computer vision project that demonstrates face detection, training, and recognition in live video.

📌 Features
Detect faces in real-time using Haar Cascade.

Train on custom datasets organized by user folders.

Recognize known faces and label them with confidence scores.

Detect and log unknown persons by saving their image frames.

Easy to extend with more training data or better models (e.g., LBPH, Dlib, FaceNet).

🛠️ Tech Stack
Python 3.x

OpenCV (opencv-contrib-python)

NumPy

Haar Cascades (for face detection)

FisherFace Recognizer (for face recognition)

⚙️ How to Run the Project

git clone https://github.com/attripratham2/face-recognition.git
cd face-recognition-opencv
📦 Step 2: Install dependencies
Make sure you have Python 3.x installed. Then install dependencies using pip:

pip install opencv-contrib-python numpy
opencv-contrib-python is important as it includes the cv2.face module.


Step 3: Add Training Data
Create a folder named datasets, and inside it, create a subfolder for each person (e.g., Pratham) containing multiple grayscale face images (ideally 50–100, 130x100 size).
You can use a script like face_data_collect.py to collect them via webcam.

Example structure:datasets/
└── Rohit Sharma/
    ├── 1.png
    ├── 2.png
    └── ...
🧠 Step 4: Train and Recognize Faces
Run the main script:

python face_recognize.py

Press ESC key to exit.
If an unknown face appears consistently, it will save a snapshot as input.jpg.
✅ Notes
FisherFaceRecognizer works well when lighting conditions vary.

You can switch to LBPHFaceRecognizer for better results on small datasets.

Haar Cascades are fast, but deep learning models (like Dlib or FaceNet) can improve accuracy.
📈 Future Enhancements
Use deep learning for better accuracy (e.g., Dlib, FaceNet).

Add GUI for user registration.

Use SQLite/MySQL to store metadata.

Save/load the trained model for faster future use.

🤝 Contributions
Feel free to fork this repo, suggest improvements, or open pull requests!

📧 Contact
Created by Pratham Attri
📫 LinkedIn: https://www.linkedin.com/in/pratham-attri-7676301a0/

🔖 License
This project is open-source and available under the MIT License.






