Face Recognition System using OpenCV and Deep Learning

This project demonstrates a complete face recognition pipeline using OpenCV and deep learning-based models. The pipeline includes face detection, face embedding generation using the OpenFace model, and classification using an SVM.

Built with 💻 by Suraj kumar singh

🔍 Project Highlights

      *****  Detect faces in real-time using a pre-trained SSD model (res10_300x300_ssd_iter_140000.caffemodel)

      *****  Generate 128D facial embeddings using the OpenFace deep neural network

      *****  Train a Support Vector Machine (SVM) to classify faces based on embeddings

      *****  Recognize faces in live video stream from webcam

📌 Workflow

        1. Face Detection

                Utilize OpenCV’s DNN module with Caffe model files:

                deploy.prototxt

        2. Face Embedding

                Extract high-level facial features using OpenFace model:openface_nn4.small2.v1.t7

       3. Model Training

                Train an SVM classifier on extracted facial embeddings

        4. Real-time Recognition

                 Use a webcam feed to recognize faces based on the trained model

🧠 How It Works

  ****  Face DetectionOpenCV DNN loads a pre-trained face detector to identify face regions.

        Face Alignment (Optional)Align faces to normalize input using facial landmarks.

        Embedding GenerationThe OpenFace network maps each face to a 128-dimensional vector representation.

        ClassificationA trained SVM uses these vectors to classify or recognize known individuals.****

🚀 Getting Started

        1. Clone the Repository

            git clone https://github.com/surajkumarsingh03/Face-Detection-system
            cd Face-Detection-system

        2. Prepare Dataset

            Create a directory named dataset

            Add images of individuals you want to train the system on

            Subdirectories should be named after the person (e.g., dataset/name/)

        3.Extract Embeddings

            python extract_embeddings.py

        4.Train the Classifier

            python train_model.py

        5.Test on Live Video

            python recognize_video.py

📦 Requirements

        Python 3.5+

        OpenCV (with cv2.dnn)

        NumPy

        Scikit-learn

        Install Dependencies

        pip install opencv-python numpy scikit-learn

        Or on Ubuntu:

        sudo apt-get install python3-opencv

📷 Sample Output

Real-time face recognition from webcam:



📚 References

OpenFace Model

OpenCV DNN Face Detector

SSD Model: res10_300x300_ssd_iter_140000.caffemodel


