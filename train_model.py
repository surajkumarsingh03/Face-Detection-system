from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

print("[INFO] loading face embeddings...")
with open("output/embeddings.pickle", "rb") as f:
    data = pickle.load(f)

print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

with open("output/recognizer.pickle", "wb") as f:
    pickle.dump(recognizer, f)

with open("output/le.pickle", "wb") as f:
    pickle.dump(le, f)
