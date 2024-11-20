import cv2
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Hàm xử lý ảnh
def preprocess_images(image_paths, size=(64, 64)):
    data = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"Could not read image: {path}")
            continue
        resized = cv2.resize(image, size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        data.append(gray.flatten())
    return np.array(data)

# Hàm huấn luyện và đánh giá mô hình
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=1)  # zero_division=1 để tránh lỗi
        recall = recall_score(y_test, y_pred, average="macro", zero_division=1)

        results[model_name] = {
            "Training Time (s)": elapsed_time,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
        }

        print(f"\nModel: {model_name}")
        print(classification_report(y_test, y_pred, target_names=["animal", "flower"], zero_division=1))  # zero_division=1

    return results

# Dữ liệu mẫu
image_paths = ['images/meo1.jpg', 'images/meo2.jpg', 'images/meo3.jpg', 'images/meo4.jpg']
labels = ['flower', 'flower', 'animal', 'animal']

# Tiền xử lý dữ liệu
X = preprocess_images(image_paths)
if X.size == 0:
    print("No valid images found. Exiting...")
    exit()

# Mã hóa nhãn
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Khởi tạo các mô hình
models = {
    "SVM": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(n_neighbors=min(2, len(X_train))),  # Giảm n_neighbors nếu số mẫu ít
    "Decision Tree": DecisionTreeClassifier(),
}

# Huấn luyện và đánh giá
results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

# In kết quả so sánh
print("\n--- Comparison of Results ---")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
