import cv2
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms, models
from PIL import Image
from collections import deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
num_classes = 7
model = models.resnet50(weights=None)
num_features = model.fc.in_features


model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_features, 256),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(256),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, num_classes)
)

# загружаем веса
model.load_state_dict(torch.load("emotion_resnet50_balanced.pth", map_location=device))

model.to(device)
model.eval()

print("Model loaded successfully")

mtcnn = MTCNN(keep_all=True, device=device)  # детектирует все лица на кадре
emotion_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']


transform_face = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

emotion_buffer = deque(maxlen=5)

# Подключение к веб-камере
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру")
    exit()

print("Press 'q' or 'Esc' to exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра")
            break

        # переводим в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # детекция лиц
        boxes, _ = mtcnn.detect(frame_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # защита от выхода за границы
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                face = frame_rgb[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face_pil = Image.fromarray(face)
                face_tensor = transform_face(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(face_tensor)
                    pred = torch.argmax(output, dim=1).item()

                
                emotion_buffer.append(pred)
                stable_pred = max(set(emotion_buffer), key=emotion_buffer.count)

                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame,
                            emotion_labels[stable_pred],
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2)

        cv2.imshow("Emotion Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Finished")