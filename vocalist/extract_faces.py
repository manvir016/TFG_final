import os
import cv2
from glob import glob

base_path = "/home/mkaur/tfg/prueba2/vocalist/videos/"
size = (96, 96)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for video_folder in sorted(os.listdir(base_path)):
    folder_path = os.path.join(base_path, video_folder)
    frames_path = os.path.join(folder_path, "frames")

    if not os.path.isdir(frames_path):
        print(f"⚠️  No se encontró 'frames/' en {folder_path}, se salta.")
        continue

    print(f"📂 Procesando: {video_folder}")
    frame_files = sorted(glob(os.path.join(frames_path, "*.jpg")), key=lambda x: int(os.path.basename(x).split('.')[0]))
    count = 0

    for frame_file in frame_files:
        img = cv2.imread(frame_file)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue  # no se detectó cara

        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        try:
            face = cv2.resize(face, size)
        except:
            continue  # si falla el resize

        out_path = os.path.join(folder_path, f"{count}.jpg")
        cv2.imwrite(out_path, face)
        count += 1

    print(f"✅ {count} caras guardadas en {video_folder}")

