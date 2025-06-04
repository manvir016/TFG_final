import os
import cv2
from glob import glob



def extract_lips_from_frames(frames_dir, output_dir):
    size = (96, 96)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted(glob(os.path.join(frames_dir, "*.jpg")), key=lambda x: int(os.path.basename(x).split('.')[0]))
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Carpeta con los frames del render")
    parser.add_argument('--output', type=str, required=True, help="Carpeta donde guardar las bocas recortadas")
    args = parser.parse_args()

    extract_lips_from_frames(args.input, args.output)

