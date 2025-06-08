import os
import subprocess

# CONFIG
input_dir = "/home/mkaur/tfg/prueba2/vocalist/datasetvoca"
output_dir = "/home/mkaur/tfg/prueba2/vocalist/videos"
test_file = "/home/mkaur/tfg/prueba2/vocalist/filelists/test.txt"
desfases = [round(0.05 * i, 2) for i in range(1, 11)]  # 0.05 to 0.5

# CLEAN test.txt
open(test_file, "w").close()

def run_cmd(cmd):
    subprocess.run(cmd, check=True)

def desync_video(input_video, delay, output_video):
    run_cmd([
        "ffmpeg", "-i", input_video,
        "-itsoffset", str(delay), "-i", input_video,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-strict", "experimental",
        output_video
    ])

def extract_audio_frames(video_path, folder):
    os.makedirs(os.path.join(folder, "frames"), exist_ok=True)
    run_cmd(["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "16000", os.path.join(folder, "audio.wav")])
    run_cmd(["ffmpeg", "-i", video_path, "-r", "25", os.path.join(folder, "frames", "%d.jpg")])

def extract_faces():
    run_cmd(["python3", "extract_faces.py"])

# MAIN LOOP
for file in sorted(os.listdir(input_dir)):
    if not file.endswith(".mp4"):
        continue

    base_name = file.replace(".mp4", "")
    input_path = os.path.join(input_dir, file)

    for delay in desfases:
        delay_str = str(delay).replace('.', '')
        folder_name = f"{base_name}_{delay_str}"
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        output_video = os.path.join(folder_path, "video.mp4")

        print(f"🎬 [{base_name}] → Desfase {delay}s → {folder_name}")
        desync_video(input_path, delay, output_video)
        extract_audio_frames(output_video, folder_path)

        with open(test_file, "a") as f:
            f.write(f"{folder_name}\n")

# EXTRAER CARAS
print("🧠 Extrayendo caras con extract_faces.py ...")
extract_faces()

print("✅ ¡Todo listo! test.txt actualizado.")

