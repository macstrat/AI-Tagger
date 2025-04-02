import os
import subprocess
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from ultralytics import YOLO

def install_requirements():
    print("📦 Installing required packages...")
    subprocess.check_call(["pip", "install", "-r", "requirements.txt"])

def download_blip2_model(model_name='flan-t5-xl'):
    print(f"⬇️ Downloading BLIP2 model: {model_name}")
    model_path = f"Salesforce/blip2-{model_name}"
    Blip2Processor.from_pretrained(model_path, cache_dir="./models")
    Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype="float32", device_map="auto", cache_dir="./models")
    print("✅ BLIP2 downloaded.")

def download_yolo_model():
    print("⬇️ Downloading YOLOv8x model...")
    yolo = YOLO("yolov8x.pt")
    yolo.model.save("models/yolov8x.pt")
    print("✅ YOLOv8x model downloaded.")

def main():
    os.makedirs("models", exist_ok=True)
    install_requirements()
    download_blip2_model()
    download_yolo_model()
    print("🎉 Setup complete!")

if __name__ == "__main__":
    main()
