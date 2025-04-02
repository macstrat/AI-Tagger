# tagger.py
import os
import json
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import subprocess
import yaml
import nltk
from nltk.corpus import wordnet as wn
from ultralytics import YOLO
import inflect
p = inflect.engine()
import re
import sys
import threading
from threading import Thread
import itertools
from itertools import cycle
import colorama
from colorama import Fore, Style, init
init(autoreset=True)


print(" ")
print("-----------------------------")
print("==  Mac's AI Image Tagger  ==")
print("-----------------------------")
print(" ")

COLOR_MAP = {
    "red": Fore.RED,
    "orange": Fore.LIGHTRED_EX,
    "yellow": Fore.YELLOW,
    "green": Fore.GREEN,
    "blue": Fore.CYAN,
    "purple": Fore.MAGENTA,
    "pink": Fore.LIGHTMAGENTA_EX,
    "brown": Fore.LIGHTYELLOW_EX,
    "beige": Fore.WHITE,
    "teal": Fore.LIGHTCYAN_EX,
    "gray": Fore.LIGHTBLACK_EX,
    "white": Fore.WHITE,
    "black": Fore.RESET,
    "sepia": Fore.LIGHTYELLOW_EX,
    "black and white": Fore.LIGHTWHITE_EX
}


BAD_TAGS = {'getting', 'donuts', 'doing', 'looking', 'thing', 'stuff', 'object'}
COLOR_FIXES = {
    'orange rose': 'white rose',
    'red rose': 'white rose',
    'pink rose': 'white rose'
}

def spinner(msg="Working...", delay=0.1):
    done_flag = {"done": False}

    def spin():
        for c in itertools.cycle('|/-\\'):
            if done_flag["done"]:
                break
            try:
                sys.stdout.write(f'\r{msg} {c}')
                sys.stdout.flush()
                time.sleep(delay)
            except KeyboardInterrupt:
                break
        sys.stdout.write('\r' + ' ' * (len(msg) + 2) + '\r')

    t = Thread(target=spin)
    t.daemon = True  # Ensures it dies with the main thread
    t.start()

    def stop():
        done_flag["done"] = True
        t.join()
    return stop
    
def chunk_list(data, chunk_size):
    """Split a list into smaller chunks."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def clean_tags(tags):
    cleaned = set()
    fragments = {"on", "with", "at", "in", "a", "the", "and", "of"}
    ignore = {"orange"} if "lemon" in tags else set()
    for tag in tags:
        tag = tag.strip().lower()

        # Skip known fragments or accidental connectors
        if tag in fragments:
            continue

        # Skip known bad pairs (e.g. orange with lemon)
        if tag in ignore:
            continue

        # Remove too-short or meaningless entries
        if len(tag) < 2:
            continue

        # Filter anything longer than 3 words (likely a sentence)
        if len(tag.split()) > 3:
            continue

        cleaned.add(tag)
    return sorted(cleaned)

# These should already be there from previous step:
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)



import face_recognition

from PIL import Image
import numpy as np

def count_faces(image_path, img_array=None):
    if img_array is None:
        img_array = np.array(Image.open(image_path).convert("RGB"))

    try:
        img = Image.open(image_path)

        # tqdm.write(f"[{image_path}] original mode: {img.mode}")

        if img.mode not in ['RGB', 'L']:
            # tqdm.write(f"[{image_path}] converting from {img.mode} to RGB")
            img = img.convert('RGB')
        
        max_size = 800  # or whatever size you prefer
        img.thumbnail((max_size, max_size))
        img_array = np.array(img)

        if img_array.ndim == 2:
            # Convert grayscale to RGB
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            # Convert single-channel grayscale to RGB
            img_array = np.repeat(img_array, 3, axis=2)
        elif img_array.ndim != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Invalid shape: {img_array.shape}")
        

        # tqdm.write(f"Image mode before face detection: {img_array.dtype}, shape: {img_array.shape}")

        face_locations = face_recognition.face_locations(img_array)
        return len(face_locations)

    except Exception as e:
        raise RuntimeError(f"Face detection error in {image_path}: {e}")
        return 0

def detect_isolated_subject(image_path, yolo_model, img=None, img_array=None, yolo_results=None):
    if img is None:
        img = Image.open(image_path).convert("RGB")
    if img_array is None:
        img_array = np.array(img)
    results = yolo_results

    """
    Returns True if the subject is fully visible (not cropped), not a person,
    and the background is mostly white or black.
    """
    try:
        from PIL import Image
        import numpy as np
        import face_recognition

        # Use preloaded image and array
        width, height = img.size


        # === 1. Check for faces (exclude people)
        face_locations = face_recognition.face_locations(img_array)
        if len(face_locations) > 0:
            # print("[REJECT] Face detected — not isolated")
            return False

        # === 2. Check for mostly white or black background
        flat_pixels = img_array.reshape(-1, 3)
        white_pixels = np.sum(np.all(flat_pixels >= 240, axis=1))
        black_pixels = np.sum(np.all(flat_pixels <= 15, axis=1))
        total_pixels = flat_pixels.shape[0]

        white_ratio = white_pixels / total_pixels
        black_ratio = black_pixels / total_pixels

        # === Skip fallback if background is white or black ===
        if white_ratio >= 0.4 or black_ratio >= 0.4:
            # print("[SKIP] Background is clean (white/black) — skipping fallback")
            return True
        
        # === Fallback to contour detection ===
        # print("[YOLO] Multiple or no objects found — falling back to contour detection")
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
           # print("[REJECT] No contour found in fallback")
            return False
        
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        box_area = w * h
        # tqdm.write(f"[Fallback] Contour area ratio: {box_area / (width * height):.2f}")
        if box_area < 0.6 * width * height:
            # print("[REJECT] Fallback area too small")
            return False
        
        
        # === 4. Check that object is not cropped at edges
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        border_thresh = 0.03  # 3% margin
        if (
            x1 < border_thresh * width or
            y1 < border_thresh * height or
            x2 > (1 - border_thresh) * width or
            y2 > (1 - border_thresh) * height
        ):
            # print("[REJECT] Object touches image edge — likely cropped")
            return False
        
        
        # print("[SUCCESS] Isolated subject detected")
        return True

    except Exception as e:
        raise RuntimeError(f"[Isolated Detection Error] {image_path}: {e}")
        return False

VALID_COLOR_TONES = {
    "black", "white", "gray", "red", "orange", "yellow", "green",
    "blue", "purple", "pink", "brown", "beige", "teal", "sepia", "black and white"
}

def get_color_tone_tag_with_blip(image):
    prompt = (
        "Choose one word from this list that best matches the overall tone or hue of this image: "
        "black, white, gray, red, orange, yellow, green, blue, purple, pink, brown, beige, teal, sepia, black and white. "
        "If none apply, respond with 'unknown'."
    )
    inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to(blip_model.device)
    output_ids = blip_model.generate(**inputs)
    result = blip_processor.decode(output_ids[0], skip_special_tokens=True)

    tag = result.strip().lower()
    return tag if tag in VALID_COLOR_TONES else None

    # Fuzzy match result against valid tones
    match = get_close_matches(result, VALID_COLOR_TONES, n=1, cutoff=0.6)
    return match[0] if match else None


from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Optional: replace with your own curated list of 90s-style activities
activity_keywords = [
    "working", "celebrating", "dancing", "meeting", "laughing", "reading", "drinking", "thinking", "arguing",
    "cooking", "eating", "shopping", "talking", "hugging", "driving", "relaxing", "walking", "running", "jumping",
    "pointing", "teaching", "typing", "writing", "calling", "cleaning", "sleeping", "waking", "watching",
    "listening", "greeting", "presenting", "playing", "gardening", "biking", "swimming", "skating", "clapping",
    "high-fiving", "singing", "posing", "climbing", "traveling", "waiting", "standing", "sitting", "thinking",
    "photographing", "filming", "repairing", "volunteering", "training", "discussing", "studying", "exercising",
    "painting", "decorating", "caring", "fixing", "celebrating", "cheering", "jogging", "interviewing",
    "counseling", "helping", "consulting", "leading", "organizing", "directing", "collaborating", "shaking hands",
    "planning", "browsing", "flipping", "explaining", "cleaning", "styling", "answering", "stretching", "attending",
    "napping", "parenting", "skateboarding", "snowboarding", "cheering", "waving", "testing", "thinking",
    "conversing", "grilling", "celebrating", "jumpstarting", "protesting", "rejoicing", "scrubbing", "examining"
]

def enrich_tags(tags, caption, face_count, image_path):
    enriched = set(tags)
    caption_lower = caption.lower()
    
    # try:
    #     img = Image.open(image_path).convert("RGB")
    #     tone_tag = get_color_tone_tag_with_blip(img)
    #     if tone_tag:
    #         tags.add(tone_tag)
    # except Exception as e:
    #     print(f"[ToneTag] Error getting tone for {image_path}: {e}")


    # Face/person logic
    if face_count >= 3:
        enriched.add("group")
    elif face_count == 2:
        if "man" in caption_lower and "woman" in caption_lower:
            enriched.update(["man", "woman", "couple"])
        elif "men" in caption_lower or caption_lower.count("man") >= 2:
            enriched.add("men")
        elif "women" in caption_lower or caption_lower.count("woman") >= 2:
            enriched.add("women")
        else:
            enriched.add("pair")
    elif face_count == 1:
        if "woman" in caption_lower or "female" in caption_lower:
            enriched.add("woman")
        elif "man" in caption_lower or "male" in caption_lower:
            enriched.add("man")
        elif any(word in caption_lower for word in ["child", "kid"]):
            enriched.add("child")
        else:
            enriched.add("person")

    # Children logic
    if any(word in caption_lower for word in ["children", "kids"]):
        enriched.add("children")
    elif any(word in caption_lower for word in ["child", "kid"]):
        enriched.add("child")

    # Accurate noun extraction and pluralization
    words = word_tokenize(caption)
    tagged = pos_tag(words)

    noun_counts = {}
    for word, pos in tagged:
        if pos.startswith("NN"):
            singular = lemmatizer.lemmatize(word.lower(), pos='n')
            noun_counts[singular] = noun_counts.get(singular, 0) + 1

    for noun, count in noun_counts.items():
        if count > 1:
            enriched.add(p.plural(noun))  # plural only if more than one
        else:
            enriched.add(noun)  # singular noun if only one occurrence

    # Exclude generic terms unless explicitly texture/material related
    generic_exclusions = {"background", "surface", "close", "next"}
    material_context = {"texture", "pattern", "tile", "fabric", "marble",
                        "wood", "metal", "grain", "stone", "granite", "abstract"}

    if not enriched & material_context:
        enriched -= generic_exclusions

    # Limited curated set of 90s-stock activities (easy to maintain, high value)
    activity_keywords = {
        "shopping", "driving", "running", "jumping", "pointing", "teaching", "typing",
        "cleaning", "sleeping", "clapping", "dancing", "reading", "cooking", "eating",
        "talking", "hugging", "relaxing", "walking", "playing", "gardening", "biking",
        "swimming", "singing", "painting", "decorating", "fixing", "cheering",
        "skateboarding", "snowboarding", "waving", "grilling"
    }

    for word in activity_keywords:
        if word in caption_lower:
            enriched.add(word)

    # Minimalist contextual inference for "family" or "work"
    if "family" in caption_lower:
        enriched.add("family")
    work_terms = {"office", "meeting", "coworker", "colleague", "desk",
                  "computer", "presentation", "conference", "business"}
    if any(term in caption_lower for term in work_terms):
        enriched.add("work")
        
    # Remove generic 'person' tag if gender is already present
    if 'person' in enriched and ('man' in enriched or 'woman' in enriched):
        enriched.discard('person')

    return sorted(enriched)

def load_config():
    """
    Loads or creates a config.yaml file, merging any user-defined settings.
    """
    config_path = 'config.yaml'
    config = {
        'write_exif': True,    # Enabled by default
        'recursive': True,     # Recurse into subfolders by default
        'model_name': 'flan-t5-xl',
        'max_tags': 30,        # Limit number of tags
        'min_tag_length': 3    # Minimum tag word length
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                config.update(user_config)
        except Exception as e:
            raise RuntimeError(f"Error reading config file: {e}")
    
    return config

def validate_tags_with_caption(caption, tags, processor, model, image):
    tag_str = ", ".join(tags)
    prompt = (
        f"Here is a description of an image: \"{caption}\"\n\n"
        f"The current tags are: [{tag_str}]\n\n"
        "Are these tags correct and relevant for the photo? Answer yes or no."
    )
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True).strip().lower()
    return answer


def setup_model(model_name):
    """
    Loads the BLIP2 model and processor using float32 to avoid half-precision issues.
    """
    try:
        if model_name == 'opt-2.7b':
            processor = Blip2Processor.from_pretrained(
                'Salesforce/blip2-opt-2.7b',
                cache_dir='./models'
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                'Salesforce/blip2-opt-2.7b',
                torch_dtype=torch.float32,
                device_map='auto',
                cache_dir='./models'
            )
        elif model_name == 'flan-t5-xl':
            processor = Blip2Processor.from_pretrained(
                'Salesforce/blip2-flan-t5-xl',
                cache_dir='./models'
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                'Salesforce/blip2-flan-t5-xl',
                torch_dtype=torch.float32,
                device_map='auto',
                cache_dir='./models'
            )
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        return processor, model
    except Exception as e:
        raise RuntimeError(f"Model setup error: {e}")
        raise

def setup_yolo(model_path='models/yolov8x.pt'):
    # tqdm.write(f"Loading YOLO model from {model_path}")
    return YOLO(model_path)
    
def extract_folder_context(folder_path):
    if "test" in folder_path.lower():
        return ""

    folder_name = os.path.basename(folder_path)
    context = re.sub(r"[-_]", " ", folder_name).strip()
    return context


def generate_batch_captions(image_paths, processor, model, preloaded_images=None):
    images = []
    for i, image_path in enumerate(image_paths):
        if preloaded_images and i < len(preloaded_images):
            image = preloaded_images[i]
        else:
            image = Image.open(image_path).convert("RGB")
        images.append(image)

    prompt_caption = (
        "Describe this photo like you would for a stock photography database. "
        "Be specific, and limit your response to one sentence. "
        "Focus on visible objects, people, setting, and composition. Avoid abstract terms."
    )

    inputs = processor(images=images, text=[prompt_caption]*len(images), return_tensors='pt', padding=True)
    for k, v in inputs.items():
        if v.dtype in [torch.float32, torch.float64, torch.float16]:
            inputs[k] = v.to(model.device, dtype=torch.float32)
        else:
            inputs[k] = v.to(model.device)

    with torch.no_grad():
        caption_ids = model.generate(**inputs, max_new_tokens=60)
    captions = processor.batch_decode(caption_ids, skip_special_tokens=True)

    # === Now tag generation ===
   

    prompt_tag = "Create 10 descriptive stock photo tags based on this sentence: "

    prompts = []
    for img_path, cap in zip(image_paths, captions):
        context = extract_folder_context(img_path)
        if context:
            full_prompt = (
                f"{prompt_tag}\"{cap.strip()}\". "
                f"This image is from a set titled \"{context}\". "
                "Avoid abstract or emotional terms. Only include clearly visible objects, settings, or actions. Return a comma-separated list."
            )
        else:
            full_prompt = (
                f"{prompt_tag}\"{cap.strip()}\". "
                "Avoid abstract or emotional terms. Only include clearly visible objects, settings, or actions. Return a comma-separated list."
            )
        prompts.append(full_prompt)
    

    tag_inputs = processor(images=images, text=prompts, return_tensors='pt', padding=True)
    for k, v in tag_inputs.items():
        if v.dtype in [torch.float32, torch.float64, torch.float16]:
            tag_inputs[k] = v.to(model.device, dtype=torch.float32)
        else:
            tag_inputs[k] = v.to(model.device)

    with torch.no_grad():
        tag_ids = model.generate(**tag_inputs, max_new_tokens=30)
    tag_outputs = processor.batch_decode(tag_ids, skip_special_tokens=True)

    # Process output
    result = []
    for cap, tag_output in zip(captions, tag_outputs):
        cap = cap.strip()

        # Strip leading/trailing quotes from model output
        tag_output = tag_output.strip().strip('"').strip("'")

        # Split on commas
        keywords = [kw.strip() for kw in tag_output.split(",") if kw.strip()]

        # Filter out keywords that are sentences (more than 3 words)
        keywords = [kw for kw in keywords if len(kw.split()) <= 3]

        # Filter junk
        keywords = [k for k in keywords if k.lower() not in {
            "description", "descriptions", "detailed", "detail", "jpg", "jpeg", "image", "photo", "picture", "pictures"
        }]

        result.append((cap, keywords))
    return result

from nltk.corpus import wordnet as wn

def extract_tags(caption, config):
    words = nltk.word_tokenize(caption.lower())
    tagged = nltk.pos_tag(words)
    
    stopwords_raw = config.get("tag_fragments", "")
    stopwords = set(f.strip().lower() for f in stopwords_raw.split(",") if f.strip())

    # Completely remove verbs from tags
    tags = set(
        w for w, pos in tagged
        if pos.startswith(('NN', 'JJ'))  # ONLY nouns and adjectives allowed
        and w not in stopwords
        and len(w) >= config.get('min_tag_length', 3)
    )

    return sorted(tags)[:config.get('max_tags', 15)]

def write_exif(image_path, caption, tags):
    try:
        exiftool_exe = r'C:\exiftool\exiftool.exe'
        
        cmd = [
            exiftool_exe,
            '-charset', 'utf8',
            '-overwrite_original',
            f'-IPTC:Caption-Abstract={caption}',
            f'-XMP-dc:Title={caption}',
            f'-XMP-dc:Description={caption}',
        ]

        # ✅ Add each tag individually to avoid quotes/wrapping
        for tag in sorted(tags):
            cmd.append(f'-XMP-dc:Subject={tag}')
            cmd.append(f'-IPTC:Keywords={tag}')

        cmd.append(image_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stderr:
            raise RuntimeError(f"ExifTool Error ({image_path}): {result.stderr.strip()}")
        # if result.stdout:
            # tqdm.write(f"ExifTool Output ({image_path}): {result.stdout.strip()}")

    except Exception as e:
        raise RuntimeError(f"Error writing EXIF for {image_path}: {e}")

def process_folder(root_folder, config, processor, model):
    yolo_model = setup_yolo('models/yolov8x.pt')
    supported_exts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    chunk_size = 20

    for dirpath, _, filenames in os.walk(root_folder):
        # 🚫 Skip folder if tagger_log.txt exists
        log_filename = config.get("log_file_name", "tagger_log.txt")
        log_file_path = os.path.join(dirpath, log_filename)
        if os.path.exists(log_file_path):
            tqdm.write(f"✅ {os.path.basename(dirpath)} has been processed and has a log file")
            continue

        folder_start_time = time.time()
        error_count = 0
        image_count = 0
        image_files = [
            f for f in filenames
            if Path(f).suffix.lower() in supported_exts
            and f.lower() != '!.jpg'
        ]
        if not image_files:
            continue

        image_paths = [os.path.join(dirpath, f) for f in image_files]
        folder_log = []  # Collects log entries to save at the end of this folder
        tqdm.write(" ")
        tqdm.write(f"📂 Found {len(image_paths)} image(s) in folder: {dirpath}")

        for chunk_idx, chunk_paths in enumerate(chunk_list(image_paths, chunk_size)):
            chunk_images = []
            chunk_arrays = []

            tqdm.write(f"\n🔄 Processing Chunk {chunk_idx+1} ({len(chunk_paths)} images)")
            import psutil
            process = psutil.Process(os.getpid())
            mem_used_mb = process.memory_info().rss / 1024 ** 2
            tqdm.write(f"🧠 Estimated RAM used: {mem_used_mb:.2f} MB")
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2
                tqdm.write(f"🖥️ Estimated GPU memory used: {gpu_mem:.2f} MB")
        
            
            for path in chunk_paths:
                img = Image.open(path).convert("RGB")
                img.thumbnail((1000, 1000), Image.LANCZOS)
                chunk_images.append(img)
                chunk_arrays.append(np.array(img))

            stop_spinner = spinner("🕒 Generating captions and tags... (this might take a while, go get some tea)")
            chunk_captions_tags = generate_batch_captions(chunk_paths, processor, model, preloaded_images=chunk_images)
            stop_spinner()

            for idx, path in enumerate(chunk_paths):
                tqdm.write(f"\n⏳ Starting image: {os.path.basename(path)}")
                img_start_time = time.time()
                filename = os.path.basename(path)
                folder_name = os.path.basename(os.path.dirname(path))

                
                # Load original image and save original mode BEFORE conversion
                original_img = Image.open(path)
                original_mode = original_img.mode
                
                # Convert to RGB and resize
                img = original_img.convert("RGB")
                img.thumbnail((1000, 1000), Image.LANCZOS)
                img_array = np.array(img)
                
                # Replace chunked versions (optional if you don’t need chunk_images anymore)
                chunk_images[idx] = img
                chunk_arrays[idx] = img_array
                
                img_array = chunk_arrays[idx]
                caption = ""
                keywords = []
                try:
                    result = chunk_captions_tags[idx]
                    if isinstance(result, tuple) and len(result) == 2:
                        caption, keywords = result
                    else:
                        raise ValueError("Invalid caption/tags format")
                except Exception as e:
                    error_lines.append(f"Caption/Tag unpacking failed: {e}")
                

                tags = set()
                error_lines = []
                face_count = 0
                isolated = False
                yolo_results = None

                try:
                    tags = set(extract_tags(caption, config))
                    tags.update(keywords)

                    face_count = count_faces(path, img_array)
                    tags = set(tags)
                    tags = set(enrich_tags(tags, caption, face_count, path))

                    yolo_results = yolo_model(path, verbose=False)[0]
                    object_counts = {}
                    for box in yolo_results.boxes:
                        if box.conf < 0.5:
                            continue
                        tag = yolo_model.names[int(box.cls)]
                        tags.add(tag)
                        object_counts[tag] = object_counts.get(tag, 0) + 1
                    for obj, count in object_counts.items():
                        tags.add(p.plural(obj) if count > 1 else obj)

                    isolated = detect_isolated_subject(path, yolo_model, img, img_array, yolo_results)
                    if isolated:
                        tags.add("isolated")

                    tags = set(clean_tags(tags))
                    
                    if config.get("validate_tags", False):
                        validation_result = validate_tags_with_caption(caption, tags, processor, model, img)
                    if "no" in validation_result or "inaccurate" in validation_result:
                        tags.add("needs_review")
                

                    if config['write_exif']:
                        write_exif(path, caption, sorted(tags))

                except Exception as e:
                    error_lines.append(str(e))

                finally:
                    # Build structured output regardless of success/failure
                    img_shape = img_array.shape
                    yolo_shape = (1, 3, img_shape[0], img_shape[1])
                    detection_count = len(yolo_results) if yolo_results else 0
                    pre_time = f"{round(yolo_results.speed['preprocess'], 1)}ms" if yolo_results and hasattr(yolo_results, 'speed') else "0ms"
                    inf_time = f"{round(yolo_results.speed['inference'], 1)}ms" if yolo_results and hasattr(yolo_results, 'speed') else "0ms"
                    post_time = f"{round(yolo_results.speed['postprocess'], 1)}ms" if yolo_results and hasattr(yolo_results, 'speed') else "0ms"

                    # tone_tag = next((t for t in tags if t in VALID_COLOR_TONES), None)
                    # color_style = COLOR_MAP.get(tone_tag, Style.RESET_ALL) if tone_tag and tone_tag != 'black' else Style.RESET_ALL

                    label_width = 17

                    if config['write_exif'] and caption and tags:
                        try:
                            write_exif(path, caption, sorted(tags))
                            tqdm.write(Fore.GREEN + "✅ Done writing EXIF" + Style.RESET_ALL)
                        except Exception as ex:
                            error_lines.append(f"EXIF write failed: {ex}")
                    
                    tqdm.write("")
                    tqdm.write(f"{Fore.CYAN}{Style.BRIGHT}Now Processing Image {filename}: {img_shape[1]}x{img_shape[0]} ({detection_count} detections), {inf_time}{Style.RESET_ALL}")
                    tqdm.write(f"{Style.DIM}{'Path:'.rjust(label_width)}   {path}")
                    if yolo_results and hasattr(yolo_results, 'speed') and yolo_results.speed:
                        pre_time = f"{round(yolo_results.speed.get('preprocess', 0), 1)}ms"
                        inf_time = f"{round(yolo_results.speed.get('inference', 0), 1)}ms"
                        post_time = f"{round(yolo_results.speed.get('postprocess', 0), 1)}ms"
                    
                        tqdm.write(f"{'YOLO Speed:'.rjust(label_width)}   {pre_time:<8}   preprocess")
                        tqdm.write(f"{' '.rjust(label_width)}   {inf_time:<8}   inference")
                        tqdm.write(f"{' '.rjust(label_width)}   {post_time:<8}   postprocess per image at shape {yolo_shape}")
                    else:
                        tqdm.write(f"{'YOLO Speed:'.rjust(label_width)}   No YOLO objects detected — skipping timing")
                    tqdm.write(f"{Style.BRIGHT}{'Isolated:'.rjust(label_width)}{Style.RESET_ALL}   {'True' if isolated else 'False'}")
                    tqdm.write(f"{Style.BRIGHT}{'ExifTool Output:'.rjust(label_width)}{Style.RESET_ALL}   {'1 image files updated' if config['write_exif'] else 'Skipped'}")
                    tqdm.write(f"{Style.BRIGHT}{'Caption:'.rjust(label_width)}{Style.RESET_ALL}   {Fore.YELLOW}{caption}{Style.RESET_ALL}")
                    tqdm.write(f"{Style.BRIGHT}{'Tags:'.rjust(label_width)}{Style.RESET_ALL}   {Fore.YELLOW}{', '.join(sorted(tags))}{Style.RESET_ALL}")
                    if config.get("validate_tags", False):
                        needs_review = "needs_review" in tags
                        if needs_review:
                            tqdm.write(f"{Style.BRIGHT}{'Tags need review:'.rjust(label_width)}{Style.RESET_ALL}   {Fore.RED}Tags may be inaccurate {validation_result}{Style.RESET_ALL}")
                        else:
                            tqdm.write(f"{Style.BRIGHT}{'Tags need review:'.rjust(label_width)}{Style.RESET_ALL}   {Fore.GREEN}Tags appear valid{Style.RESET_ALL}")
                    if original_mode != "RGB":
                        tqdm.write(f"{Style.BRIGHT}{'Original mode:'.rjust(label_width)}{Style.RESET_ALL}   {original_mode} → RGB")
                    else:
                        tqdm.write(f"{Style.BRIGHT}{'Original mode:'.rjust(label_width)}{Style.RESET_ALL}   RGB")
                    
                    # tqdm.write(f"{Style.BRIGHT}{'Color Analysis'.rjust(label_width)}{Style.RESET_ALL}")
                    # # Pull the tone tag from tags (added during enrich_tags)
                    # tone_tag = next((t for t in tags if t in VALID_COLOR_TONES), None)
                    # 
                    # if tone_tag:
                    #     tqdm.write(f"{'Dominant Tone:'.rjust(label_width)}   {tone_tag}")
                    # else:
                    #     tqdm.write(f"{'Dominant Tone:'.rjust(label_width)}   Unknown or not detected")
                    
                    
                    tqdm.write("")
                    if error_lines:
                        tqdm.write(Fore.RED + Style.BRIGHT + f"{'Errors:'.rjust(label_width)}   {error_lines[0]}" + Style.RESET_ALL)
                        for line in error_lines[1:]:
                            tqdm.write(" " * (label_width + 3) + line)
                    else:
                        tqdm.write(Fore.GREEN + Style.BRIGHT + f"{'Errors:'.rjust(label_width)}   None" + Style.RESET_ALL)
                    tqdm.write(Fore.LIGHTBLACK_EX + "─" * 80 + Style.RESET_ALL)
                    tqdm.write("")                    
                    
                    #color_style = COLOR_MAP.get(tone_tag, Style.RESET_ALL) if tone_tag and tone_tag != 'black' else Style.RESET_ALL

                    
                    # Save log
                    img_duration = time.time() - img_start_time
                    process = psutil.Process(os.getpid())
                    mem_used_mb = process.memory_info().rss / 1024 ** 2
                    gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0

                    folder_log.append(f"Image: {filename}")
                    folder_log.append(f"Folder: {folder_name}")
                    folder_log.append(f"Full Path: {path}")
                    folder_log.append(" ")
                    folder_log.append(f"Caption: {caption}")
                    folder_log.append(f"Tags: {', '.join(sorted(tags))}")
                    if config.get("validate_tags", False):
                        needs_review = "needs_review" in tags
                        folder_log.append(f"Tags need review: {'True' if needs_review else 'False'}")
                    if original_mode != "RGB":
                        folder_log.append(f"Original mode: {original_mode} → RGB")
                    else:
                        folder_log.append("Original mode: RGB")
                    # folder_log.append(f"Dominant Tone: {tone_tag if tone_tag else 'Unknown or not detected'}")

                    
                    
                    
                    folder_log.append(f"Isolated: {'True' if isolated else 'False'}")
                    folder_log.append(f"Image Processing Time: {img_duration:.2f} seconds")
                    folder_log.append(" ")
                    if error_lines:
                        folder_log.append(f"Errors: {error_lines[0]}")
                        for line in error_lines[1:]:
                            folder_log.append(" " * 9 + line)
                    else:
                        folder_log.append("Errors: None")
                    folder_log.append(f"RAM Used: {mem_used_mb:.2f} MB")
                    folder_log.append(f"GPU Memory Used: {gpu_mem:.2f} MB")

                    folder_log.append("-" * 60)
                    if error_lines:
                        error_count += 1
                    image_count += 1



        # ✅ Place this AFTER the full folder is done (outside the chunk loop)
                # Folder summary
        folder_total_time = time.time() - folder_start_time
        minutes, seconds = divmod(int(folder_total_time), 60)
        
        summary_lines = [
            "=== Folder Summary ===",
            f"Images Processed:  {image_count}",
            f"Total Time:        {minutes}m {seconds}s",
            f"Errors Detected:   {error_count}",
            "======================="
        ]
        if folder_log:
            log_path = log_file_path
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(summary_lines + [""] + folder_log))
            tqdm.write(f"📄 Saved log to {log_path}")

def choose_folder():
    """
    GUI file dialog to select a folder for processing.
    """
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Choose a folder to process")
    if not folder:
        messagebox.showinfo("No folder selected", "Operation cancelled.")
        exit()
    return folder

if __name__ == '__main__':
    blip_processor, blip_model = setup_model('flan-t5-xl')  # ✅ Required for tone tag
    config = load_config()
    target_folder = choose_folder()
    process_folder(target_folder, config, blip_processor, blip_model)

    # Clean up memory after run
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("\n🎉 All done! If running again soon, consider restarting your terminal to fully free memory.")
