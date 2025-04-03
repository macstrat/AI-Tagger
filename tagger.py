# tagger.py
# === Standard Library ===
import os
import sys
import re
import json
import time
import subprocess
import threading
from threading import Thread
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import itertools
from itertools import cycle
import argparse

# === Third-Party Libraries ===
import torch
import yaml
import numpy as np
import colorama
from colorama import Fore, Style, init
init(autoreset=True)
from tqdm import tqdm
from PIL import Image
import face_recognition
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from ultralytics import YOLO
import inflect
p = inflect.engine()

# === NLTK (for POS tagging and lemmatization) ===
import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
lemmatizer = WordNetLemmatizer()
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)



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

VALID_COLOR_TONES = {
    "black", "white", "gray", "red", "orange", "yellow", "green",
    "blue", "purple", "pink", "brown", "beige", "teal", "sepia", "black and white"
}

# Utility Functions 
def ding(): # Plays a notification sound when processing is complete.
    try:
        if os.name == 'nt':  # Windows systems
            import winsound
            winsound.MessageBeep()
        else:  # macOS / Linux
            print('\a', end='', flush=True)  # ASCII bell triggers terminal beep
    except:
        pass

def spinner(msg="Working...", delay=0.1):  # Displays a rotating spinner while waiting
    done_flag = {"done": False}  # Shared flag to stop spinner loop

    def spin():
        for c in itertools.cycle('|/-\\'):  # Infinite rotation sequence
            if done_flag["done"]:  # Exit if told to stop
                break
            try:
                sys.stdout.write(f'\r{msg} {c}')  # Show spinner next to message
                sys.stdout.flush()
                time.sleep(delay)  # Delay between frames
            except KeyboardInterrupt:
                break  # Allow safe exit on Ctrl+C
        sys.stdout.write('\r' + ' ' * (len(msg) + 2) + '\r')  # Clear the line

    t = Thread(target=spin)
    t.daemon = True  # Spinner thread exits when main program ends
    t.start()

    def stop():  # Stop function to end spinner
        done_flag["done"] = True
        t.join()

    return stop  # Return stop function so caller can end the spinner

def chunk_list(data, chunk_size):  # Splits a list into smaller chunks
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]  # Yield sublist from i to i+chunk_size

def clean_utf8(text):  # Cleans a string to ensure valid UTF-8 encoding
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore").strip() # Encode to UTF-8 and decode back, ignoring bad characters, then strip whitespace



# Configuration & Loading
def load_config(config_path="config.yaml"):  # Loads the configuration from config.yaml or user path
    # Default fallback settings
    config = {
        'write_exif': True,       # Write EXIF metadata by default
        'recursive': True,        # Process subfolders by default
        'model_name': 'flan-t5-xl',  # Sets the default model
        'max_tags': 30,           # Limit max number of tags
        'min_tag_length': 3       # Minimum length for each tag word
    }

    if os.path.exists(config_path):  # If config.yaml exists
        print(f"⚠️ Loading user config from: {config_path}")
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)    # Load user-defined overrides
                config.update(user_config)         # Merge into defaults
        except Exception as e:
            raise RuntimeError(f"Error reading config file: {e}")  # Surface load errors

    return config  # Return merged config dictionary


def setup_model(model_name):  # Loads the BLIP2 model and processor
    try:
        if model_name == 'opt-2.7b':
            # Load BLIP2 with OPT 2.7B backend
            processor = Blip2Processor.from_pretrained(
                'Salesforce/blip2-opt-2.7b',
                cache_dir='./models'
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                'Salesforce/blip2-opt-2.7b',
                torch_dtype=torch.float32,  # Avoid half-precision issues
                device_map='auto',
                cache_dir='./models'
            )

        elif model_name == 'flan-t5-xl':
            # Load BLIP2 with FLAN-T5-XL backend
            processor = Blip2Processor.from_pretrained(
                'Salesforce/blip2-flan-t5-xl',
                cache_dir='./models'
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                'Salesforce/blip2-flan-t5-xl',
                torch_dtype=torch.float32,  # Required for many GPUs
                device_map='auto',
                cache_dir='./models'
            )

        else:
            raise ValueError(f"Unsupported model_name: {model_name}")  # Invalid config fallback

        return processor, model  # Return model + processor pair

    except Exception as e:
        raise RuntimeError(f"Model setup error: {e}")  # Surface model loading failure

def setup_yolo(model_path='models/yolov11x.pt'):  # Loads the YOLO object detection model
    return YOLO(model_path)  # Load model from given path using Ultralytics



# Tagging & Captioning
def extract_tags(caption, config):  # Extracts nouns and adjectives from the caption as potential tags
    words = nltk.word_tokenize(caption.lower())  # Tokenize caption into lowercase words
    tagged = nltk.pos_tag(words)  # Part-of-speech tag each word

    # Load any user-defined stopwords (tag fragments) from config
    stopwords_raw = config.get("tag_fragments", "")
    stopwords = set(f.strip().lower() for f in stopwords_raw.split(",") if f.strip())

    tags = set(
        w for w, pos in tagged
        if pos.startswith(('NN', 'JJ'))  # Only keep nouns (NN) and adjectives (JJ)
        and w not in stopwords           # Skip user-defined throwaway words
        and len(w) >= config.get('min_tag_length', 3)  # Enforce minimum word length
    )

    return sorted(tags)[:config.get('max_tags', 15)]  # Return a trimmed, sorted list

def enrich_tags(tags, caption, face_count, image_path):  # Adds context-aware tags to supplement the base list
    enriched = set(tags)
    caption_lower = caption.lower()

    # === FACE/PEOPLE LOGIC ===
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

    # Children-specific fallback (plural vs singular)
    if any(word in caption_lower for word in ["children", "kids"]):
        enriched.add("children")
    elif any(word in caption_lower for word in ["child", "kid"]):
        enriched.add("child")

    # === NOUN COUNTING AND PLURALIZATION ===
    words = word_tokenize(caption)
    tagged = pos_tag(words)

    noun_counts = {}
    for word, pos in tagged:
        if pos.startswith("NN"):  # Only nouns
            singular = lemmatizer.lemmatize(word.lower(), pos='n')  # Normalize to singular
            noun_counts[singular] = noun_counts.get(singular, 0) + 1

    for noun, count in noun_counts.items():
        if count > 1:
            enriched.add(p.plural(noun))  # Use plural if seen multiple times
        else:
            enriched.add(noun)

    # === FILTER GENERIC, NON-VISUAL WORDS UNLESS MATERIAL CONTEXT PRESENT ===
    generic_exclusions = {"background", "surface", "close", "next"}
    material_context = {"texture", "pattern", "tile", "fabric", "marble",
                        "wood", "metal", "grain", "stone", "granite", "abstract"}

    if not enriched & material_context:  # If no material terms, strip generic ones
        enriched -= generic_exclusions

    # === ACTIVITY INFERENCE BASED ON 90s-STOCK COMMON TERMS ===
    activity_keywords = {
        "working", "celebrating", "dancing", "meeting", "laughing", "reading", "drinking", "thinking", "arguing",
        "cooking", "eating", "shopping", "talking", "hugging", "driving", "relaxing", "walking", "running", "jumping",
        "pointing", "teaching", "typing", "writing", "calling", "cleaning", "sleeping", "waking", "watching",
        "listening", "greeting", "presenting", "playing", "gardening", "biking", "swimming", "skating", "clapping",
        "high-fiving", "singing", "posing", "climbing", "traveling", "waiting", "standing", "sitting", "photographing",
        "filming", "repairing", "volunteering", "training", "discussing", "studying", "exercising",
        "painting", "decorating", "caring", "fixing", "cheering", "jogging", "interviewing",
        "counseling", "helping", "consulting", "leading", "organizing", "directing", "collaborating", "shaking hands",
        "planning", "browsing", "flipping", "explaining", "styling", "answering", "stretching", "attending",
        "napping", "parenting", "skateboarding", "snowboarding", "waving", "testing", "conversing",
        "grilling", "jumpstarting", "protesting", "rejoicing", "scrubbing", "examining"
    }

    for word in activity_keywords:
        if word in caption_lower:
            enriched.add(word)

    # === SPECIAL CASES: WORK & FAMILY ===
    if "family" in caption_lower:
        enriched.add("family")

    work_terms = {
        "office", "meeting", "coworker", "colleague", "desk",
        "computer", "presentation", "conference", "business"
    }
    if any(term in caption_lower for term in work_terms):
        enriched.add("work")

    # === CLEANUP: REMOVE REDUNDANT 'person' IF GENDER INFO PRESENT ===
    if 'person' in enriched and ('man' in enriched or 'woman' in enriched):
        enriched.discard('person')

    return sorted(enriched)

    
def validate_tags_with_caption(caption, tags, processor, model, image):  # Uses a language model to check if the tags are relevant to the caption
    tag_str = ", ".join(tags)
    
    # Image validation prompt
    prompt = (
        f"Here is a description of an image: \"{caption}\"\n\n"
        f"The current tags are: [{tag_str}]\n\n"
        "Are these tags correct and relevant for the photo? Answer yes or no."
    )
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)  # Tokenize and send input to model device
    out = model.generate(**inputs)  # Generate model response
    answer = processor.decode(out[0], skip_special_tokens=True).strip().lower()  # Decode and clean the model output
    return answer
    
def generate_batch_captions(image_paths, processor, model, preloaded_images=None):  # Generates one caption and one tag list per image
    images = []
    for i, image_path in enumerate(image_paths):
        if preloaded_images and i < len(preloaded_images):  # Use preloaded image if available
            image = preloaded_images[i]
        else:
            image = Image.open(image_path).convert("RGB")  # Load image from disk and convert to RGB
        images.append(image)

    # Prompt for generating descriptive, stock-style captions
    prompt_caption = config.get("prompt_caption", (
        "Describe this photo like you would for a stock photography database. "
        "Be specific, and limit your response to one sentence. "
        "Focus on visible objects, people, setting, and composition. Avoid abstract terms."
    ))

    # Prepare input batch for caption generation
    inputs = processor(images=images, text=[prompt_caption]*len(images), return_tensors='pt', padding=True)
    for k, v in inputs.items():
        if v.dtype in [torch.float32, torch.float64, torch.float16]:
            inputs[k] = v.to(model.device, dtype=torch.float32)  # Send tensors to model device (GPU/CPU)
        else:
            inputs[k] = v.to(model.device)

    with torch.no_grad():
        caption_ids = model.generate(**inputs, max_new_tokens=60)  # Generate caption token IDs
    captions = processor.batch_decode(caption_ids, skip_special_tokens=True)  # Decode token IDs into strings

    # === Now tag generation ===

    # Prompt for tag generation based on caption + optional folder context
    prompt_tag = config.get("prompt_tag", (
        "You are an AI describing stock photography. Generate a comma-separated list of 8–12 short, relevant tags based only on visible content. "
        "Do NOT include file names, folder names, numbers, or abstract/emotional words. "
        "Only tag what is clearly seen in the photo — such as objects, people, settings, and actions."
    ))

    prompts = []
    for img_path, cap in zip(image_paths, captions):
        folder_path = os.path.dirname(img_path)  # Get folder from image path
        raw_folder = os.path.basename(folder_path)
        context = "" if is_junk_folder_name(raw_folder) else extract_folder_context(folder_path)  # Get folder context unless junk
        
        if context:
            full_prompt = f"{prompt_tag} Caption: \"{cap.strip()}\". Context: \"{context}\""  # Include folder context if available
        else:
            full_prompt = f"{prompt_tag} Caption: \"{cap.strip()}\""  # No context fallback
        
        prompts.append(full_prompt)

    # Prepare input batch for tag generation
    tag_inputs = processor(images=images, text=prompts, return_tensors='pt', padding=True)
    for k, v in tag_inputs.items():
        if v.dtype in [torch.float32, torch.float64, torch.float16]:
            tag_inputs[k] = v.to(model.device, dtype=torch.float32)
        else:
            tag_inputs[k] = v.to(model.device)

    with torch.no_grad():
        tag_ids = model.generate(**tag_inputs, max_new_tokens=30)  # Generate tag token IDs
    tag_outputs = processor.batch_decode(tag_ids, skip_special_tokens=True)  # Decode tag strings

    # === Post-process output ===
    result = []
    for cap, tag_output in zip(captions, tag_outputs):
        cap = cap.strip()  # Clean up caption

        tag_output = tag_output.strip().strip('"').strip("'")  # Remove stray quotes

        keywords = [kw.strip() for kw in tag_output.split(",") if kw.strip()]  # Split into individual tags

        keywords = [kw for kw in keywords if len(kw.split()) <= 3]  # Remove multi-word phrases

        keywords = [k for k in keywords if k.lower() not in {  # Remove known bad outputs
            "description", "descriptions", "detailed", "detail", "jpg", "jpeg", "image", "photo", "picture", "pictures"
        }]

        result.append((cap, keywords))  # Add cleaned caption and tags
    return result  # Final list of (caption, tags)



# Tag Cleaning / Processing

def clean_tags(tags, config=None):  # Cleans up tag list by removing fragments, junk, and long phrases
    cleaned = set()
    fragments = set(config.get("tag_fragments", "a, an, the, in, on, of, and, with, to, at, for").split(",")) # Common throwaway words
    min_length = config.get("min_tag_length", 2) if config else 2

    for tag in tags:
        tag = tag.strip().lower()  # Normalize whitespace and casing

        if tag in fragments:  # Skip small linking words
            continue

        if len(tag) < min_length:  # Skip too-short tags
            continue

        if len(tag.split()) > 3:  # Skip tags that are likely full sentences
            continue

        cleaned.add(tag)  # Keep tag if it passed all checks

    return sorted(cleaned)  # Return cleaned, sorted list of tags

def choose_folder():  # Opens a GUI dialog to select a folder path
    root = tk.Tk()  # Create a hidden root window
    root.withdraw()  # Hide the main tkinter window
    folder = filedialog.askdirectory(title="Choose a folder to process")  # Open folder picker dialog

    if not folder:  # If user cancels or closes the dialog
        messagebox.showinfo("No folder selected", "Operation cancelled.")  # Show alert
        exit()  # Exit the script immediately

    return folder  # Return the selected folder path



# Folder Context Inference

def extract_folder_context(folder_path):  # Extracts a cleaned context string from the folder name
    folder_name = os.path.basename(folder_path)  # Get the last part of the path (the folder name)

    cleaned = re.sub(r"[-_]", " ", folder_name).lower()  # Replace dashes/underscores with spaces and lowercase it

    words = cleaned.split()  # Split folder name into words

    # Keep words that contain letters and are longer than 2 characters
    meaningful = [w for w in words if any(c.isalpha() for c in w) and len(w) > 2]

    context = " ".join(meaningful).strip()  # Rejoin into a cleaned phrase

    # Return context unless it's junk or empty
    return context if context and not is_junk_folder_name(context) else ""

def is_junk_folder_name(name):  # Determines if a folder name is too short or meaningless for context
    name = name.lower()  # Normalize to lowercase

    if len(name) <= 3:  # Too short to be useful
        return True

    if re.fullmatch(r'[a-z0-9_\-]+', name):  # Only contains letters, digits, underscores or hyphens (likely meaningless)
        return True

    return False  # Name looks meaningful enough to use
    


# Image Analysis

def count_faces(image_path, img_array=None):  # Detects and counts the number of faces in an image
    if img_array is None:
        img_array = np.array(Image.open(image_path).convert("RGB"))  # Load and convert image if not pre-supplied

    try:
        img = Image.open(image_path)  # Open image from path

        if img.mode not in ['RGB', 'L']:  # Convert unusual modes to RGB
            img = img.convert('RGB')
        
        max_size = 800  # Resize image to speed up face detection
        img.thumbnail((max_size, max_size))
        img_array = np.array(img)

        if img_array.ndim == 2:
            img_array = np.stack((img_array,)*3, axis=-1)  # Convert grayscale to RGB
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            img_array = np.repeat(img_array, 3, axis=2)  # Convert single-channel to RGB
        elif img_array.ndim != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Invalid shape: {img_array.shape}")  # Ensure it's proper RGB

        face_locations = face_recognition.face_locations(img_array)  # Detect face bounding boxes
        return len(face_locations)  # Return number of detected faces

    except Exception as e:
        raise RuntimeError(f"Face detection error in {image_path}: {e}")  # Surface error clearly
        return 0  # (Unreachable due to raise, but safe fallback)

def detect_isolated_subject(image_path, yolo_model, img=None, img_array=None, yolo_results=None, config=None):
    if img is None:
        img = Image.open(image_path).convert("RGB")  # Load image if not preloaded
    if img_array is None:
        img_array = np.array(img)  # Convert image to numpy array
    results = yolo_results

    """
    Returns True if the subject is fully visible (not cropped), not a person,
    and the background is mostly white or black.
    """

    try:
        from PIL import Image
        import numpy as np
        import face_recognition

        width, height = img.size  # Get image dimensions
        bg_thresh = config.get("isolated_bg_ratio", 0.4) if config else 0.4

        # === 1. Check for faces (exclude people)
        face_locations = face_recognition.face_locations(img_array)
        if len(face_locations) > 0:
            return False  # If any face is detected, reject as not isolated

        # === 2. Check for mostly white or black background
        flat_pixels = img_array.reshape(-1, 3)  # Flatten to 2D list of RGB pixels
        white_pixels = np.sum(np.all(flat_pixels >= 240, axis=1))  # Count nearly white pixels
        black_pixels = np.sum(np.all(flat_pixels <= 15, axis=1))   # Count nearly black pixels
        total_pixels = flat_pixels.shape[0]

        white_ratio = white_pixels / total_pixels
        black_ratio = black_pixels / total_pixels

        if white_ratio >= bg_thresh or black_ratio >= bg_thresh:
            return True  # Accept if background is at least 40% white or black

        # === 3. Fallback to contour detection if background is mixed
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)  # Invert bright background
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find external shapes
        if not contours:
            return False  # No shape found — not isolated

        largest = max(contours, key=cv2.contourArea)  # Get largest contour
        x, y, w, h = cv2.boundingRect(largest)  # Get bounding box
        box_area = w * h

        if box_area < 0.6 * width * height:
            return False  # Reject if bounding box is too small

        # === 4. Check that object is not cropped at edges
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        border_thresh = 0.03  # 3% margin from edges
        if (
            x1 < border_thresh * width or
            y1 < border_thresh * height or
            x2 > (1 - border_thresh) * width or
            y2 > (1 - border_thresh) * height
        ):
            return False  # Reject if object touches any image edge

        return True  # Passes all checks → considered isolated

    except Exception as e:
        raise RuntimeError(f"[Isolated Detection Error] {image_path}: {e}")  # Surface error clearly
        return False  # (Unreachable but safe fallback)



# Metadata Handling (EXIF)
    
def read_existing_keywords(image_path):  # Reads existing EXIF keywords from an image file
    try:
        exiftool_exe = r'C:\exiftool\exiftool.exe'  # Path to ExifTool executable

        result = subprocess.run(
            [exiftool_exe, "-Keywords", "-XMP-dc:Subject", "-s3", image_path],  # Read keywords and XMP subject tags
            capture_output=True, text=True
        )

        lines = result.stdout.strip().splitlines()  # Split output into lines
        keywords = set()

        for line in lines:
            if line:
                for kw in line.split(","):  # Handle comma-separated entries
                    cleaned = clean_utf8(kw.strip().lower())  # Normalize keyword
                    if cleaned:
                        keywords.add(cleaned)

        return sorted(keywords)  # Return sorted list of unique keywords

    except Exception as e:
        return []  # Return empty list on failure

def write_exif(image_path, caption, tags):  # Writes a caption and tags to image metadata using ExifTool
    try:
        exiftool_exe = r'C:\exiftool\exiftool.exe'  # Path to ExifTool executable

        # Step 1: Read existing keywords from image
        existing_tags = read_existing_keywords(image_path)
        merged_tags = sorted(set(existing_tags) | set(tags))  # Combine new and existing tags (remove duplicates)

        # Step 2: Strip existing Photoshop metadata blocks to avoid conflicts
        subprocess.run(
            [exiftool_exe, "-Photoshop:All=", "-overwrite_original", image_path],
            capture_output=True, text=True
        )

        # Step 3: Build ExifTool write command with caption and tags
        cmd = [
            exiftool_exe,
            '-charset', 'utf8',  # Ensure correct encoding
            '-overwrite_original',
            f'-IPTC:Caption-Abstract={clean_utf8(caption)}',
            f'-XMP-dc:Title={clean_utf8(caption)}',
            f'-XMP-dc:Description={clean_utf8(caption)}',
        ]

        # Add each tag as both XMP subject and IPTC keyword
        for tag in sorted(merged_tags):
            tag = clean_utf8(tag)
            cmd.append(f'-XMP-dc:Subject={tag}')
            cmd.append(f'-IPTC:Keywords={tag}')

        cmd.append(image_path)  # Append image path at the end of the command

        result = subprocess.run(cmd, capture_output=True, text=True)  # Run command
        if result.stderr:
            raise RuntimeError(f"ExifTool Error ({image_path}): {result.stderr.strip()}")  # Handle metadata errors

    except Exception as e:
        raise RuntimeError(f"Error writing EXIF for {image_path}: {e}")  # Catch and re-raise any errors



# Core Processing

def process_folder(root_folder, config, processor, model, yolo_model):  # Processes all image files in a folder, chunked with logging and tagging
    # === File extensions to scan for
    supported_exts = [ext.strip().lower() for ext in config.get("supported_exts", ".jpg,.jpeg,.png,.tif,.tiff").split(",") if ext.strip()]  # File types to process
    print("Supported extensions:", supported_exts)
    chunk_size = config.get("chunk_size", 20)  # How many images to process at once (batch size)
    thumbnail_size = config.get("thumbnail_size", 1000)


    if config.get("recursive", True): 
        walker = os.walk(root_folder)
    else:
        walker = [(root_folder, [], os.listdir(root_folder))]
        # print("🔧 Recursive mode:", config.get("recursive", True))    # Debug output
        # print("🗂️ Target folder:", root_folder)    # Debug output

    
    for dirpath, _, filenames in walker:  # Walk through all folders
        # print(f"🔍 Walking into folder: {dirpath}")    # Debug output
        # print(f"📁 Files found: {filenames}")    # Debug output
        log_filename = config.get("log_file_name", "tagger_log.txt")  # Custom log file name
        log_file_path = os.path.join(dirpath, log_filename)

        if os.path.exists(log_file_path):  # Skip already-processed folders
            tqdm.write(f"✅ {os.path.basename(dirpath)} has been processed and has a log file")
            continue

        folder_start_time = time.time()
        error_count = 0
        image_count = 0
        ram_usage_list = []
        gpu_usage_list = []
        total_ram_used = 0.0
        total_gpu_used = 0.0

        # Filter images by extension and skip '!.jpg'
        image_files = [
            f for f in filenames
            if Path(f).suffix.lower() in supported_exts
            and f.lower() != '!.jpg'
        ]
        # print(f"Found {len(image_files)} supported images in: {dirpath}")    # Debug output
        if not image_files:
            continue

        image_paths = [os.path.join(dirpath, f) for f in image_files]
        folder_log = []  # To be written to tagger_log.txt
        tqdm.write(" ")
        tqdm.write(f"📂 Found {len(image_paths)} image(s) in folder: {dirpath}")

        for chunk_idx, chunk_paths in enumerate(chunk_list(image_paths, chunk_size)):  # Process in batches
            chunk_images = []        # Stores resized image objects
            chunk_arrays = []        # Stores numpy arrays of resized images
            original_modes = []      # Stores original image modes (e.g., L, CMYK) before conversion to RGB
            

            tqdm.write(f"\n🔄 Processing Chunk {chunk_idx+1} ({len(chunk_paths)} images)")
            import psutil
            process = psutil.Process(os.getpid())
            mem_used_mb = process.memory_info().rss / 1024 ** 2  # Show RAM usage
            tqdm.write(f"🧠 Estimated RAM used: {mem_used_mb:.2f} MB")
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2  # Show GPU usage
                tqdm.write(f"🖥️ Estimated GPU memory used: {gpu_mem:.2f} MB")

            for path in chunk_paths:
                img = Image.open(path)
                original_mode = img.mode
                img = img.convert("RGB")
                chunk_images.append(img)
                original_modes.append(original_mode)
                chunk_arrays.append(np.array(img))

            # Run caption and tag generation with spinner
            stop_spinner = spinner("🕒 Generating captions and tags... (this might take a while, go get some tea)")
            chunk_captions_tags = generate_batch_captions(chunk_paths, processor, model, preloaded_images=chunk_images)
            stop_spinner()

            for idx, path in enumerate(chunk_paths):  # Handle each image result
                tqdm.write(f"\n⏳ Starting image: {os.path.basename(path)}")
                img_start_time = time.time()
                filename = os.path.basename(path)
                folder_name = os.path.basename(os.path.dirname(path))

                img = chunk_images[idx]
                original_mode = original_modes[idx]
                img.thumbnail((thumbnail_size, thumbnail_size), Image.LANCZOS)

                img_array = np.array(img)

                chunk_images[idx] = img
                chunk_arrays[idx] = img_array

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

                    yolo_results = yolo_model(path, verbose=False)[0]  # YOLO is only used for isolation

                    isolated = detect_isolated_subject(path, yolo_model, img, img_array, yolo_results, config)
                    if isolated:
                        tags.add("isolated")

                    tags = set(clean_tags(tags, config))
                    max_tags = config.get("max_tags", 30)
                    tags = sorted(tags)[:max_tags]

                    if config.get("validate_tags", False):
                        validation_result = validate_tags_with_caption(caption, tags, processor, model, img)
                    if "no" in validation_result or "inaccurate" in validation_result:
                        tags.add("needs_review")

                except Exception as e:
                    error_lines.append(str(e))

                finally:
                    # ========== STRUCTURED OUTPUT + LOGGING ========== #
                    img_shape = img_array.shape
                    yolo_shape = (1, 3, img_shape[0], img_shape[1])
                    detection_count = len(yolo_results) if yolo_results else 0
                    pre_time = f"{round(yolo_results.speed['preprocess'], 1)}ms" if yolo_results and hasattr(yolo_results, 'speed') else "0ms"
                    inf_time = f"{round(yolo_results.speed['inference'], 1)}ms" if yolo_results and hasattr(yolo_results, 'speed') else "0ms"
                    post_time = f"{round(yolo_results.speed['postprocess'], 1)}ms" if yolo_results and hasattr(yolo_results, 'speed') else "0ms"

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
                        tqdm.write(f"{Style.BRIGHT}{'Original mode:'.rjust(label_width)}{Style.RESET_ALL}   {'Greyscale' if original_mode == 'L' else original_mode} → RGB")
                    else:
                        tqdm.write(f"{Style.BRIGHT}{'Original mode:'.rjust(label_width)}{Style.RESET_ALL}   RGB")

                    tqdm.write("")
                    if error_lines:
                        tqdm.write(Fore.RED + Style.BRIGHT + f"{'Errors:'.rjust(label_width)}   {error_lines[0]}" + Style.RESET_ALL)
                        for line in error_lines[1:]:
                            tqdm.write(" " * (label_width + 3) + line)
                    else:
                        tqdm.write(Fore.GREEN + Style.BRIGHT + f"{'Errors:'.rjust(label_width)}   None" + Style.RESET_ALL)
                    tqdm.write(Fore.LIGHTBLACK_EX + "─" * 80 + Style.RESET_ALL)
                    tqdm.write("")

                    # === LOG FILE WRITING ===
                    img_duration = time.time() - img_start_time
                    process = psutil.Process(os.getpid())
                    mem_used_mb = process.memory_info().rss / 1024 ** 2
                    gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2
                    ram_usage_list.append(mem_used_mb)
                    gpu_usage_list.append(gpu_mem)
                    
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
                        folder_log.append(f"Original mode: {'Greyscale' if original_mode == 'L' else original_mode} → RGB")
                    else:
                        folder_log.append("Original mode: RGB")
                    folder_log.append(f"Isolated: {'True' if isolated else 'False'}")
                    folder_log.append(f"Image Processing Time: {img_duration:.2f} seconds")
                    folder_log.append(" ")
                    if error_lines:
                        folder_log.append(f"Errors: {error_lines[0]}")
                        for line in error_lines[1:]:
                            folder_log.append(" " * 9 + line)
                    else:
                        folder_log.append("Errors: None")
                    folder_log.append("-" * 60)

                    if error_lines:
                        error_count += 1
                    image_count += 1

        # === FINAL SUMMARY FOR FOLDER ===
        folder_total_time = time.time() - folder_start_time
        minutes, seconds = divmod(int(folder_total_time), 60)
        avg_ram_used = sum(ram_usage_list) / len(ram_usage_list) if ram_usage_list else 0
        avg_gpu_used = sum(gpu_usage_list) / len(gpu_usage_list) if gpu_usage_list else 0


        summary_lines = [
            "=== Folder Summary ===",
            f"Images Processed:  {image_count}",
            f"Total Time:        {minutes}m {seconds}s",
            f"Errors Detected:   {error_count}",
            f"Avg RAM Used:      {avg_ram_used:.2f} MB",
            f"Avg GPU Used:      {avg_gpu_used:.2f} MB",
            "======================="
        ]
        if folder_log:
            log_path = log_file_path
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(summary_lines + [""] + folder_log))
            tqdm.write(f"📄 Saved log to {log_path}")
            tqdm.write(f"[✔] Finished folder: {os.path.basename(dirpath)}")
            ding()



# Entry Point

if __name__ == '__main__':  # Entry point: only runs when script is executed directly
    import argparse

    # === Step 1: Parse optional command-line arguments
    parser = argparse.ArgumentParser(description="Run AI tagging pipeline")
    parser.add_argument('--folder', type=str, help='Target folder to process (optional)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file (default: config.yaml)')
    args = parser.parse_args()

    # === Step 2: Load configuration from YAML
    config = load_config(args.config)  # Load user-defined settings from YAML

    # === Step 3: Load AI models
    blip_processor, blip_model = setup_model(config["model_name"])  # Load captioning + tag refinement model
    yolo_model = setup_yolo(config["yolo_model_path"])  # Load YOLO object detection model

    # === Step 4: Determine folder (CLI or fallback to GUI)
    if args.folder:
        target_folder = args.folder  # Use folder from command-line argument
    else:
        target_folder = choose_folder()  # Prompt user to select folder via GUI

    # === Step 5: Run main image tagging pipeline
    process_folder(target_folder, config, blip_processor, blip_model, yolo_model)

    # === Step 6: Clean up memory and GPU cache
    import gc
    import torch

    gc.collect()  # Trigger Python garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Free unused GPU memory
        torch.cuda.ipc_collect()  # Clean up inter-process comms (for stability)

    print("\n🎉 All done! If running again soon, consider restarting your terminal to fully free memory.")  # Friendly reminder
