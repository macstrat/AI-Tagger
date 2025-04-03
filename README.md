```
___  ___              _            ___   _____     _____                                  
|  \/  |             ( )          / _ \ |_   _|   |_   _|                                 
| .  . |  __ _   ___ |/  ___     / /_\ \  | |       | |    __ _   __ _   __ _   ___  _ __ 
| |\/| | / _` | / __|   / __|    |  _  |  | |       | |   / _` | / _` | / _` | / _ \| '__|
| |  | || (_| || (__    \__ \    | | | | _| |_      | |  | (_| || (_| || (_| ||  __/| |   
\_|  |_/ \__,_| \___|   |___/    \_| |_/ \___/      \_/   \__,_| \__, | \__, | \___||_|   
                                                                  __/ |  __/ |            
                                                                 |___/  |___/                 


AI Image Tagger v1.2 - Caption, Tag, and Metadata Your Images
```

# AI Image Tagger (v1.2)

> A powerful Python-based tool for auto-captioning and keyword-tagging image folders using BLIP2 and YOLOv8 — with metadata writing, isolation detection, and contextual enrichment.

---

## Command Line Usage

If just the arguements are run with the script, a window to select teh root folder will appear.
To run the AI Image Tagger from the command line, use the following syntax:

```bash
python tagger_1.3_cleaned.py --folder "D:\AI Image Tagger\Images\Sample Set" --config "config.yaml"
```

### Notes:
- Ensure the folder path does **not** have a trailing backslash (`\`).
  
---

## Features

- **Caption Generation** (BLIP2 Flan-T5-XL)
- **Object Detection** (YOLOv11x)
- **Face Detection** (via `face_recognition`)
- **Enriched Tagging** with gender, pluralization, activities, and folder context
- **Isolation Detection** for clean product-style shots
- **EXIF Metadata Writing** (via ExifTool)
- **Auto-logging per folder** to prevent reprocessing

---

## Requirements

- Python 3.12.2
- GPU recommended for performance (BLIP2 + YOLO)
- OS: Windows, macOS, or Linux

---

## Quick Start

### System Requirements
- **Recommended GPU:** NVIDIA RTX 3060 or better (8GB+ VRAM)
- **Minimum RAM:** 16GB (32GB preferred for large batches)
- **Disk Space:** ~15GB after setup (models + dependencies)
- **OS Compatibility:** Windows 10+, macOS 11+, Ubuntu 20.04+

### Important Notes
- **ExifTool** must be installed manually:
  - [Download ExifTool](https://exiftool.org/)
  - Extract it and set the path in `config.yaml` under `exiftool_path`
  - Make sure to rename it to "exiftool.exe" (usually remove the "(k)" from the file name)
- **face_recognition** depends on `dlib`, which requires extra setup. this is NOT optional:
  - **Windows:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), check C++ workload ONLY during install (dont install anything else)
  - **macOS/Linux:** Install `cmake`, `boost`, and ensure `brew install dlib` or build from source

### Mac/Linux Adjustments
- Update `exiftool_path` in `config.yaml`:
  ```yaml
  exiftool_path: /usr/local/bin/exiftool  # or wherever it's installed
  ```
- Ensure `pip` installs use a virtual environment and dependencies are resolved for your architecture (ARM/M1 support tested on Python 3.11+)

### 1. Clone the Repo
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Run the Setup Script (Auto Installs Everything)
```bash
python setup_tagger.py
```
This will:
- Install dependencies (`requirements.txt`)
- Download BLIP2 (Flan-T5-XL)
- Download YOLOv11x weights

### 3. Place Your Images
Put your `.jpg`, `.png`, `.tif`, or `.tiff` images inside a folder (or nested folders).

### 4. Run the Tagger
```bash
python tagger_1.2.py
```
You will be prompted to select a folder. The tool will recursively process all subfolders unless configured otherwise.

---

## What It Does

### Per Image:
- Generates a caption using BLIP2
- Extracts tags using noun/adjective analysis, BLIP2 prompts, and YOLO object detection
- Enriches tags with:
  - Gender / group detection (via face count)
  - Activity keyword matching
  - Folder name context (soft influence)
- Detects "isolated" objects if:
  - Background is mostly white/black (>= 40%)
  - Subject is fully visible and not a person
- Validates tags (optional BLIP2 check)
- Writes tags and caption to EXIF metadata
- Outputs a formatted terminal summary

---

## Configuration (`config.yaml`)

| Setting              | Description |
|---------------------|-------------|
| `exiftool_path`     | Path to ExifTool (required for EXIF writing) |
| `write_exif`        | Write tags/captions to image metadata |
| `recursive`         | Process subfolders automatically |
| `validate_tags`     | Uses BLIP2 to check tag relevance |
| `model_name`        | `flan-t5-xl` or `opt-2.7b` |
| `max_tags`          | Max number of tags per image |
| `min_tag_length`    | Discard tags shorter than this length |
| `chunk_size`        | Controls how many images are processed per batch |
| `thumbnail_size`    | Resize images before tagging |
| `supported_exts`    | Extensions allowed for processing |
| `prompt_caption`    | Prompt used for BLIP2 caption generation |
| `prompt_tag`        | Prompt used for BLIP2 tag generation |

---

## Advanced Features

### Isolation Tagging
- Detects if a **non-human object** is fully visible with a mostly white or black background
- Adds tag: `isolated`

### Dominant Color (Planned)
- Future update: Will label color tones (`monochrome`, `sepia`, `duotone`, etc.)

### Tag Validation
- BLIP2 is asked to confirm if tags match the caption
- If not, adds tag: `needs_review`

### Metadata Writing
- `XMP:Title`, `XMP:Description`, and `IPTC:Caption` written from caption
- `IPTC:Keywords` and `XMP:Subject` populated with tags

---

## Log File (`tagger_log.txt`)
For each processed folder, a log is written containing:
- Image name, path, caption
- Final tags (sorted)
- Isolation status
- Original image mode (RGB, CMYK, etc.)
- Tag validation result
- RAM/GPU usage
- Errors (if any)

This prevents reprocessing the same folder unless the log is removed.

---

## Performance Tips

| Tip | Why |
|-----|-----|
| Lower `chunk_size` | Reduces memory pressure on large batches |
| Resize large images | Speeds up face/YOLO/BLIP2 processing |
| Use a GPU | BLIP2 is slow on CPU |
| Set `write_exif: false` | Speeds up runs if metadata not needed |
| Increase thumbnail size | Improves caption quality (at cost of speed) |

---

## Troubleshooting

- "face_recognition not found": This depends on `dlib`, which may require Visual Studio Build Tools (Windows) or `cmake` + `boost` (macOS/Linux). On Windows:
  - Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Make sure you check the C++ build tools option
- "EXIFTool path not found": Check the `exiftool_path` in `config.yaml`
- Tagging seems off?
  - Try enabling `validate_tags: true`
  - Check the terminal summary and log for issues

- "face_recognition not found": Ensure dlib is properly installed (can be tricky on some platforms)
- "EXIFTool path not found": Check the `exiftool_path` in `config.yaml`
- Tagging seems off?
  - Try enabling `validate_tags: true`
  - Check the terminal summary and log for issues

---

## Folder Context Biasing
The folder name (after the dash `-`) is used to softly bias tag prompts.
For example, a folder named:
```
Hemera 07 - Medical Technology
```
Will bias tags toward "medical" concepts if relevant.

---

## File & Folder Rules
- Supported: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`
- EXIF-writing requires ExifTool installed
- Log file is named `tagger_log.txt`
- Re-running will **skip already-logged folders**

---

## Future Roadmap
- Color-based `monochrome`, `duotone`, `sepia` detection
- Improved caption prompt tuning
- More intelligent tag cleanup
- Optional export to CSV/JSON metadata

---

LICENSE

This project is licensed under a custom non-commercial license.
- Free for personal, academic, or non-commercial research use.
- Commercial use, redistribution, or integration into paid products or services is prohibited without permission.
- You must comply with the licenses of any third-party tools or models used (e.g., YOLOv8, ExifTool, BLIP2).

To obtain a commercial license, please open an issue or reach out via GitHub: https://github.com/macstrat/AI-Tagger


---

## Credits
- Salesforce BLIP2 (https://huggingface.co/Salesforce)
- Ultralytics YOLOv8 (https://github.com/ultralytics/ultralytics)
- ExifTool (https://exiftool.org)

---

> Built by Macstrat. Optimized for speed, transparency, and real-world image libraries.

