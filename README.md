# Video Summarization Project (Modular)
This folder contains modular scripts for a publishable, lightweight video summarization pipeline.
Designed to run on Kaggle with a P100 GPU available.

Structure:
- preprocess.py : frame sampling, shot detection, audio features
- embed.py      : MobileNetV2 / CLIP feature extraction and SimCLR-lite per-video refine
- select.py     : multimodal scoring, KMeans baseline, MMR selection, knapsack wrapper
- explain.py    : Grad-CAM overlays and YOLOv8 object detection rationales
- eval.py       : metrics (F-score, diversity, Kendall tau) and experiment runner
- demo_notebook.ipynb : Jupyter demo notebook to run the pipeline on a single video

Requirements (install on Kaggle):
```
pip install opencv-python-headless librosa torch torchvision pytorch-grad-cam ultralytics facenet-pytorch scenedetect scikit-learn tqdm seaborn datasets --quiet
```

Notes:
- Use P100 to enable CLIP if desired, and to speed up SimCLR-lite refinement and YOLO inference.
- The scripts are modular: call them from the demo notebook or import functions directly for experiments.
