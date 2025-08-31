"""
explain.py
- gradcam_overlay(frame, model, target_layer)
- object detections via YOLOv8 (Ultralytics) to produce textual rationales
"""

import cv2, numpy as np, os

def gradcam_overlay(model, target_layer, frame, transform, device='cpu', out_path=None):
    # model should be the embedding model used for Grad-CAM (MobileNet backbone wrapper)
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except Exception as e:
        raise ImportError("Install pytorch-grad-cam")
    import torch
    x = transform(frame).unsqueeze(0).to(device)
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device!='cpu'))
    grayscale_cam = cam(input_tensor=x, targets=None)[0]
    vis = show_cam_on_image(frame.astype(np.float32)/255.0, grayscale_cam, use_rgb=True)
    if out_path:
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return vis

def yolo_objects_on_frames(frames, model_name='yolov8n.pt'):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise ImportError("Install ultralytics")
    model = YOLO(model_name)
    results = []
    for img in frames:
        res = model(img, verbose=False)[0]
        labels = [res.names[int(c)] for c in res.boxes.cls.cpu().numpy().tolist()] if res.boxes is not None else []
        results.append(labels)
    return results

def textual_rationale_for_shot(labels, audio_peak=False):
    reasons = []
    if labels:
        reasons.append("Contains: " + ", ".join(set(labels)))
    if audio_peak:
        reasons.append("High audio energy")
    if reasons:
        return "; ".join(reasons)
    return "Selected for representativeness"
