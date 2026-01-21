import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
from PIL import Image

print("[INFO] Loading BLIP-large model for scene description...")
device = "cuda" if torch.cuda.is_available() else "cpu"

midas = None
transform = None


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
print("[INFO] BLIP-large model loaded successfully.")


def get_styled_caption(frame):
   
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.capitalize()


def get_depth_alert(frame):
    
    global midas, transform

    if midas is None:
        print("[INFO] Loading MiDaS (Depth) model for the first time...")
        midas_model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform
        print("[INFO] MiDaS (Depth) model loaded.")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    h, w = depth_map.shape
    cx, cy = w // 2, h // 2
    region_size = int(h * 0.1)
    center_region = depth_map[cy - region_size:cy + region_size,
                              cx - region_size:cx + region_size]

    avg_depth = center_region.mean()
    normalized_depth = (avg_depth - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)


    steps_estimate =2

    if normalized_depth > 0.6:
        alert = f"There is something very close, about {steps_estimate} steps away."
    elif normalized_depth > 0.4:
        alert = f"There is an object roughly {steps_estimate} steps ahead."
    else:
        alert = f"The area in front seems clear, beyond {steps_estimate} steps."
    
    return alert, steps_estimate


def generate_enhanced_caption(frame):
    
    depth_alert, steps = get_depth_alert(frame)
    caption = get_styled_caption(frame)
    return f"{caption}. It appears to be about {steps} steps away. {depth_alert}"
