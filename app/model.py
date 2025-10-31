import torch
import torch.nn as nn
import numpy as np
import cv2
from unet import UNet   # <-- your model class file

import torch
import torch.nn as nn
from unet import UNet  # rename this to match your actual file name

def load_model(model_path, device):
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize model with same config as training
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Load weights (important!)
    model.load_state_dict(checkpoint["model_state"])

    model.eval()
    return model



# ----------------------------
# 2. Preprocess Input Image
# ----------------------------
def preprocess(image):
    # image = numpy array (H, W, 3) BGR from OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB

    image = cv2.resize(image, (512, 512))           # SAME SIZE used during training
    image = image / 255.0                           # normalize to 0-1
    image = np.transpose(image, (2, 0, 1))          # (H,W,3) -> (3,H,W)
    image = np.expand_dims(image, axis=0)           # add batch dimension: (1,3,H,W)

    image = torch.tensor(image, dtype=torch.float32)
    return image

# ----------------------------
# 3. Postprocess Model Output
# ----------------------------
def postprocess(mask):
    mask = torch.sigmoid(mask)                   # ensure values are 0-1
    mask = mask.cpu().detach().numpy()[0, 0]     # (1,1,H,W) -> (H,W)
    mask = (mask > 0.5).astype(np.uint8) * 255   # threshold + convert to image
    return mask

# ----------------------------
# 4. Prediction Function
# ----------------------------
def predict(model, image_path, device):
    image = cv2.imread(image_path)
    input_tensor = preprocess(image).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    mask = postprocess(output)
    return image, mask
