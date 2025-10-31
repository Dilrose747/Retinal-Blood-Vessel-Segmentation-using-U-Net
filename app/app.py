import gradio as gr
import torch
from model import load_model, predict

# -------------------------------------------------
# 1. Device Setup (CPU is fine for deployment)
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# 2. Load Model Once (important for speed)
# -------------------------------------------------
model = load_model("best_unet_model.pth", device)

# -------------------------------------------------
# 3. Inference Wrapper for Gradio
# -------------------------------------------------
def segment_fundus(image):
    # image from gradio comes as a PIL image
    # convert PIL -> save temporary file
    image = image.convert("RGB")
    image.save("temp_input.png")

    original, mask = predict(model, "temp_input.png", device)

    return mask

# -------------------------------------------------
# 4. Build Gradio Interface
# -------------------------------------------------
interface = gr.Interface(
    fn=segment_fundus,
    inputs=gr.Image(type="pil", label="Upload Fundus Image"),
    outputs=gr.Image(type="numpy", label="Segmented Blood Vessel Mask"),
    title="Retinal Vessel Segmentation - U-Net",
    description="Upload a retinal fundus image and get the segmented vessel mask."
)

# -------------------------------------------------
# 5. Launch App
# -------------------------------------------------
if __name__ == "__main__":
    interface.launch()
