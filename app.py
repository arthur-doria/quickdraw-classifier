import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = [
    "apple", "banana", "cat", "car", "clock",
    "fish", "house", "star", "tree", "t-shirt"
]
MODEL_PATH = "quickdraw_cnn.pth"

class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    model = QuickDrawCNN(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model

model = load_model()

def preprocess_canvas(img_arr):
    if img_arr.shape[-1] == 4:
        mask = (img_arr[:, :, 3] > 50) & (img_arr[:, :, 0] > 50)
    else:
        mask = img_arr[:, :, 0] > 50

    if not np.any(mask):
        return None

    mask_uint8 = mask.astype(np.uint8) * 255
    rows = np.any(mask_uint8 > 0, axis=1)
    cols = np.any(mask_uint8 > 0, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = mask_uint8[rmin:rmax+1, cmin:cmax+1]

    cropped_img = Image.fromarray(cropped).filter(ImageFilter.MaxFilter(size=3))

    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cropped_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    centered = np.zeros((28, 28), dtype=np.float32)
    pad_top = (28 - new_h) // 2
    pad_left = (28 - new_w) // 2
    centered[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = np.array(resized)

    return centered

st.set_page_config(page_title="QuickDraw Classifier", layout="wide")
st.title("QuickDraw Sketch Classifier")
st.markdown("### Possible Objects :")
st.markdown("## **apple — banana — cat — car — clock — fish — house — star — tree — t-shirt**")

if "reset_count" not in st.session_state:
    st.session_state["reset_count"] = 0

col_canvas, col_pred = st.columns([1, 1])

with col_canvas:
    st.write("Draw :")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=380,
        width=380,
        drawing_mode="freedraw",
        return_image_data=True,
        update_streamlit=True,
        key=f"canvas_{st.session_state['reset_count']}",
    )

    if st.button("Clear", use_container_width=True):
        st.session_state["reset_count"] += 1
        st.rerun()

with col_pred:
    st.write("Answer :")
    if canvas_result is not None and canvas_result.image_data is not None:
        processed = preprocess_canvas(canvas_result.image_data)
        if processed is not None:
            tensor = torch.tensor(processed / 255.0, dtype=torch.float32).view(1, 1, 28, 28)
            with torch.no_grad():
                probs = F.softmax(model(tensor), dim=1).squeeze().numpy()

            top_indices = np.argsort(probs)[::-1][:3]
            for idx in top_indices:
                prob = float(probs[idx])
                st.write(f"**{CLASSES[idx]}** ({prob * 100:.1f} %)")
                st.progress(prob)
        else:
            st.info("Draw an object")
