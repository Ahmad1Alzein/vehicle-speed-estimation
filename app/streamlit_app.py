# app/streamlit_app.py
"""
Streamlit UI for:
- Upload video
- Select 2 lines (4 points) on the first frame (click-to-pick OR manual fallback)
- Edit per-class max speed rules (table) WITHOUT resetting
- Run YOLO + DeepSORT + SpeedEstimator pipeline
- Display + download processed video and CSV

Fixes included:
- Line picker image is bigger by upscaling (works with older streamlit-image-coordinates)
- After each click, st.rerun() so P1 appears immediately
- Rules table does NOT reset while editing:
    - uses st.session_state as the single source of truth
    - uses a stable key for st.data_editor
    - optionally uses a form + Apply button (enabled below)
- Processed video display works reliably:
    - st.video(Path(file).read_bytes()) instead of st.video(path)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# --- ensure repo root is on sys.path so "import src...." works ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.speed.geometry import Line
from src.tracking.DeepSORT import process_video

Point = Tuple[int, int]


# -------------------------
# Helpers
# -------------------------
def save_upload_to_temp(uploaded_file) -> str:
    """Save uploaded file to a temp path and return the path."""
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def read_first_frame(video_path: str) -> np.ndarray:
    """Read the first frame of a video."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to read first frame from video.")
    return frame


def draw_overlay(frame_bgr: np.ndarray, pts: List[Point]) -> np.ndarray:
    """Draw points and line previews on a copy of the frame."""
    vis = frame_bgr.copy()

    # Draw clicked points
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), 6, (255, 255, 255), -1)
        cv2.putText(
            vis,
            f"P{i+1}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # Line 1 preview
    if len(pts) >= 2:
        cv2.line(vis, pts[0], pts[1], (0, 255, 255), 3)

    # Line 2 preview
    if len(pts) >= 4:
        cv2.line(vis, pts[2], pts[3], (255, 0, 255), 3)

    return vis


@st.cache_resource
def load_yolo_classnames(model_path: str) -> List[str]:
    """Load YOLO model once to get class names (cached)."""
    model = YOLO(model_path)
    return list(model.names.values())


def build_default_rules(class_names: List[str]) -> pd.DataFrame:
    """
    Create a default rules table (editable by user).
    Adjust defaults as you like.
    """
    defaults = []
    for c in class_names:
        lc = c.lower()
        if "car" in lc:
            defaults.append(80.0)
        elif "truck" in lc:
            defaults.append(50.0)
        elif "bus" in lc:
            defaults.append(60.0)
        else:
            defaults.append(60.0)

    return pd.DataFrame({"class": class_names, "max_kmh": defaults})


def df_to_rules_dict(df: pd.DataFrame) -> Dict[str, float]:
    """Convert the rules editor dataframe to a dict."""
    return {str(row["class"]): float(row["max_kmh"]) for _, row in df.iterrows()}


# -------------------------
# Line selection UI
# -------------------------
def line_picker_ui(first_frame_bgr: np.ndarray) -> Tuple[Line, Line]:
    """
    Select 4 points (2 lines) from the first frame.

    Compatible fix:
    - Older streamlit-image-coordinates has NO use_container_width argument
    - We upscale the image ourselves for bigger display
    - We map click coords back to original frame coords
    - st.rerun() after each click so P1 appears immediately
    """
    st.subheader("Step 1: Select two lines (4 points)")

    h0, w0 = first_frame_bgr.shape[:2]
    st.caption(f"Frame size: {w0} x {h0}. Click 2 points for Line 1, then 2 points for Line 2.")

    # try optional click component
    click_component = None
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        click_component = streamlit_image_coordinates
    except Exception:
        click_component = None

    # Bigger display width (UI only)
    display_w = st.slider("Line selection image width (px)", 800, 1800, 1400, 50)

    if click_component is not None:
        st.info("Click on the image (4 clicks total). Use Reset if needed.")

        if "picked_points" not in st.session_state:
            st.session_state.picked_points = []

        pts: List[Point] = st.session_state.picked_points

        # Draw overlay on original frame
        vis = draw_overlay(first_frame_bgr, pts)

        # Resize for display
        scale = display_w / w0
        display_h = int(h0 * scale)
        vis_disp = cv2.resize(vis, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        vis_rgb = cv2.cvtColor(vis_disp, cv2.COLOR_BGR2RGB)

        click = click_component(vis_rgb, key="img_clicker")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reset points"):
                st.session_state.picked_points = []
                st.rerun()
        with c2:
            st.write("Selected points:", st.session_state.picked_points)

        # If user clicked, map back to original coords and rerun immediately
        if click and len(pts) < 4:
            x_disp = int(click["x"])
            y_disp = int(click["y"])

            x = int(x_disp / scale)
            y = int(y_disp / scale)

            # Clamp to valid pixel range
            x = max(0, min(w0 - 1, x))
            y = max(0, min(h0 - 1, y))

            # Avoid duplicates (same exact click)
            if len(pts) == 0 or (x, y) != pts[-1]:
                pts.append((x, y))
                st.session_state.picked_points = pts
                st.rerun()

        if len(st.session_state.picked_points) < 4:
            st.warning("Select 4 points to continue.")
            st.stop()

        p = st.session_state.picked_points
        return Line(a=p[0], b=p[1]), Line(a=p[2], b=p[3])

    # ---- Fallback manual mode ----
    st.warning(
        "Optional dependency not found: `streamlit-image-coordinates`.\n\n"
        "Fallback: enter the 4 points manually below."
    )

    st.image(cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Line 1")
        l1x1 = st.number_input("L1 x1", min_value=0, max_value=w0 - 1, value=50)
        l1y1 = st.number_input("L1 y1", min_value=0, max_value=h0 - 1, value=50)
        l1x2 = st.number_input("L1 x2", min_value=0, max_value=w0 - 1, value=200)
        l1y2 = st.number_input("L1 y2", min_value=0, max_value=h0 - 1, value=50)

    with col2:
        st.markdown("### Line 2")
        l2x1 = st.number_input("L2 x1", min_value=0, max_value=w0 - 1, value=50)
        l2y1 = st.number_input("L2 y1", min_value=0, max_value=h0 - 1, value=200)
        l2x2 = st.number_input("L2 x2", min_value=0, max_value=w0 - 1, value=200)
        l2y2 = st.number_input("L2 y2", min_value=0, max_value=h0 - 1, value=200)

    line1 = Line(a=(int(l1x1), int(l1y1)), b=(int(l1x2), int(l1y2)))
    line2 = Line(a=(int(l2x1), int(l2y1)), b=(int(l2x2), int(l2y2)))
    return line1, line2


# -------------------------
# Streamlit app
# -------------------------
st.set_page_config(page_title="Vehicle Speed Estimation", layout="wide")
st.title("Vehicle Tracking + Speed Estimation (YOLO + DeepSORT)")

st.markdown(
    """
**Flow**
1) Upload a video  
2) Select 2 lines (4 points)  
3) Edit max speeds per class (rules table)  
4) Run → download processed video + CSV
"""
)

# Model path
default_model_path = "./models/yolov8s/weights/best.pt"
model_path = st.text_input("YOLO model path", value=default_model_path)

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
if not uploaded:
    st.stop()

video_path = save_upload_to_temp(uploaded)

st.subheader("Uploaded video")
# Show bytes for reliable playback
st.video(Path(video_path).read_bytes())

first_frame = read_first_frame(video_path)

# Line selection
line1, line2 = line_picker_ui(first_frame)

# Distance
st.subheader("Step 2: Real distance between the two lines")
distance_m = st.number_input("Distance (meters)", min_value=1.0, value=8.0, step=1.0)

# -------------------------
# Rules table (NO resetting)
# -------------------------
st.subheader("Step 3: Max speed rules (km/h) per YOLO class")

class_names = load_yolo_classnames(model_path)

# Initialize once or when classes change
if "rules_df" not in st.session_state:
    st.session_state.rules_df = build_default_rules(class_names)
    st.session_state.rules_df_classes = class_names

if st.session_state.get("rules_df_classes") != class_names:
    st.session_state.rules_df = build_default_rules(class_names)
    st.session_state.rules_df_classes = class_names

# Choose behavior: apply button (recommended) to avoid mid-edit resets
use_apply_button = st.checkbox("Use 'Apply rules' button (recommended)", value=True)

if use_apply_button:
    with st.form("rules_form"):
        edited_df = st.data_editor(
            st.session_state.rules_df,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            key="rules_editor_form"
        )
        apply_rules = st.form_submit_button("Apply rules")

    if apply_rules:
        st.session_state.rules_df = edited_df
else:
    # Instant mode (still stable because key + session_state)
    edited_df = st.data_editor(
        st.session_state.rules_df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        key="rules_editor_instant"
    )
    st.session_state.rules_df = edited_df

rules_dict = df_to_rules_dict(st.session_state.rules_df)

# -------------------------
# Run
# -------------------------
st.subheader("Step 4: Run processing")

colA, colB, colC = st.columns(3)
with colA:
    conf = st.slider("YOLO confidence", 0.05, 0.90, 0.25, 0.05)
with colB:
    infer_width = st.selectbox("Inference width", [416, 512, 640, 800], index=2)
with colC:
    device = st.text_input("Device", value="cpu", help='Use "cpu" or GPU like "0"')

run = st.button("Run")

if run:
    out_dir = tempfile.mkdtemp(prefix="veh_speed_")
    out_video = os.path.join(out_dir, "processed.mp4")
    out_csv = os.path.join(out_dir, "results.csv")

    with st.spinner("Processing..."):
        rows = process_video(
            video_path=video_path,
            output_video_path=out_video,
            model_path=model_path,
            line1=line1,
            line2=line2,
            distance_m=float(distance_m),
            rules=rules_dict,
            conf=float(conf),
            infer_width=int(infer_width),
            device=device,
            # Optional stronger tracking stability (helps missing speed due to ID switches):
            tracker_max_age=60,
            tracker_n_init=2,
        )

        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)

    st.success("Done!")

    st.subheader("Processed video")
    # Show bytes for reliable playback
    st.video(Path(out_video).read_bytes())

    st.subheader("Results table")
    st.dataframe(df, use_container_width=True)

    # Downloads
    with open(out_video, "rb") as f:
        st.download_button("Download processed video", f, file_name="processed.mp4")
    with open(out_csv, "rb") as f:
        st.download_button("Download results CSV", f, file_name="results.csv")
