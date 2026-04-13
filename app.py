import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hat Detection – YOLOv8",
    page_icon="⛑️",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #555;
        font-size: 1rem;
        margin-top: 0;
    }
    .metric-card {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        border: 1px solid #d0dff8;
    }
    .metric-label { font-size: 0.8rem; color: #666; text-transform: uppercase; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1a73e8; }
    .badge-head    { background:#fff3cd; color:#856404; border-radius:6px; padding:3px 10px; font-weight:600; }
    .badge-helmet  { background:#d1e7dd; color:#0f5132; border-radius:6px; padding:3px 10px; font-weight:600; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⛑️ Hat Detection — YOLOv8 OBB</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and let the model detect <b>heads</b> and <b>helmets</b> in real time.</p>', unsafe_allow_html=True)
st.divider()

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_path = st.text_input(
        "Model path (.pt)",
        value="hat_detection_tuned_best.pt",
        help="Path to your trained YOLOv8 OBB weights file.",
    )

    conf_threshold = st.slider(
        "Confidence threshold", min_value=0.10, max_value=0.95,
        value=0.25, step=0.05,
        help="Only show detections above this confidence score."
    )

    iou_threshold = st.slider(
        "IoU threshold (NMS)", min_value=0.10, max_value=0.95,
        value=0.45, step=0.05,
    )

    show_labels  = st.toggle("Show labels",      value=True)
    show_conf    = st.toggle("Show confidence",  value=True)

    st.divider()
    st.markdown("### 📊 Tuned Model Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Precision", "0.942")
    col2.metric("Recall",    "0.876")
    col1.metric("mAP50",     "0.931")
    col2.metric("mAP50-95",  "0.758")

    st.divider()
    st.caption("Built with [Ultralytics YOLOv8](https://docs.ultralytics.com) · OBB variant")

# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    if not os.path.exists(path):
        return None, f"❌ Model file not found: `{path}`"
    try:
        m = YOLO(path)
        return m, None
    except Exception as e:
        return None, str(e)

model, model_error = load_model(model_path)

if model_error:
    st.error(model_error)
    st.info(
        "**How to get the model file:**\n"
        "1. Run the training notebook and download `hat_detection_tuned_best.pt`.\n"
        "2. Place it in the same folder as this `app.py`.\n"
        "3. Update the path in the sidebar if needed."
    )
    st.stop()

st.success(f"✅ Model loaded: `{model_path}`", icon="🟢")

# ─── Upload & Predict ────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📤 Upload an image to test",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Drag & drop or click to browse."
)

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    img_np  = np.array(pil_img)

    col_orig, col_pred = st.columns(2, gap="large")

    with col_orig:
        st.markdown("#### 🖼️ Original Image")
        st.image(pil_img, use_container_width=True)
        st.caption(f"Size: {pil_img.width} × {pil_img.height} px")

    with col_pred:
        st.markdown("#### 🔍 Detection Result")

        with st.spinner("Running inference…"):
            t0 = time.perf_counter()
            results = model.predict(
                source=img_np,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

        result = results[0]

        # ── Draw OBB annotations ──────────────────────────────────────────
        annotated = result.plot(
            labels=show_labels,
            conf=show_conf,
            line_width=2,
        )
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)
        st.caption(f"⏱️ Inference: {elapsed_ms:.1f} ms")

    # ─── Detection Summary ───────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Detection Summary")

    if result.obb is not None and len(result.obb.cls) > 0:
        cls_ids = result.obb.cls.cpu().numpy().astype(int)
        confs   = result.obb.conf.cpu().numpy()
        names   = [result.names[c] for c in cls_ids]

        head_count   = names.count("head")
        helmet_count = names.count("helmet")

        m1, m2, m3 = st.columns(3)
        m1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total detections</div>
            <div class="metric-value">{len(names)}</div>
        </div>""", unsafe_allow_html=True)

        m2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🧠 Heads</div>
            <div class="metric-value">{head_count}</div>
        </div>""", unsafe_allow_html=True)

        m3.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⛑️ Helmets</div>
            <div class="metric-value">{helmet_count}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Per-detection details")
        rows = []
        for i, (name, conf) in enumerate(zip(names, confs), 1):
            badge = f'<span class="badge-{name}">{name}</span>'
            bar   = f'<progress value="{conf:.2f}" max="1" style="width:120px"></progress>'
            rows.append(f"<tr><td>{i}</td><td>{badge}</td><td>{conf:.3f} {bar}</td></tr>")

        table_html = f"""
        <table style="width:100%; border-collapse:collapse; font-size:0.9rem;">
          <thead>
            <tr style="background:#f0f4ff;">
              <th style="padding:8px; text-align:left;">#</th>
              <th style="padding:8px; text-align:left;">Class</th>
              <th style="padding:8px; text-align:left;">Confidence</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>"""
        st.markdown(table_html, unsafe_allow_html=True)

        # ── Download annotated image ─────────────────────────────────────
        st.divider()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            Image.fromarray(annotated_rgb).save(tmp.name, quality=95)
            with open(tmp.name, "rb") as f:
                st.download_button(
                    label="⬇️ Download annotated image",
                    data=f.read(),
                    file_name=f"detection_{uploaded.name}",
                    mime="image/jpeg",
                )

    else:
        st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")

else:
    st.info("👆 Upload an image above to get started.")

    # Show example class legend
    st.markdown("### 🏷️ Detectable Classes")
    c1, c2 = st.columns(2)
    c1.markdown("""
    **🧠 Head**
    - Unprotected / bare head  
    - Precision: 0.946 | Recall: 0.910  
    - mAP50: 0.963
    """)
    c2.markdown("""
    **⛑️ Helmet**
    - Safety / hard hat  
    - Precision: 0.939 | Recall: 0.841  
    - mAP50: 0.899
    """)
