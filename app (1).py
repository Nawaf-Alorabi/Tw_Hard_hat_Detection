import os
import io
import cv2
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit as st

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# =========================================
# CONFIG
# =========================================
st.set_page_config(
    page_title="AquaGuard AI | Drowning Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_NAME = "AquaGuard AI"
DEFAULT_METRICS = {
    "before": {"Precision": 0.740, "Recall": 0.690, "mAP50": 0.747, "mAP50-95": 0.467},
    "after":  {"Precision": 0.983, "Recall": 0.780, "mAP50": 0.877, "mAP50-95": 0.564},
}
PER_CLASS = pd.DataFrame(
    {
        "Class": ["Drowning", "Swimming"],
        "Precision": [1.000, 0.966],
        "Recall": [0.561, 1.000],
        "mAP50": [0.759, 0.995],
    }
)

USE_CASES = [
    "Swimming pools and private compounds",
    "Beach and waterfront safety monitoring",
    "Resorts, hotels, and water parks",
    "Smart-city safety systems",
]

# =========================================
# STYLES
# =========================================
st.markdown(
    """
    <style>
    :root {
        --bg: #081425;
        --panel: rgba(255,255,255,0.06);
        --panel-2: rgba(255,255,255,0.09);
        --border: rgba(255,255,255,0.12);
        --text: #F4F8FC;
        --muted: #A9BCD0;
        --accent: #4DD0E1;
        --accent-2: #6C9EFF;
        --danger: #FF6B6B;
        --success: #74E39B;
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(76,208,225,0.16), transparent 25%),
            radial-gradient(circle at top left, rgba(108,158,255,0.14), transparent 22%),
            linear-gradient(180deg, #07111F 0%, #0A1627 45%, #081425 100%);
        color: var(--text);
    }

    .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10,19,34,0.98), rgba(8,20,37,0.98));
        border-right: 1px solid var(--border);
    }

    .hero {
        padding: 2.2rem 2rem;
        border: 1px solid var(--border);
        background: linear-gradient(135deg, rgba(77,208,225,0.16), rgba(108,158,255,0.10), rgba(255,255,255,0.03));
        border-radius: 24px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(77,208,225,0.14);
        border: 1px solid rgba(77,208,225,0.25);
        color: #CFF8FF;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    .metric-card, .glass-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.1rem 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 16px 40px rgba(0,0,0,0.18);
        height: 100%;
    }

    .metric-title {font-size: 0.88rem; color: var(--muted); margin-bottom: 0.35rem;}
    .metric-value {font-size: 1.85rem; font-weight: 800; color: var(--text);}
    .metric-sub {font-size: 0.82rem; color: var(--muted);}

    .section-title {
        font-size: 1.2rem;
        font-weight: 800;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }

    .feature-card {
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
        min-height: 170px;
    }

    .tiny-label {color: var(--muted); font-size: 0.85rem;}
    .strong {font-weight: 700;}

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid var(--border);
        padding: 0.8rem;
        border-radius: 16px;
    }

    .footnote {
        color: var(--muted);
        font-size: 0.85rem;
        margin-top: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================
# HELPERS
# =========================================
@st.cache_resource(show_spinner=False)
def load_model_from_local(path: str):
    if YOLO is None:
        return None, "Ultralytics is not installed. Run: pip install ultralytics"
    if not os.path.exists(path):
        return None, f"Model file not found at: {path}"
    try:
        return YOLO(path), None
    except Exception as e:
        return None, str(e)


def load_model_from_upload(uploaded_file):
    if YOLO is None:
        return None, "Ultralytics is not installed. Run: pip install ultralytics"
    try:
        suffix = Path(uploaded_file.name).suffix or ".pt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        model = YOLO(tmp_path)
        return model, None
    except Exception as e:
        return None, str(e)


def get_model():
    source = st.session_state.get("model_source", "local")
    if source == "upload" and st.session_state.get("uploaded_model") is not None:
        return load_model_from_upload(st.session_state["uploaded_model"])
    local_path = st.session_state.get("local_model_path", "best.pt")
    return load_model_from_local(local_path)


def infer_image(model, pil_image, conf):
    results = model.predict(pil_image, conf=conf, verbose=False)
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    boxes_count = 0
    if hasattr(results[0], "obb") and results[0].obb is not None:
        cls_ids = results[0].obb.cls.cpu().numpy().astype(int).tolist() if results[0].obb.cls is not None else []
        confs = results[0].obb.conf.cpu().numpy().tolist() if results[0].obb.conf is not None else []
        boxes_count = len(cls_ids)
    elif hasattr(results[0], "boxes") and results[0].boxes is not None:
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int).tolist() if results[0].boxes.cls is not None else []
        confs = results[0].boxes.conf.cpu().numpy().tolist() if results[0].boxes.conf is not None else []
        boxes_count = len(cls_ids)
    else:
        cls_ids, confs = [], []

    names = results[0].names if hasattr(results[0], "names") else {}
    labels = [names.get(c, str(c)) for c in cls_ids]
    detections = pd.DataFrame({"Label": labels, "Confidence": confs}) if labels else pd.DataFrame(columns=["Label", "Confidence"])
    return annotated, detections, boxes_count


def process_video(model, uploaded_video, conf, stride=5):
    t0 = time.time()
    suffix = Path(uploaded_video.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(uploaded_video.read())
        input_path = tmp_in.name

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
        output_path = tmp_out.name

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_id = 0
    summary = []
    progress = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % stride == 0:
            results = model.predict(frame, conf=conf, verbose=False)
            rendered = results[0].plot()
            names = results[0].names if hasattr(results[0], "names") else {}
            if hasattr(results[0], "obb") and results[0].obb is not None and results[0].obb.cls is not None:
                cls_ids = results[0].obb.cls.cpu().numpy().astype(int).tolist()
            elif hasattr(results[0], "boxes") and results[0].boxes is not None and results[0].boxes.cls is not None:
                cls_ids = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
            else:
                cls_ids = []
            labels = [names.get(c, str(c)) for c in cls_ids]
            summary.extend(labels)
            writer.write(rendered)
        else:
            writer.write(frame)

        frame_id += 1
        progress.progress(min(frame_id / total_frames, 1.0))

    cap.release()
    writer.release()
    progress.empty()

    counts = pd.Series(summary).value_counts().reset_index()
    if not counts.empty:
        counts.columns = ["Label", "Count"]
    else:
        counts = pd.DataFrame(columns=["Label", "Count"])

    runtime = time.time() - t0
    return output_path, counts, runtime


def show_metric_card(title, value, subtext=""):
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>{title}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-sub'>{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================
# SIDEBAR
# =========================================
st.sidebar.markdown(f"## 🌊 {PROJECT_NAME}")
st.sidebar.caption("AI-powered drowning detection dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Detection Studio", "Performance Analytics", "Operational Insights"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Setup")
model_source = st.sidebar.radio("Model source", ["local", "upload"], horizontal=True)
st.session_state["model_source"] = model_source

if model_source == "local":
    st.session_state["local_model_path"] = st.sidebar.text_input("Local model path", value="best.pt")
else:
    st.session_state["uploaded_model"] = st.sidebar.file_uploader("Upload YOLO weights", type=["pt"])

conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: place your trained `best.pt` in the same folder as `app.py` for instant use.")

# =========================================
# PAGE: OVERVIEW
# =========================================
if page == "Overview":
    st.markdown(
        f"""
        <div class='hero'>
            <div class='badge'>Computer Vision Safety System</div>
            <h1 style='margin:0;'>Preventive Water Safety with Real-Time AI Detection</h1>
            <p style='color:#D8E7F7; font-size:1.05rem; max-width:900px;'>
                {PROJECT_NAME} is a polished Streamlit interface for your YOLOv8-based drowning detection project.
                It transforms the notebook into a product-style experience for demos, judging, and deployment storytelling.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_metric_card("Precision", f"{DEFAULT_METRICS['after']['Precision']:.3f}", "After tuning")
    with c2:
        show_metric_card("Recall", f"{DEFAULT_METRICS['after']['Recall']:.3f}", "After tuning")
    with c3:
        show_metric_card("mAP50", f"{DEFAULT_METRICS['after']['mAP50']:.3f}", "After tuning")
    with c4:
        show_metric_card("mAP50-95", f"{DEFAULT_METRICS['after']['mAP50-95']:.3f}", "After tuning")

    st.write("")
    a, b = st.columns([1.15, 0.85], gap="large")

    with a:
        st.markdown("<div class='section-title'>Project Story</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='glass-card'>
                <p><span class='strong'>Objective:</span> detect risky drowning situations versus normal swimming activity using a fine-tuned YOLOv8 model.</p>
                <p><span class='strong'>Model family:</span> YOLOv8 OBB (oriented bounding box) for more flexible detection geometry.</p>
                <p><span class='strong'>Improvement path:</span> the notebook shows a baseline model and an improved tuned version using better augmentation, optimizer, and training settings.</p>
                <p><span class='strong'>Why this matters:</span> fast visual monitoring can support early intervention in pools, beaches, resorts, and smart safety environments.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("<div class='section-title'>Before vs After Tuning</div>", unsafe_allow_html=True)
        compare_df = pd.DataFrame({
            "Metric": list(DEFAULT_METRICS["before"].keys()),
            "Before Tuning": list(DEFAULT_METRICS["before"].values()),
            "After Tuning": list(DEFAULT_METRICS["after"].values()),
        })
        compare_long = compare_df.melt(id_vars="Metric", var_name="Version", value_name="Score")
        fig = px.bar(compare_long, x="Metric", y="Score", color="Version", barmode="group", text_auto=".3f")
        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            font=dict(color="#F4F8FC"),
            legend_title_text="",
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with b:
        st.markdown("<div class='section-title'>What the App Includes</div>", unsafe_allow_html=True)
        for title, desc in [
            ("Detection Studio", "Run drowning/swimming detection on uploaded images or videos."),
            ("Performance Dashboard", "Present benchmark metrics before/after tuning and per-class quality."),
            ("Judge-Friendly Presentation", "A polished layout for competitions, demos, and project showcases."),
            ("Deployment-Ready Flow", "Supports loading your trained `best.pt` directly for real inference."),
        ]:
            st.markdown(
                f"""
                <div class='feature-card'>
                    <div style='font-size:1.03rem; font-weight:800; margin-bottom:0.4rem;'>{title}</div>
                    <div class='tiny-label'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

    st.markdown("<div class='section-title'>Potential Deployment Environments</div>", unsafe_allow_html=True)
    cols = st.columns(len(USE_CASES))
    for col, label in zip(cols, USE_CASES):
        with col:
            st.markdown(f"<div class='feature-card'><div class='strong'>{label}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='footnote'>Metrics shown on this page were extracted from your notebook content.</div>", unsafe_allow_html=True)

# =========================================
# PAGE: DETECTION STUDIO
# =========================================
elif page == "Detection Studio":
    st.title("Detection Studio")
    st.caption("Upload an image or a short video to test your trained YOLO model.")

    model, model_error = get_model()
    if model is None:
        st.error(model_error or "Model could not be loaded.")
        st.info("Place `best.pt` next to `app.py`, or upload the weights file from the sidebar.")
    else:
        mode = st.radio("Input type", ["Image", "Video"], horizontal=True)

        if mode == "Image":
            uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_image is not None:
                image = Image.open(uploaded_image).convert("RGB")
                c1, c2 = st.columns(2, gap="large")
                with c1:
                    st.image(image, caption="Original image", use_container_width=True)
                with st.spinner("Running detection..."):
                    annotated, detections, boxes_count = infer_image(model, image, conf_threshold)
                with c2:
                    st.image(annotated, caption="Annotated prediction", use_container_width=True)

                d1, d2, d3 = st.columns(3)
                d1.metric("Detected objects", boxes_count)
                d2.metric("Drowning detections", int((detections["Label"] == "Drowning").sum()) if not detections.empty else 0)
                d3.metric("Swimming detections", int((detections["Label"] == "Swimming").sum()) if not detections.empty else 0)

                if not detections.empty:
                    st.subheader("Detection details")
                    det_table = detections.copy()
                    det_table["Confidence"] = det_table["Confidence"].map(lambda x: f"{x:.3f}")
                    st.dataframe(det_table, use_container_width=True, hide_index=True)

                    label_counts = detections["Label"].value_counts().reset_index()
                    label_counts.columns = ["Label", "Count"]
                    fig = px.pie(label_counts, names="Label", values="Count", hole=0.55)
                    fig.update_layout(
                        height=360,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#F4F8FC"),
                        margin=dict(l=20, r=20, t=20, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No objects were detected at the selected confidence threshold.")

        else:
            uploaded_video = st.file_uploader("Upload a short video", type=["mp4", "mov", "avi", "mkv"])
            stride = st.slider("Process every Nth frame", 1, 12, 5)
            if uploaded_video is not None:
                with st.spinner("Processing video and generating annotated output..."):
                    output_path, counts, runtime = process_video(model, uploaded_video, conf_threshold, stride=stride)
                st.success(f"Video processed successfully in {runtime:.1f} seconds.")
                st.video(output_path)

                c1, c2 = st.columns([0.9, 1.1])
                with c1:
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "Download annotated video",
                            data=f,
                            file_name="annotated_drowning_detection.mp4",
                            mime="video/mp4",
                        )
                with c2:
                    if not counts.empty:
                        fig = px.bar(counts, x="Label", y="Count", text_auto=True)
                        fig.update_layout(
                            height=350,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(255,255,255,0.03)",
                            font=dict(color="#F4F8FC"),
                            margin=dict(l=20, r=20, t=20, b=20),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No detections were found across the processed frames.")

# =========================================
# PAGE: PERFORMANCE ANALYTICS
# =========================================
elif page == "Performance Analytics":
    st.title("Performance Analytics")
    st.caption("A polished summary of model quality extracted from the notebook.")

    top1, top2 = st.columns([1.05, 0.95], gap="large")
    with top1:
        st.subheader("Benchmark Comparison")
        radar_categories = list(DEFAULT_METRICS["after"].keys())
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(DEFAULT_METRICS["before"].values()),
            theta=radar_categories,
            fill='toself',
            name='Before Tuning'
        ))
        fig.add_trace(go.Scatterpolar(
            r=list(DEFAULT_METRICS["after"].values()),
            theta=radar_categories,
            fill='toself',
            name='After Tuning'
        ))
        fig.update_layout(
            polar=dict(bgcolor="rgba(255,255,255,0.03)", radialaxis=dict(visible=True, range=[0,1])),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F4F8FC"),
            height=460,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with top2:
        st.subheader("Interpretation")
        st.markdown(
            """
            <div class='glass-card'>
                <p><span class='strong'>Precision improved strongly</span>, which means detections became much more reliable after tuning.</p>
                <p><span class='strong'>Recall also improved</span>, but it is still lower than precision, indicating some difficult drowning cases are still missed.</p>
                <p><span class='strong'>Swimming detection is extremely strong</span>, while drowning remains the harder class.</p>
                <p><span class='strong'>Main challenge:</span> drowning examples appear visually complex and are likely underrepresented compared with swimming samples.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Per-Class Performance")
        melted = PER_CLASS.melt(id_vars="Class", var_name="Metric", value_name="Score")
        fig = px.bar(melted, x="Metric", y="Score", color="Class", barmode="group", text_auto=".3f")
        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            font=dict(color="#F4F8FC"),
            margin=dict(l=20, r=20, t=20, b=20),
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Class-Level Takeaways")
        st.dataframe(PER_CLASS.style.format({"Precision":"{:.3f}","Recall":"{:.3f}","mAP50":"{:.3f}"}), use_container_width=True, hide_index=True)
        st.markdown(
            """
            <div class='footnote'>
                Drowning recall is the most important weakness to improve next, because missing a real drowning case is more critical than a normal swimming detection.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Improvement Priorities")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("<div class='feature-card'><div class='strong'>Add more drowning samples</div><div class='tiny-label'>Increase difficult positive cases, poses, angles, and water conditions.</div></div>", unsafe_allow_html=True)
    with p2:
        st.markdown("<div class='feature-card'><div class='strong'>Balance the dataset</div><div class='tiny-label'>Reduce class imbalance between swimming and drowning examples.</div></div>", unsafe_allow_html=True)
    with p3:
        st.markdown("<div class='feature-card'><div class='strong'>Stress-test on real scenes</div><div class='tiny-label'>Validate on videos with reflections, crowding, and occlusion.</div></div>", unsafe_allow_html=True)

# =========================================
# PAGE: OPERATIONAL INSIGHTS
# =========================================
else:
    st.title("Operational Insights")
    st.caption("Turn the notebook into a competition-ready product story.")

    c1, c2 = st.columns([1.05, 0.95], gap="large")
    with c1:
        st.subheader("Value Proposition")
        st.markdown(
            """
            <div class='glass-card'>
                <p>This system can act as an <span class='strong'>early-warning visual assistant</span> for water safety teams.</p>
                <p>Instead of replacing human supervision, it enhances monitoring by highlighting suspicious activity faster.</p>
                <p>That makes it especially strong as a <span class='strong'>decision-support layer</span> in busy or wide-area environments.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Suggested Demo Flow")
        st.markdown(
            """
            1. Start with the problem and real-world impact.  
            2. Show your benchmark metrics before and after tuning.  
            3. Run live image/video inference inside the app.  
            4. Explain why drowning recall is the next optimization target.  
            5. End with deployment potential in pools, beaches, and smart safety systems.
            """
        )

    with c2:
        st.subheader("System Limitations")
        limitation_df = pd.DataFrame(
            {
                "Area": ["Dataset", "Lighting", "Occlusion", "View Angle", "Class Similarity"],
                "Impact": [
                    "Too few drowning cases can limit generalization",
                    "Reflections and shadows may reduce confidence",
                    "Crowded scenes can hide critical actions",
                    "Unusual camera angles can distort detection quality",
                    "Swimming and drowning can look visually similar in some frames",
                ],
            }
        )
        st.dataframe(limitation_df, use_container_width=True, hide_index=True)

    st.write("")
    st.subheader("Competition-Ready Positioning")
    cards = st.columns(4)
    content = [
        ("Premium UX", "Clean interface, clear navigation, and polished visual storytelling."),
        ("Technical Depth", "YOLO-based detection, tuning comparison, and deployment-ready workflow."),
        ("Practical Impact", "Addresses a high-stakes real-world safety problem."),
        ("Scalable Vision", "Can extend to CCTV integration, alerting, and centralized safety monitoring."),
    ]
    for col, (title, desc) in zip(cards, content):
        with col:
            st.markdown(f"<div class='feature-card'><div class='strong'>{title}</div><div class='tiny-label'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='footnote'>To run real inference, keep your trained YOLO weights available as <code>best.pt</code> or upload them from the sidebar.</div>",
        unsafe_allow_html=True,
    )
