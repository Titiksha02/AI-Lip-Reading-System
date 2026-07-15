import streamlit as st
import tempfile
from predict import predict_video

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="AI Lip Reading System",
    page_icon="🎥",
    layout="wide"
)

# ======================
# CUSTOM CSS
# ======================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b
    );
}

.hero {
    text-align: center;
    padding: 20px;
}

.hero h1 {
    color: white;
    font-size: 55px;
    font-weight: bold;
}

.hero p {
    color: #cbd5e1;
    font-size: 20px;
}

.card {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}

.prediction-box {
    background: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: #2563eb;
    font-size: 28px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================

st.markdown("""
<div class="hero">
<h1>🎥 AI Lip Reading System</h1>
<p>
Converting Silent Lip Movements Into Text
Using Deep Learning & Computer Vision
</p>
</div>
""", unsafe_allow_html=True)

# ======================
# INFO CARDS
# ======================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
    <h3>Dataset</h3>
    <h2>GRID Corpus</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h3>Frames</h3>
    <h2>75</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <h3>Model</h3>
    <h2>LipNet</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ======================
# PROJECT OVERVIEW
# ======================

with st.expander("📖 Project Overview"):

    st.write("""
    This project converts silent lip
    movements into text using
    Deep Learning and Computer Vision.

    Workflow:

    Video Upload
    → Lip Extraction
    → Conv3D Layers
    → BiGRU Layers
    → CTC Decoder
    → Text Prediction
    """)

# ======================
# VIDEO UPLOAD
# ======================

uploaded_file = st.file_uploader(
    "📂 Upload Video",
    type=["mp4", "avi", "mpg", "mpeg"]
)

if uploaded_file:

    st.video(uploaded_file)

    if st.button("🔍 Predict Speech"):

        with st.spinner("Analyzing Lip Movements..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as tmp:

                tmp.write(uploaded_file.read())

                prediction = predict_video(tmp.name)

        st.markdown(f"""
        <div class="prediction-box">
        Prediction<br><br>
        {prediction}
        </div>
        """, unsafe_allow_html=True)

# ======================
# ARCHITECTURE
# ======================

st.markdown("## ⚙️ System Architecture")

st.code("""
Video Input
     ↓
Lip Extraction
     ↓
Conv3D Layers
     ↓
BiGRU Layers
     ↓
CTC Decoder
     ↓
Predicted Text
""")

# ======================
# DATASET
# ======================

with st.expander("📚 GRID Corpus Dataset"):

    st.write("""
    GRID Corpus is a large audio-visual
    speech dataset containing thousands
    of videos from multiple speakers.

    Example Sentence:

    bin blue by e four soon
    """)

# ======================
# FOOTER
# ======================

st.markdown("""
<div class="footer">
<hr>
AI-Based Lip Reading System<br>
Final Year BE Project 2026<br>
Developed by Titiksha Chaudhari
</div>
""", unsafe_allow_html=True)