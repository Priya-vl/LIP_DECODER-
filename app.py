import streamlit as st
import tempfile
import os
import base64
import cv2
import numpy as np
import pydub
import queue  # Added this to fix the error in the live decoding block
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from utils import visual_feature_extraction, deep_lip_decode, translate_content, face_mesh

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Lip Decoder",
    page_icon="lip_logo.png",  #
    layout="wide",
    initial_sidebar_state="collapsed"
)

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Helper to encode local images for HTML/CSS
def get_base64(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return ""


logo_b64 = get_base64("lip_logo.png")  #

# --- 2. COMPLETE STYLING WITH ANIMATIONS ---
st.markdown(f"""
<style>
    /* Global White Background */
    .stApp {{
        background-color: #af9aff !important;
    }}

    /* Animation Definitions */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-20px); }}
        100% {{ transform: translateY(0px); }}
    }}

    /* Main Container with Fade-In */
    .main-container {{
        margin-top: 120px;
        text-align: center;
        width: 100% !important;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #000000;
        animation: fadeIn 1.2s ease-out;
    }}

    /* Persistent Off-White Top Bar */
    .top-navbar {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #ffffff;
        height: 70px;
        padding: 0 20px;
        display: flex;
        align-items: center;
        z-index: 9999;
        border-bottom: 1px solid #eeeeee;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
    }}
    
    .nav-logo {{
        height: 45px; 
        margin-right: 15px;
    }}

    .nav-text {{
        font-family: 'Arial Black', sans-serif;
        font-size: 24px; 
        font-weight: 900;
        color: #000000;
        line-height: 1; 
    }}

    /* Hero Logo with Floating Animation */
    .hero-logo {{
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
        width: 480px; /* Big Logo */
        margin-bottom: 30px;
        border-radius: 20px;
        animation: float 4s ease-in-out infinite;
        box-shadow: 0px 15px 45px rgba(0,0,0,0.1);
    }}

    /* Enlarged Hero Title */
    .hero-title {{
        font-size: 5.5rem !important; /* Little Big Title */
        font-weight: 900;
        color: #000000 !important;
        margin-bottom: 15px;
        font-family: 'Zuume Rough Bold', sans-serif;
        text-align: center;
        width: 100%;
    }}

    /* Explore Now Button: White Background, Black Text */
    div.stButton > button {{
        background-color: #1f2f56 !important;
        color: #ffffff !important; 
        border: none !important;             /* Remove the black border */
        border-radius: 60px !important;      /* Extra rounded pill shape */
        padding: 15px 70px !important;
        font-weight: 800 !important;         /* Extra bold text */
        font-size: 80px !important;          /* Large, readable text */
        text-transform: uppercase;           /* Forces EXPLORE NOW look */
        letter-spacing: 1.5px !important;    /* Slight spacing for readability */
        font-family: 'Impact' !important;
        padding: 18px 80px !important;
        width: auto !important; 
        display: block;
        margin: 0 auto;
        transition: 0.3s ease-in-out;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2); /* Adds a subtle lift */
        }}

    /* Hover Effect: Lighten slightly or scale */
    div.stButton > button:hover {{ 
        background-color: #2a3b66 !important; 
        transform: scale(1.05);  
        box-shadow: 0px 8px 25px rgba(0,0,0,0.3);        }}
        

    /* Language Cards Styling (Image 2 style) */
    .lang-card {{
        border-radius: 35px;
        padding: 60px 20px;
        text-align: left;
        color: white;
        margin-bottom: 15px;
        transition: 0.3s;
    }}
    .lang-card:hover {{ transform: translateY(-10px); }}
    .lang-card h2 {{ color: white !important; font-size: 2.8rem; }}

    /* Source Boxes Styling (Image 3 style) */
    .source-box {{
        border-radius: 40px;
        padding: 50px 20px;
        text-align: center;
        background-color: #1f2f56 !important; /* Navy Blue background */
        color: #ffffff !important;            /* White text */
        transition: 0.3s ease-in-out;
        cursor: pointer;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 20px;
        border: none;
    }}
    
    .source-box:hover {{
        transform: translateY(-10px) scale(1.02);
        background-color: #2a3b66 !important; /* Lighten slightly on hover */
        box-shadow: 0px 15px 40px rgba(0,0,0,0.3);
    }}

    /* Ensure icons and text inside are white */
    .source-box h1, .source-box h2 {{
        color: #ffffff !important;
        margin: 10px 0;
    }}

    /* Fix for text visibility */
    h1, h2, h3 {{
        color: ##b4b4b4 ;
    }}
</style>

<div class="top-navbar">
    <img src="data:image/jpeg;base64,{logo_b64}" class="nav-logo">
    <div class="nav-text">
        LIP <span style="color: #8c52ff;">DECODER</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'page' not in st.session_state: st.session_state.page = 'landing'
if 'target_lang' not in st.session_state: st.session_state.target_lang = "English"
if 'target_lang_code' not in st.session_state: st.session_state.target_lang_code = 'en'
if 'video_path' not in st.session_state: st.session_state.video_path = None


def navigate(p):
    st.session_state.page = p
    st.rerun()


# --- 4. PAGE LOGIC ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# PAGE 1: LANDING
if st.session_state.page == 'landing':
    st.markdown(f'<img src="data:image/jpeg;base64,{logo_b64}" class="hero-logo">', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">LIP <span style="color:#8706ed;">DECODER</span></h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 1.4rem; margin-bottom: 50px;max-width: 800px;text-align: center;margin-left: auto; margin-right: auto;'>Experience seamless communication: Transforming silent lip movements into instant text translation.</p>",
        unsafe_allow_html=True)

    if st.button("EXPLORE NOW"):
        navigate('language')

# PAGE 2: LANGUAGE SELECTION
elif st.session_state.page == 'language':
    st.markdown("<h1 style='margin-bottom: 50px;'>SELECT LANGUAGE</h1>", unsafe_allow_html=True)
    langs = [
        ("English", "en", "#5c7cff"), ("Hindi", "hi", "#6200ea"),
        ("Tamil", "ta", "#00bcd4"), ("Kannada", "kn", "#00838f"),
        ("Telugu", "te", "#ce93d8")
    ]
    cols = st.columns(3)
    for idx, (name, code, color) in enumerate(langs):
        with cols[idx % 3]:
            st.markdown(f'<div class="lang-card" style="background:{color};"><h2>{name}</h2></div>',
                        unsafe_allow_html=True)
            if st.button(f"Start Now", key=f"btn_{name}"):
                st.session_state.target_lang = name
                st.session_state.target_lang_code = code
                navigate('source')
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back"): navigate('landing')

# PAGE 3: SOURCE SELECTION
elif st.session_state.page == 'source':
    st.markdown("<h1 style='margin-bottom: 50px;'>CHOOSE INPUT SOURCE</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="source-box"><h1 style="font-size: 100px;">🗁</h1><h2>RECORDED VIDEO</h2></div>',
                    unsafe_allow_html=True)
        if st.button("UPLOAD FILE"):
            st.session_state.input_method = 'upload'
            navigate('analyze')
    with c2:
        st.markdown('<div class="source-box"><h1 style="font-size: 100px;">🎥</h1><h2>LIVE VIDEO</h2></div>',
                    unsafe_allow_html=True)
        if st.button("START CAMERA"):
            st.session_state.input_method = 'live'
            navigate('analyze')
    if st.button("← Back"): navigate('language')


# PAGE 4: ANALYSIS
elif st.session_state.page == 'analyze':
    st.markdown(f"<h1>ANALYZING: {st.session_state.target_lang.upper()}</h1>", unsafe_allow_html=True)
# --- MODE 1: UPLOADED FILE ---
    if st.session_state.input_method == 'upload':
        uploaded = st.file_uploader("Upload Video", type=['mp4', 'mov', 'webm'])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded.read())
            st.session_state.video_path = tfile.name
            st.video(st.session_state.video_path)
            if st.button("ANALYZE LIP MOVEMENTS"):
                with st.spinner("Processing Model Sequences..."):
                    visual_feature_extraction(st.session_state.video_path)
                    raw = deep_lip_decode(st.session_state.video_path)
                    translated = translate_content(raw, st.session_state.target_lang_code)
                    st.markdown(f"""
                    <div style="background:#f48585; padding:40px; border-radius:25px; border-left: 12px solid #8c52ff; margin-top:30px; text-align:left;">
                        <h2 style="color:#8c52ff !important;">Output:</h2>
                        <h1 style="font-size:2.0rem; color:#000000 !important;">"{translated}"</h1>
                    </div>
                    """, unsafe_allow_html=True)
    # --- MODE 2: LIVE CAMERA & AUDIO ---
    else:
        # These lines MUST be indented 4 spaces relative to the 'else'
        st.info("Live Mode: Face detection and lip capture active.")


        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            # Prove visual layer is active: Detect Face landmarks
            face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return frame


        webrtc_ctx = webrtc_streamer(
            key="lip-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": True},
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            if st.button("🔴 STOP & DECODE NOW"):
                # 1. Check if the audio receiver is actually ready
                if webrtc_ctx.audio_receiver is None:
                    st.error("Audio receiver is not ready. Please wait a moment or refresh.")
                else:
                    with st.spinner("Processing live audio for translation..."):
                        try:
                            # 2. Extract audio frames with a safety timeout
                            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)

                            if not audio_frames:
                                st.warning("No audio detected. Please speak into the microphone.")
                            else:
                                # 3. Stitch audio frames using pydub
                                sound_chunk = pydub.AudioSegment.empty()
                                for f in audio_frames:
                                    s = pydub.AudioSegment(
                                        data=f.to_ndarray().tobytes(),
                                        sample_width=f.format.bytes,
                                        frame_rate=f.sample_rate,
                                        channels=len(f.layout.channels)
                                    )
                                    sound_chunk += s

                                # 4. Export and Decode
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
                                    sound_chunk.export(tmp_wav.name, format="wav")
                                    raw = deep_lip_decode(tmp_wav.name)
                                    translated = translate_content(raw, st.session_state.target_lang_code)

                                    st.markdown(f"""
                                            <div class="output-box">
                                                <h2 style="color:#8c52ff;">Live Output:</h2>
                                                <h1 style="color:#000000;">"{translated}"</h1>
                                            </div>
                                            """, unsafe_allow_html=True)
                        except queue.Empty:
                            st.warning("Lip movement is not clear. Speak for a longer duration.")
                        except Exception as e:
                            st.error(f"Processing Error: {e}")
    if st.button("Start Over"): navigate('landing')


st.markdown('</div>', unsafe_allow_html=True)
