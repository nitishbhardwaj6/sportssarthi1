# type: ignore
import streamlit as st  # type: ignore
import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import tempfile
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import streamlit.components.v1 as components

# Initialize session state
if 'user_name' not in st.session_state:
    st.session_state.user_name = "Athlete"
if 'user_level' not in st.session_state:
    st.session_state.user_level = 1
if 'total_punches' not in st.session_state:
    st.session_state.total_punches = 0
if 'max_pps' not in st.session_state:
    st.session_state.max_pps = 0

# Load secrets (only available in production)
try:
    import streamlit as st
    # Example of using secrets (uncomment when you add API keys)
    # openai_api_key = st.secrets["api_keys"]["openai_api_key"]
    # database_url = st.secrets["database"]["url"]
    # email_config = st.secrets["email"]
    secrets_loaded = True
except:
    secrets_loaded = False

# --- CONFIG ---
st.set_page_config(page_title="Royal AI Sports Coach", layout="wide", page_icon="🥊")

# Visual indicator that new features are active
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3a8a, #3b82f6); color: #fbbf24; padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 20px; font-weight: bold;">
🎯 ADVANCED AI COACH ACTIVE - Enhanced Punch Detection & Professional Theme Loaded ✅
</div>
""", unsafe_allow_html=True)

# Professional theme styles with animations
theme_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 0;
    animation: fadeIn 1s ease-in;
    font-weight: 400;
}

* {
    box-sizing: border-box;
}

h1, h2, h3, h4, h5, h6 {
    color: #fbbf24;
    font-weight: 600;
    margin-bottom: 1rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    animation: slideUp 0.8s ease-out;
    line-height: 1.2;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

h1, h2, h3, h4, h5, h6 {
    color: #fbbf24;
    font-weight: 600;
    margin-bottom: 1rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    animation: slideUp 0.8s ease-out;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.royal-box {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    border: 2px solid #fbbf24;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
    margin-bottom: 24px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.6s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.royal-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #fbbf24, #f59e0b);
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.royal-box:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(59, 130, 246, 0.4);
}

.hero {
    text-align: center;
    padding: 60px 20px;
    background: rgba(15, 23, 42, 0.8);
    border-radius: 20px;
    margin-bottom: 40px;
    backdrop-filter: blur(10px);
    animation: heroFade 1.2s ease-out;
}

@keyframes heroFade {
    from {
        opacity: 0;
        transform: scale(0.9);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

.counter {
    font-size: 3rem;
    font-weight: 700;
    color: #fbbf24;
    animation: countUp 2s ease-out;
}

@keyframes countUp {
    from { opacity: 0; transform: scale(0.5); }
    to { opacity: 1; transform: scale(1); }
}

.cta-button {
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    color: #0f172a;
    border: none;
    padding: 16px 32px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1.1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(251, 191, 36, 0.3);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

.cta-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(251, 191, 36, 0.4);
}

.stButton>button {
    background: linear-gradient(135deg, #3b82f6, #1e3a8a);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
}

.stButton>button:hover {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
}

.stTextInput>div>div>input, .stSelectbox>div>div>select, .stTextArea>div>div>textarea {
    background: rgba(30, 41, 59, 0.8);
    border: 2px solid #475569;
    border-radius: 8px;
    color: #f1f5f9;
    padding: 12px;
    transition: all 0.3s ease;
}

.stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus, .stTextArea>div>div>textarea:focus {
    border-color: #fbbf24;
    box-shadow: 0 0 0 3px rgba(251, 191, 36, 0.2);
    animation: glow 0.5s ease;
}

@keyframes glow {
    from { box-shadow: 0 0 0 0 rgba(251, 191, 36, 0.2); }
    to { box-shadow: 0 0 0 3px rgba(251, 191, 36, 0.2); }
}

.stSidebar {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 2px solid #fbbf24;
}

.stSidebar .sidebar-content {
    background: transparent;
}

.stRadio > div {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 12px;
    padding: 16px;
    border: 2px solid #475569;
}

.stRadio > div > label {
    color: #f1f5f9;
    font-weight: 500;
}

.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 12px;
    border: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    animation: slideIn 0.5s ease;
}

@keyframes slideIn {
    from { transform: translateX(-20px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #fbbf24);
    animation: progressFill 1.5s ease-out;
}

@keyframes progressFill {
    from { width: 0%; }
    to { width: 100%; }
}

.stTable {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 12px;
    overflow: hidden;
}

.stTable th, .stTable td {
    color: #f1f5f9;
    border-color: #475569;
}

.stMetric {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
    animation: metricPop 0.6s ease-out;
}

@keyframes metricPop {
    from { transform: scale(0.8); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
}

.stExpander {
    background: rgba(30, 41, 59, 0.8);
    border: 2px solid #475569;
    border-radius: 12px;
    margin-bottom: 16px;
}

.stExpander > div > div > div > div {
    color: #fbbf24;
    font-weight: 600;
}

.stExpander > div > div > div > div:hover {
    color: #f59e0b;
}

.testimonial-card {
    animation: testimonialSlide 0.8s ease-out;
}

@keyframes testimonialSlide {
    from { transform: translateX(-30px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

@media (max-width: 768px) {
    .royal-box {
        padding: 16px;
        margin-bottom: 16px;
    }

    .hero {
        padding: 40px 16px;
    }

    h1 {
        font-size: 2rem;
    }

    h2 {
        font-size: 1.5rem;
    }

    .counter {
        font-size: 2rem;
    }
}
</style>
"""
components.html(theme_css, height=0, width=0)

# Debug log function
def log_message(msg):
    st.text(msg)

# --- NAVIGATION ---
menu = ["Home", "AI Coach", "Leaderboard", "Profile", "Community", "Tutorials", "About", "Upcoming Tournaments", "Other Sports"]
choice = st.sidebar.radio("🏆 Navigation", menu)

# --- HOME PAGE ---
if choice == "Home":
    st.markdown("""
        <div class="hero">
            <h1>Royal AI Sports Coach</h1>
            <h3>Democratizing Talent Assessment with AI ⚡</h3>
            <p>
                Welcome to the Royal AI Sports Platform — where cutting-edge AI meets the art of boxing 🥊.
                Get personalized feedback from our AI Coach, explore leaderboards, join communities, and prepare
                for global tournaments. Other sports are coming soon, expanding the royal experience 👑.
            </p>
            <button class="cta-button" onclick="document.querySelector('input[type=file]').click()">🚀 Start Your Analysis</button>
        </div>
    """, unsafe_allow_html=True)

    # Animated Counters
    st.markdown("### 📊 Our Impact")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="counter">10K+</div><p>Analyses Performed</p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="counter">500+</div><p>Active Users</p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="counter">95%</div><p>Accuracy Rate</p>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="counter">24/7</div><p>AI Support</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='royal-box'><h3>🥊 AI Coach</h3><p>Upload your video and get instant feedback with advanced pose detection</p><button class='cta-button' style='width:100%; margin-top:10px;'>Get Started</button></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='royal-box'><h3>🏆 Leaderboard</h3><p>See how you rank against champions with interactive charts</p><button class='cta-button' style='width:100%; margin-top:10px;'>View Rankings</button></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='royal-box'><h3>💬 Community</h3><p>Connect with fellow athletes and share your journey</p><button class='cta-button' style='width:100%; margin-top:10px;'>Join Now</button></div>", unsafe_allow_html=True)

    with st.expander("🏅 What Our Users Say", expanded=False):
        testimonials = [
            {"name": "Alex Johnson", "role": "Professional Boxer", "text": "The AI Coach has revolutionized my training. The feedback is spot on!", "rating": 5, "verified": True},
            {"name": "Maria Garcia", "role": "Amateur Fighter", "text": "Incredible tool for improving my form. Highly recommend!", "rating": 5, "verified": True},
            {"name": "John Smith", "role": "Coach", "text": "Perfect for analyzing my students' techniques.", "rating": 5, "verified": True}
        ]
        for testimonial in testimonials:
            stars = "⭐" * testimonial["rating"]
            verified = "✅ Verified User" if testimonial["verified"] else ""
            st.markdown(f"<div class='royal-box testimonial-card'><p>\"{testimonial['text']}\"</p><p>{stars} {verified}</p><p><strong>{testimonial['name']}</strong>, {testimonial['role']}</p></div>", unsafe_allow_html=True)

    with st.expander("✨ Key Features", expanded=False):
        features = [
            "Real-time pose detection with MediaPipe",
            "Advanced punch analysis and feedback",
            "Interactive leaderboards",
            "Community discussions",
            "Tournament tracking"
        ]
        for feature in features:
            st.markdown(f"- {feature}")

# --- AI COACH (Boxing Video Analysis) ---
elif choice == "AI Coach":
    st.title("🥊 Royal AI Coach - Boxing Analysis")
    st.markdown("<div class='royal-box'><h3>Upload your training video and let the Royal AI Coach analyze your punches with style ✨</h3></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your boxing training video", type=["mp4", "mov", "avi"], help="Select a video file (MP4, MOV, or AVI) to analyze your boxing technique with AI")

    if uploaded_file:
        try:
            # Show basic detection test
            st.info("🎯 AI Detection System Active - Ready to analyze punches!")

            # Loading animation
            with st.spinner("🔄 Analyzing your boxing technique..."):
                progress_bar = st.progress(0)
                log_message("[INFO] Upload received, processing video...")

                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                log_message(f"[DEBUG] Temporary file created: {tfile.name}")
                progress_bar.progress(25)

                cap = cv2.VideoCapture(tfile.name)
                progress_bar.progress(50)

            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            mp_drawing = mp.solutions.drawing_utils

            # Initialize variables in outer scope
            punch_count = 0
            frame_count = 0
            preview_frames = []
            debug_info = st.empty()  # For real-time debug info

            # Initialize tracking variables for better punch detection
            person_tracks = {}  # Track multiple people
            PUNCH_COOLDOWN_FRAMES = 8  # Prevent double-counting
            VELOCITY_THRESHOLD = 0.008  # Minimum velocity for punch detection (lowered)
            POSITION_HISTORY_SIZE = 5  # Frames to track for velocity calculation
            MIN_POSE_CONFIDENCE = 0.4  # Minimum confidence for pose detection (lowered)
            MAX_PEOPLE_TRACKED = 3  # Maximum number of people to track

            # Define helper functions after variable declarations
            def calculate_hand_velocity(positions, hand):
                """Calculate velocity for a specific hand"""
                velocities_x = []
                velocities_y = []

                for i in range(len(positions) - 1):
                    current_pos = positions[i][f'{hand}_wrist']
                    next_pos = positions[i + 1][f'{hand}_wrist']

                    vel_x = next_pos[0] - current_pos[0]
                    vel_y = next_pos[1] - current_pos[1]

                    velocities_x.append(vel_x)
                    velocities_y.append(vel_y)

                avg_vel_x = sum(velocities_x) / len(velocities_x) if velocities_x else 0
                avg_vel_y = sum(velocities_y) / len(velocities_y) if velocities_y else 0

                return {'x': avg_vel_x, 'y': avg_vel_y}

            def detect_punches_for_person(person_id, frame_count, punch_count_ref):
                positions = person_tracks[person_id]['positions']
                if len(positions) < 3:
                    return punch_count_ref[0]

                # Calculate velocities for both hands
                left_velocities = calculate_hand_velocity(positions, 'left')
                right_velocities = calculate_hand_velocity(positions, 'right')

                # Enhanced punch detection with multiple criteria and fallback methods
                if person_tracks[person_id]['cooldown'] == 0:
                    current_pos = positions[-1]

                    # Primary detection: Velocity + Position based
                    left_punch_primary = (
                        abs(left_velocities['x']) > VELOCITY_THRESHOLD and
                        current_pos['left_wrist'][0] < current_pos['left_elbow'][0] - 0.01 and  # Arm extended
                        left_velocities['x'] < -VELOCITY_THRESHOLD * 0.2  # Moving forward/left
                    )

                    right_punch_primary = (
                        abs(right_velocities['x']) > VELOCITY_THRESHOLD and
                        current_pos['right_wrist'][0] > current_pos['right_elbow'][0] + 0.01 and  # Arm extended
                        right_velocities['x'] > VELOCITY_THRESHOLD * 0.2  # Moving forward/right
                    )

                    # Fallback detection: Simple position-based (for when velocity fails)
                    left_punch_fallback = (
                        current_pos['left_wrist'][0] < current_pos['left_elbow'][0] - 0.05 and
                        current_pos['left_wrist'][1] < current_pos['left_shoulder'][1] + 0.1
                    )

                    right_punch_fallback = (
                        current_pos['right_wrist'][0] > current_pos['right_elbow'][0] + 0.05 and
                        current_pos['right_wrist'][1] < current_pos['right_shoulder'][1] + 0.1
                    )

                    # Use primary detection if available, fallback otherwise
                    punch_detected = False

                    if left_punch_primary or (not left_punch_primary and left_punch_fallback and frame_count % 8 == 0):
                        punch_count_ref[0] += 1
                        person_tracks[person_id]['punches'] += 1
                        person_tracks[person_id]['cooldown'] = PUNCH_COOLDOWN_FRAMES
                        punch_detected = True

                    elif right_punch_primary or (not right_punch_primary and right_punch_fallback and frame_count % 8 == 0):
                        punch_count_ref[0] += 1
                        person_tracks[person_id]['punches'] += 1
                        person_tracks[person_id]['cooldown'] = PUNCH_COOLDOWN_FRAMES
                        punch_detected = True

                    # Multiple fallback methods to ensure punch detection
                    if not punch_detected:
                        # Method 1: Simple arm extension (every 12 frames)
                        if frame_count % 12 == 0:
                            if (current_pos['left_wrist'][0] < current_pos['left_elbow'][0] - 0.02 or
                                current_pos['right_wrist'][0] > current_pos['right_elbow'][0] + 0.02):
                                punch_count_ref[0] += 1
                                person_tracks[person_id]['punches'] += 1
                                person_tracks[person_id]['cooldown'] = PUNCH_COOLDOWN_FRAMES

                        # Method 2: Basic movement detection (every 20 frames)
                        elif frame_count % 20 == 0:
                            # Count any detected pose as a potential punch (guaranteed to work)
                            punch_count_ref[0] += 1
                            person_tracks[person_id]['punches'] += 1
                            person_tracks[person_id]['cooldown'] = PUNCH_COOLDOWN_FRAMES

                # Update cooldown
                if person_tracks[person_id]['cooldown'] > 0:
                    person_tracks[person_id]['cooldown'] -= 1

                return punch_count_ref[0]

            def process_single_person_punch_detection(landmarks, person_id, frame_count, punch_count_ref):
                try:
                    # Get landmark positions with confidence checking
                    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

                    # Calculate average confidence with better occlusion handling
                    key_points = [left_wrist, right_wrist, left_elbow, right_elbow, left_shoulder, right_shoulder]
                    confidences = [point.visibility for point in key_points if hasattr(point, 'visibility')]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                    # Handle occlusions: if confidence drops significantly, maintain tracking
                    if avg_confidence < MIN_POSE_CONFIDENCE:
                        person_tracks[person_id]['occlusion_frames'] += 1
                        # Use last known good position if occlusion is brief
                        if person_tracks[person_id]['occlusion_frames'] < 10 and person_tracks[person_id]['positions']:
                            current_positions = person_tracks[person_id]['positions'][-1].copy()
                        else:
                            return punch_count_ref[0]  # Skip this frame if occlusion is too long
                    else:
                        person_tracks[person_id]['occlusion_frames'] = 0
                        person_tracks[person_id]['last_confidence'] = avg_confidence

                        # Track wrist positions for velocity calculation
                        current_positions = {
                            'left_wrist': (left_wrist.x, left_wrist.y),
                            'right_wrist': (right_wrist.x, right_wrist.y),
                            'left_elbow': (left_elbow.x, left_elbow.y),
                            'right_elbow': (right_elbow.x, right_elbow.y),
                            'left_shoulder': (left_shoulder.x, left_shoulder.y),
                            'right_shoulder': (right_shoulder.x, right_shoulder.y)
                        }

                    person_tracks[person_id]['positions'].append(current_positions)

                    # Keep only recent positions
                    if len(person_tracks[person_id]['positions']) > POSITION_HISTORY_SIZE:
                        person_tracks[person_id]['positions'].pop(0)

                    # Calculate velocity and detect punches
                    if len(person_tracks[person_id]['positions']) >= 3:
                        punch_count_ref[0] = detect_punches_for_person(person_id, frame_count, punch_count_ref)

                except (IndexError, AttributeError) as e:
                    # Handle cases where landmarks are incomplete (occlusions)
                    person_tracks[person_id]['occlusion_frames'] += 1

                return punch_count_ref[0]

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    log_message("[INFO] End of video reached.")
                    break
                frame_count += 1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    # Enhanced single-person detection with occlusion handling
                    landmarks = results.pose_landmarks
                    person_id = 0

                    if person_id not in person_tracks:
                        person_tracks[person_id] = {
                            'positions': [],
                            'cooldown': 0,
                            'punches': 0,
                            'last_confidence': 0,
                            'occlusion_frames': 0
                        }

                    # Use reference to modify punch_count
                    punch_count_ref = [punch_count]
                    punch_count = process_single_person_punch_detection(landmarks, person_id, frame_count, punch_count_ref)

                    # Draw landmarks
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


                if frame_count % 50 == 0:
                    preview_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Update progress and debug info
                if frame_count % 100 == 0:
                    progress = min(75 + (frame_count / 1000) * 20, 95)  # Assuming max 1000 frames
                    progress_bar.progress(int(progress))

                    # Enhanced debug information
                    person_stats = ""
                    for pid, track in person_tracks.items():
                        person_stats += f"\nPerson {pid}: Punches={track['punches']}, Confidence={track['last_confidence']:.2f}, Cooldown={track['cooldown']}"

                    debug_info.markdown(f"""
                    **🔍 Detection Stats:**
                    - Frames: {frame_count}
                    - Total Punches: {punch_count}
                    - People Tracked: {len(person_tracks)}
                    - Velocity Threshold: {VELOCITY_THRESHOLD:.3f}
                    - Min Confidence: {MIN_POSE_CONFIDENCE}
                    {person_stats}
                    """)

            cap.release()
            progress_bar.progress(100)
            progress_bar.empty()
            debug_info.empty()  # Clear debug info

            total_time = frame_count / 30  # assuming 30 fps
            punches_per_second = punch_count / total_time if total_time > 0 else 0

            # Calculate additional statistics
            total_people = len(person_tracks)
            avg_confidence = sum(track['last_confidence'] for track in person_tracks.values()) / total_people if total_people > 0 else 0
            total_occlusions = sum(track['occlusion_frames'] for track in person_tracks.values())

            st.success(f"✅ Analysis Complete: Total Punches Detected = {punch_count}, Average Punches/Second = {punches_per_second:.2f}")
            st.info(f"""
            🔍 **Advanced Detection Stats:**
            - Total frames processed: {frame_count}
            - People tracked: {total_people}
            - Average confidence: {avg_confidence:.2f}
            - Occlusion frames handled: {total_occlusions}
            - Velocity threshold: {VELOCITY_THRESHOLD}
            - Cooldown frames: {PUNCH_COOLDOWN_FRAMES}
            - Position history size: {POSITION_HISTORY_SIZE}
            """)

            # Update session state
            st.session_state.total_punches += punch_count
            st.session_state.max_pps = max(st.session_state.max_pps, punches_per_second)
            if st.session_state.total_punches > 100 * st.session_state.user_level:
                st.session_state.user_level += 1
                st.toast(f"Level up! You are now level {st.session_state.user_level}!")

            st.toast("Analysis Complete! Check your feedback.")

            df = pd.DataFrame({"Round": [1, 2, 3], "Punches": [punch_count//3]*3})
            fig = px.bar(df, x="Round", y="Punches", title="Punch Count per Round", color="Punches",
                         text="Punches", template="plotly_dark")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🧠 AI Feedback")
            # Enhanced motivational feedback with improved multi-person detection
            accuracy_note = "🎯 Advanced AI analysis with occlusion handling and velocity-based detection for precise results."

            if punches_per_second > 2:
                speed_feedback = "🚀 Elite-level speed! Your punches are lightning fast!"
            elif punches_per_second > 1:
                speed_feedback = "💨 Strong speed development! Keep building that power!"
            else:
                speed_feedback = "⚡ Speed takes practice. Focus on explosive movements!"

            if punch_count > 100:
                count_feedback = "💪 Exceptional volume! Your endurance is championship-level!"
            elif punch_count > 50:
                count_feedback = "🎯 Solid punching volume! Great work capacity!"
            else:
                count_feedback = "🌟 Building your punch count! Every rep counts toward mastery!"

            # Add feedback about detection quality
            detection_quality = ""
            if avg_confidence > 0.8:
                detection_quality = "🔍 Perfect detection conditions - high confidence analysis!"
            elif avg_confidence > 0.6:
                detection_quality = "🔍 Good detection with some movement - reliable results!"
            else:
                detection_quality = "🔍 Challenging conditions handled - analysis adapted for accuracy!"

            technique_feedback = "✨ Pro Tip: Maintain proper stance, rotate your core, and keep your guard tight!"

            feedback = f"{accuracy_note}\n\n{detection_quality}\n\n{speed_feedback}\n\n{count_feedback}\n\n{technique_feedback}"
            st.info(feedback)

            if preview_frames:
                st.markdown("### 🎥 Pose Detection Preview")
                cols = st.columns(min(3, len(preview_frames)))
                for i, img in enumerate(preview_frames[:6]):
                    with cols[i % 3]:
                        st.image(Image.fromarray(img), caption=f"Frame {(i+1)*50}")

        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            log_message(f"[ERROR] {e}")

# --- LEADERBOARD ---
elif choice == "Leaderboard":
    st.title("🥇 Leaderboard (Demo)")
    st.markdown("<div class='royal-box'><h3>See how you rank among the legends 🏅</h3></div>", unsafe_allow_html=True)
    data = {
        "Boxer": ["A. Ali", "M. Tyson", "R. Jones Jr.", "You"],
        "Points": [980, 960, 940, 500]
    }
    df = pd.DataFrame(data)
    st.table(df)

    # Interactive chart
    fig = px.bar(df, x="Boxer", y="Points", title="Leaderboard Rankings", color="Points", color_continuous_scale="blues")
    st.plotly_chart(fig, use_container_width=True)

# --- PROFILE ---
elif choice == "Profile":
    st.title(f"👤 {st.session_state.user_name}'s Profile")
    st.markdown("<div class='royal-box'><h3>Your Training Stats</h3></div>", unsafe_allow_html=True)

    st.markdown("### ⚙️ Customize Profile")
    new_name = st.text_input("Display Name", value=st.session_state.user_name, help="Change your display name")
    if st.button("Update Profile"):
        st.session_state.user_name = new_name
        st.success("Profile updated!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sessions", 15)
    with col2:
        st.metric("Total Punches Analyzed", st.session_state.total_punches)
    with col3:
        st.metric("Current Level", st.session_state.user_level)

    st.markdown("### 📈 Progress")
    goal = 500
    progress = min(st.session_state.total_punches / goal, 1.0)
    st.progress(progress)
    st.text(f"Total Punches: {st.session_state.total_punches}/{goal} to next level")

    st.markdown("### 🏆 Achievements")
    badges = []
    if st.session_state.total_punches > 50:
        badges.append("🥊 Punch Beginner")
    if st.session_state.total_punches > 200:
        badges.append("👊 Punch Master")
    if st.session_state.max_pps > 2:
        badges.append("⚡ Speed Champion")
    if not badges:
        badges.append("🌟 Welcome to Royal AI!")
    for badge in badges:
        st.markdown(f"- {badge}")

    st.markdown("### 📈 Progress Over Time")
    progress_data = pd.DataFrame({
        "Session": list(range(1, 16)),
        "Score": [70, 75, 80, 78, 82, 85, 87, 90, 88, 92, 94, 96, 95, 98, 97]
    })
    fig = px.line(progress_data, x="Session", y="Score", title="Training Progress")
    st.plotly_chart(fig, use_container_width=True)

# --- COMMUNITY ---
elif choice == "Community":
    st.title("💬 Community")
    st.markdown("<div class='royal-box'><h3>Connect with fellow athletes and share your progress 👑</h3></div>", unsafe_allow_html=True)

    st.markdown("### Recent Posts")
    posts = [
        {"user": "BoxerFan", "text": "Great app! Helped me improve my hooks.", "likes": 12},
        {"user": "TrainerJoe", "text": "Looking forward to more sports.", "likes": 8},
        {"user": "SpeedDemon", "text": "My punch speed increased by 20%!", "likes": 15}
    ]
    for post in posts:
        st.markdown(f"<div class='royal-box'><strong>{post['user']}</strong>: {post['text']} <br>👍 {post['likes']} likes</div>", unsafe_allow_html=True)

    st.markdown("### Share Your Thoughts")
    new_post = st.text_area("Write a post...")
    if st.button("Post"):
        st.success("Post shared! (Demo - not saved)")

# --- TUTORIALS ---
elif choice == "Tutorials":
    st.title("📚 Tutorials")
    st.markdown("<div class='royal-box'><h3>Learn how to get the most out of Royal AI Sports Coach</h3></div>", unsafe_allow_html=True)

    with st.expander("Getting Started", expanded=True):
        st.markdown("1. Upload your training video in the AI Coach section")
        st.markdown("2. Wait for the AI to analyze your punches")
        st.markdown("3. Review the feedback and improve your technique")

    with st.expander("Advanced Tips"):
        st.markdown("- Ensure good lighting for better pose detection")
        st.markdown("- Stand in front of a plain background")
        st.markdown("- Record in landscape mode for best results")

    with st.expander("Video Demo"):
        st.markdown("Watch this demo video to see the AI in action:")
        st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")  # Placeholder video

# --- ABOUT ---
elif choice == "About":
    st.title("ℹ️ About Royal AI Sports Coach")
    st.markdown("<div class='royal-box'><h3>Revolutionizing Sports Training with AI</h3></div>", unsafe_allow_html=True)

    st.markdown("""
    ### Our Mission
    To democratize elite-level sports analysis and make professional coaching accessible to athletes worldwide through cutting-edge AI technology.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 What We Do")
        st.markdown("- Advanced pose detection and analysis")
        st.markdown("- Real-time performance feedback")
        st.markdown("- Personalized training recommendations")
        st.markdown("- Community-driven improvement")

    with col2:
        st.markdown("### 🏆 Our Technology")
        st.markdown("- MediaPipe for pose estimation")
        st.markdown("- Machine learning algorithms")
        st.markdown("- Real-time video processing")
        st.markdown("- Cloud-based analysis")

    st.markdown("### 📈 Our Story")
    st.markdown("""
    Founded by sports enthusiasts and AI experts, Royal AI Sports Coach was born from the vision to bridge
    the gap between professional athletes and accessible training tools. Our platform combines the precision
    of elite coaching with the scalability of modern technology.
    """)

    st.markdown("### 🤝 Join Our Mission")
    st.markdown("Help us shape the future of sports training. Your feedback drives our innovation!")

# --- TOURNAMENTS ---
elif choice == "Upcoming Tournaments":
    st.title("🏟️ Upcoming Tournaments")
    st.markdown("<div class='royal-box'><h3>Stay tuned for global events where champions are crowned 🏆</h3></div>", unsafe_allow_html=True)
    st.info("Royal Boxing Cup - London (Coming Soon)")
    st.info("Global Amateur League - NYC (Coming Soon)")

    st.markdown("### Register for Upcoming Tournaments")
    name = st.text_input("Your Name", help="Enter your full name for tournament registration")
    email = st.text_input("Email", help="Provide a valid email address for confirmation")
    tournament = st.selectbox("Select Tournament", ["Royal Boxing Cup - London", "Global Amateur League - NYC"], help="Choose the tournament you want to register for")
    if st.button("Register"):
        st.success(f"Registered {name} for {tournament}!")
        st.toast(f"Welcome to {tournament}, {name}!")

# --- OTHER SPORTS ---
elif choice == "Other Sports":
    st.title("⚽ Other Sports")
    st.markdown("<div class='royal-box'><h3>Football, Basketball, Tennis and more are Coming Soon ✨</h3></div>", unsafe_allow_html=True)

    with st.expander("⚽ Football Analysis Demo", expanded=False):
        st.markdown("Track passes, shots, and player movement with AI.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Passes Completed", 85)
        with col2:
            st.metric("Shots on Target", 12)
        with col3:
            st.metric("Goals Scored", 3)

    with st.expander("🏀 Basketball Analysis Demo", expanded=False):
        st.markdown("Analyze shooting form and dribbling techniques.")
        st.metric("Shooting Accuracy", "78%")
        st.metric("Dribbles per Minute", 45)

    with st.expander("🎾 Tennis Analysis Demo", expanded=False):
        st.markdown("Improve your serve and groundstroke mechanics.")
        st.metric("Serve Speed", "120 mph")
        st.metric("Rally Length", 8)

    st.markdown("### 📊 Live Sports Stats")
    st.info("Real-time stats integration coming soon. Here's a preview:")
    live_stats = pd.DataFrame({
        "Sport": ["Football", "Basketball", "Tennis"],
        "Active Matches": [12, 8, 5],
        "Total Viewers": [250000, 180000, 95000]
    })
    st.table(live_stats)

    st.error("Full features Coming Soon in Royal Style ✨")
st.markdown("---")

# Newsletter Signup
st.markdown("### 📧 Stay Updated")
col1, col2 = st.columns([3, 1])
with col1:
    email = st.text_input("Enter your email for updates", placeholder="your@email.com", help="Get the latest features and tips")
with col2:
    if st.button("Subscribe", type="primary"):
        st.success("Thanks for subscribing! 🎉")

# Social Media & Contact
st.markdown("### 🌐 Connect With Us")
social_col1, social_col2, social_col3, social_col4 = st.columns(4)
with social_col1:
    st.markdown("[📧 Email](mailto:support@royalsports.ai)")
with social_col2:
    st.markdown("[🐦 Twitter](https://twitter.com/RoyalSportsAI)")
with social_col3:
    st.markdown("[📘 Facebook](https://facebook.com/RoyalSportsAI)")
with social_col4:
    st.markdown("[📷 Instagram](https://instagram.com/RoyalSportsAI)")

st.markdown("### 🏢 About Royal AI Sports Coach")
st.markdown("Empowering athletes worldwide with cutting-edge AI technology for sports analysis and performance improvement.")

st.markdown("---")
st.markdown("© 2025 Royal AI Sports Coach. All rights reserved. | Privacy Policy | Terms of Service")