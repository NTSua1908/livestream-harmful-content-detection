"""
Streamlit Dashboard for Real-time Harmful Content Detection Monitoring
Upgraded Version: Includes Image Evidence, Working Time Filters, and Grid Layout.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import time
import cv2
import numpy as np
from utils import MongoDBHandler, decode_base64_to_image

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Harmful Content Monitor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-card {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 5px solid #ccc;
        color: #1a1a1a;
        font-weight: 500;
    }
    .alert-HIGH { 
        background-color: #ffebee; 
        border-color: #f44336;
        color: #c62828;
    }
    .alert-MEDIUM { 
        background-color: #fff3e0; 
        border-color: #ff9800;
        color: #e65100;
    }
    .alert-LOW { 
        background-color: #fffde7; 
        border-color: #fbc02d;
        color: #f57f17;
    }
    
    /* Make the alert level text more visible */
    .alert-card strong {
        font-size: 1.1em;
    }
    
    /* Image Grid Styling */
    .stImage { border-radius: 5px; }
</style>
""",
    unsafe_allow_html=True,
)


# --- 2. HELPER FUNCTIONS ---


@st.cache_resource
def get_db_handler():
    """Get MongoDB handler (cached to avoid reconnecting)"""
    return MongoDBHandler()


def format_timestamp(ts):
    """Format timestamp for display"""
    if not ts:
        return "N/A"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def get_start_time(time_range_str):
    """Convert dropdown selection to actual timestamp"""
    now = datetime.now()
    if time_range_str == "Last 1 hour":
        return (now - timedelta(hours=1)).timestamp()
    elif time_range_str == "Last 6 hours":
        return (now - timedelta(hours=6)).timestamp()
    elif time_range_str == "Last 24 hours":
        return (now - timedelta(hours=24)).timestamp()
    elif time_range_str == "Last 7 days":
        return (now - timedelta(days=7)).timestamp()
    elif time_range_str == "All Time":
        return 0  # Show all data from beginning
    return (now - timedelta(hours=1)).timestamp()  # Default


def display_alert_row(alert):
    """Render a single alert using HTML/CSS"""
    level = alert.get("type", "LOW")
    ts = format_timestamp(alert.get("timestamp"))
    details = alert.get("details", "")
    type_ = alert.get("detection_type", "Unknown")
    conf = alert.get("confidence", 0)

    st.markdown(
        f"""
        <div class="alert-card alert-{level}">
            <strong>🚨 [{level}] {type_}</strong> <span style="float:right">{ts}</span><br>
            <small>Confidence: {conf:.1%}</small><br>
            {details}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- 3. MAIN DASHBOARD LOGIC ---


def convert_to_timestamp(ts):
    """Convert timestamp to float if it's a datetime object"""
    if isinstance(ts, datetime):
        return ts.timestamp()
    return float(ts) if ts else 0


def main():
    st.title("🚨 Livestream Security Monitor")
    st.markdown("Real-time AI analysis for harmful content detection")

    # --- SIDEBAR: SETTINGS ---
    st.sidebar.title("⚙️ Configuration")

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 3, 60, 5)

    # Filters
    st.sidebar.subheader("Filters")
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days", "All Time"],
        index=4,  # Default to "Last 24 hours"
    )
    selected_levels = st.sidebar.multiselect(
        "Alert Levels", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"]
    )

    # --- DATA FETCHING ---
    try:
        db = get_db_handler()
    except Exception as e:
        st.error(f"❌ Database Error: {e}")
        st.stop()

    # Calculate timestamps
    start_ts = get_start_time(time_range)

    # Fetch data (Fetching a bit more to filter in Python for simplicity)
    raw_detections = db.get_recent_detections(limit=1000)
    raw_alerts = db.get_recent_alerts(limit=500)

    detections = [
        d
        for d in raw_detections
        if convert_to_timestamp(d.get("timestamp", 0)) >= start_ts
    ]

    alerts = [
        a
        for a in raw_alerts
        if convert_to_timestamp(a.get("timestamp", 0)) >= start_ts
        and a.get("type", "LOW") in selected_levels
    ]

    # --- TABS LAYOUT ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Overview", "🚨 Alerts Log", "📹 Video Evidence", "🎤 Audio Analysis", "📋 Detection Report"]
    )

    # === TAB 1: OVERVIEW ===
    with tab1:
        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)

        harmful_count = len([d for d in detections if d.get("is_harmful")])
        high_priority = len([a for a in alerts if a.get("type") == "HIGH"])

        col1.metric("Total Frames Analyzed", len(detections))
        col2.metric("Harmful Detections", harmful_count)
        col3.metric("Total Alerts", len(alerts))
        col4.metric("High Priority", high_priority)

        st.divider()

        # Charts
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Alerts Timeline")
            if alerts:
                df_alerts = pd.DataFrame(alerts)
                df_alerts["datetime"] = pd.to_datetime(df_alerts["timestamp"], unit="s")
                fig = px.scatter(
                    df_alerts,
                    x="datetime",
                    y="confidence",
                    color="type",
                    symbol="detection_type",
                    color_discrete_map={
                        "HIGH": "red",
                        "MEDIUM": "orange",
                        "LOW": "gold",
                    },
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No alerts in this period.")

        with c2:
            st.subheader("Detection Distribution")
            if alerts:
                counts = {}
                for a in alerts:
                    t = a.get("detection_type", "Unknown")
                    counts[t] = counts.get(t, 0) + 1

                fig = px.pie(
                    names=list(counts.keys()), values=list(counts.values()), hole=0.4
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No data available.")

    # === TAB 2: ALERTS LOG ===
    with tab2:
        st.subheader("Live Alert Feed")
        if alerts:
            for alert in alerts[:20]:  # Show latest 20
                display_alert_row(alert)

            if len(alerts) > 20:
                st.caption(f"... and {len(alerts) - 20} more alerts.")
        else:
            st.success("No alerts matching filters.")

    # === TAB 3: VIDEO EVIDENCE (Major Upgrade) ===
    with tab3:
        st.subheader("📸 Harmful Content Detection Results - Video Evidence")

        harmful_frames = [d for d in detections if d.get("is_harmful")]

        if harmful_frames:
            st.warning(f"Found {len(harmful_frames)} frames with harmful content.")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            # Count detections by type
            detection_types = {}
            for frame in harmful_frames:
                for det in frame.get("detections", []):
                    det_type = det.get("class", "Unknown")
                    detection_types[det_type] = detection_types.get(det_type, 0) + 1
            
            col1.metric("Total Detections", len(harmful_frames))
            col2.metric("Detection Types Found", len(detection_types))
            col3.metric("Time Range", time_range)
            
            # Display detection breakdown
            st.markdown("**Detection Breakdown:**")
            for det_type, count in detection_types.items():
                st.write(f"- {det_type}: {count} detections")
            
            st.divider()

            # Grid Layout (3 Columns) for image display
            cols = st.columns(3)

            # Show latest 30 frames to avoid memory issues
            for idx, item in enumerate(harmful_frames[:30]):
                col = cols[idx % 3]  # Distribute items across columns

                with col:
                    with st.container(border=True):
                        # Try to decode and display image (preferring annotated frame)
                        img_displayed = False
                        
                        # First try to show annotated frame with boxes
                        if "data" in item:
                            try:
                                img = decode_base64_to_image(item["data"])
                                if img is not None:
                                    st.image(img, channels="BGR", width='stretch')
                                    img_displayed = True
                            except Exception as e:
                                pass
                        
                        # If annotated not available, try original
                        if not img_displayed and "original_data" in item:
                            try:
                                img = decode_base64_to_image(item["original_data"])
                                if img is not None:
                                    st.image(img, channels="BGR", width='stretch')
                                    img_displayed = True
                            except Exception as e:
                                pass
                        
                        # If still no image, create placeholder with detection info
                        if not img_displayed:
                            # Create a placeholder image showing detection info
                            placeholder = np.ones((400, 400, 3), dtype=np.uint8) * 200
                            
                            # Add text on placeholder
                            cv2.putText(placeholder, "No Image Data", (50, 150),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                            cv2.putText(placeholder, "Available", (80, 200),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                            
                            # Add detection count
                            det_count = len(item.get("detections", []))
                            cv2.putText(placeholder, f"Detections: {det_count}", (50, 280),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                            
                            # Display placeholder
                            st.image(placeholder, channels="BGR", width='stretch')
                            img_displayed = True

                        # Meta info
                        ts = format_timestamp(item.get("timestamp"))
                        st.markdown(f"**Time:** {ts}")
                        
                        # Frame ID
                        st.markdown(f"**Frame ID:** {item.get('frame_id', 'N/A')}")

                        # List Detections with details
                        st.markdown("**Detections:**")
                        for det in item.get("detections", []):
                            det_type = det.get('class', 'Unknown')
                            confidence = det.get('confidence', 0)
                            bbox = det.get('bbox', None)
                            
                            st.markdown(
                                f"🔴 **{det_type}**\n"
                                f"- Confidence: {confidence:.1%}\n"
                                f"- Bounding Box: {bbox if bbox else 'N/A'}"
                            )
        else:
            st.success("✅ No harmful video frames detected in this period.")

    # === TAB 4: AUDIO ANALYSIS ===
    with tab4:
        st.subheader("� Audio & Speech Analysis - Detection Results")

        # Lấy dữ liệu audio (có chunk_id) và lọc theo thời gian
        audio_events = [
            d
            for d in raw_detections
            if "chunk_id" in d
            and convert_to_timestamp(d.get("timestamp", 0)) >= start_ts
        ]

        if audio_events:
            # Sắp xếp mới nhất lên đầu
            audio_events.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            
            harmful_sounds = [
                "Screaming",
                "Yelling",
                "Explosion",
                "Gunshot, gunfire",
                "Bang",
            ]
            
            dangerous_sound_count = len([
                e for e in audio_events 
                if (e.get("sound_label") in harmful_sounds and e.get("sound_confidence", 0) > 0.3)
                or e.get("is_screaming", False)
            ])
            
            toxic_speech_count = len([e for e in audio_events if e.get("is_toxic", False)])
            
            col1.metric("Total Audio Chunks", len(audio_events))
            col2.metric("Dangerous Sounds Detected", dangerous_sound_count)
            col3.metric("Toxic Speech Detected", toxic_speech_count)
            col4.metric("Time Range", time_range)
            
            st.divider()

            # Display high-priority audio events first
            st.markdown("### 🚨 High Priority Audio Events")
            
            high_priority_shown = False
            
            for item in audio_events[:50]:
                timestamp = format_timestamp(item.get("timestamp"))
                text = item.get("transcribed_text", "")
                is_toxic = item.get("is_toxic", False)
                sound_label = item.get("sound_label", "Speech")
                sound_conf = item.get("sound_confidence", 0.0)
                is_screaming = item.get("is_screaming", False)

                harmful_sounds = [
                    "Screaming",
                    "Yelling",
                    "Explosion",
                    "Gunshot, gunfire",
                    "Bang",
                ]

                # --- CASE A: ÂM THANH NGUY HIỂM (TIẾNG NỔ, SÚNG, HÉT) ---
                if is_screaming or (sound_label in harmful_sounds and sound_conf > 0.3):
                    high_priority_shown = True
                    st.markdown(
                        f"""
                    <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; border-left: 6px solid #f44336; margin-bottom: 15px;">
                        <h4 style="color: #c62828; margin:0;">🔊 DANGER SOUND DETECTED</h4>
                        <strong style="color: #c62828;">Sound Type:</strong> {sound_label}<br>
                        <strong style="color: #c62828;">Confidence:</strong> {sound_conf:.1%}<br>
                        <strong>Detected at:</strong> {timestamp}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # --- CASE B: LỜI NÓI ĐỘC HẠI ---
                if is_toxic:
                    high_priority_shown = True
                    st.markdown(
                        f"""
                    <div style="background-color: #fff3e0; padding: 15px; border-radius: 8px; border-left: 6px solid #ff9800; margin-bottom: 15px;">
                        <h4 style="color: #e65100; margin:0;">🤬 TOXIC SPEECH DETECTED</h4>
                        <strong>Detected at:</strong> {timestamp}<br>
                        <strong style="color: #e65100;">Content:</strong> <em>"{text}"</em>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            
            if not high_priority_shown:
                st.info("✅ No high-priority audio events detected in this period.")
            
            st.divider()
            
            # Show normal audio logs
            st.markdown("### 📊 All Audio Events Log")
            
            with st.expander("📖 Click to expand full audio log", expanded=False):
                for item in audio_events[:50]:
                    timestamp = format_timestamp(item.get("timestamp"))
                    text = item.get("transcribed_text", "")
                    is_toxic = item.get("is_toxic", False)
                    sound_label = item.get("sound_label", "Speech")
                    sound_conf = item.get("sound_confidence", 0.0)
                    is_screaming = item.get("is_screaming", False)
                    
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            st.write(f"**Time:** {timestamp}")
                            st.write(f"**Sound:** {sound_label}")
                            st.write(f"**Conf:** {sound_conf:.1%}")
                        
                        with col2:
                            if is_toxic:
                                st.error(f"🤬 Toxic: {text}")
                            elif is_screaming:
                                st.warning(f"🔊 Screaming detected")
                            else:
                                st.write(f"📝 {text if text else '(No speech)'}")

        else:
            st.info("No audio analysis data found in the selected period.")

    # === TAB 5: DETECTION REPORT - Per-Livestream Detection Summary ===
    with tab5:
        st.subheader("📋 Livestream Detection Report - Per-Video Harmful Content")
        st.markdown("Grouped harmful content detection by livestream/video source")

        harmful_frames = [d for d in detections if d.get("is_harmful")]

        if harmful_frames:
            # === Group detections by video/livestream ===
            # Using frame_id as video identifier (can be extended for actual video_id)
            videos = {}  # {video_id: [frames with detections]}
            
            for frame in harmful_frames:
                # Extract video identifier from frame_id or use timestamp as fallback
                video_id = frame.get("frame_id", "unknown")
                if video_id == -1 or video_id == "unknown":
                    # Fallback: group by hour
                    ts = frame.get("timestamp", 0)
                    video_id = f"Stream_{datetime.fromtimestamp(ts).strftime('%Y%m%d_%H')}"
                else:
                    video_id = f"Video_{video_id}"
                
                if video_id not in videos:
                    videos[video_id] = []
                videos[video_id].append(frame)
            
            # === Overall Statistics ===
            st.markdown("### 📊 Overall Detection Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Total Harmful Videos", len(videos))
            col2.metric("Total Harmful Frames", len(harmful_frames))
            
            total_detections = sum(len(f.get("detections", [])) for f in harmful_frames)
            col3.metric("Total Detections", total_detections)
            
            all_confidences = []
            for frame in harmful_frames:
                for det in frame.get("detections", []):
                    all_confidences.append(det.get("confidence", 0) * 100)
            avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            col4.metric("Avg. Confidence", f"{avg_conf:.1f}%")
            
            st.divider()
            
            # === Per-Video Report ===
            st.markdown("### 🎥 Harmful Content by Livestream/Video")
            
            # Sort videos by number of detections (most harmful first)
            sorted_videos = sorted(videos.items(), key=lambda x: len(x[1]), reverse=True)
            
            for video_idx, (video_id, frames_in_video) in enumerate(sorted_videos, 1):
                # Collect all labels in this video
                video_labels = {}  # {label: count}
                video_detections = []
                video_timestamps = []
                
                for frame in frames_in_video:
                    video_timestamps.append(frame.get("timestamp", 0))
                    for det in frame.get("detections", []):
                        label = det.get("class", "Unknown")
                        confidence = det.get("confidence", 0)
                        video_labels[label] = video_labels.get(label, 0) + 1
                        video_detections.append({
                            "label": label,
                            "confidence": confidence,
                            "frame_time": format_timestamp(frame.get("timestamp"))
                        })
                
                # Create expandable section for each video
                with st.expander(
                    f"🎥 [{video_idx}] {video_id} - {len(frames_in_video)} frames, {len(video_detections)} detections",
                    expanded=(video_idx == 1)  # Expand first video by default
                ):
                    # Video-level metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric("Frames with Detection", len(frames_in_video))
                    col2.metric("Total Detections", len(video_detections))
                    col3.metric("Unique Labels", len(video_labels))
                    
                    video_confs = [d.get("confidence", 0) * 100 for f in frames_in_video for d in f.get("detections", [])]
                    video_avg_conf = sum(video_confs) / len(video_confs) if video_confs else 0
                    col4.metric("Avg. Confidence", f"{video_avg_conf:.1f}%")
                    
                    st.markdown("---")
                    
                    # === Labels Detected in This Video ===
                    st.markdown("**🏷️ Labels Detected in This Video:**")
                    
                    label_data = []
                    for label, count in sorted(video_labels.items(), key=lambda x: x[1], reverse=True):
                        # Calculate avg confidence for this label in this video
                        label_confs = [d.get("confidence", 0) * 100 
                                      for d in video_detections 
                                      if d["label"] == label]
                        label_avg_conf = sum(label_confs) / len(label_confs) if label_confs else 0
                        
                        label_data.append({
                            "🏷️ Label": label,
                            "Count": count,
                            "Avg Confidence": f"{label_avg_conf:.1f}%",
                            "Max Confidence": f"{max(label_confs):.1f}%"
                        })
                    
                    df_labels = pd.DataFrame(label_data)
                    st.dataframe(df_labels, width='stretch', hide_index=True)
                    
                    st.markdown("---")
                    
                    # === Detection Timeline in This Video ===
                    st.markdown("**📹 Detection Timeline:**")
                    
                    timeline_data = []
                    for det in video_detections:
                        timeline_data.append({
                            "Time": det["frame_time"],
                            "Label": det["label"],
                            "Confidence": f"{det['confidence']:.1%}"
                        })
                    
                    df_timeline = pd.DataFrame(timeline_data)
                    st.dataframe(df_timeline, width='stretch', hide_index=True)
                    
                    st.markdown("---")
                    
                    # === Sample Images from This Video ===
                    st.markdown("**📸 Sample Detected Frames (max 6):**")
                    
                    # Show up to 6 sample frames from this video
                    sample_frames = frames_in_video[:6]
                    cols = st.columns(min(3, len(sample_frames)))
                    
                    for frame_idx, frame in enumerate(sample_frames):
                        col = cols[frame_idx % len(cols)]
                        with col:
                            with st.container(border=True):
                                # Try to show image
                                img_displayed = False
                                
                                if "data" in frame:
                                    try:
                                        img = decode_base64_to_image(frame["data"])
                                        if img is not None:
                                            st.image(img, channels="BGR", width='stretch')
                                            img_displayed = True
                                    except:
                                        pass
                                
                                if not img_displayed and "original_data" in frame:
                                    try:
                                        img = decode_base64_to_image(frame["original_data"])
                                        if img is not None:
                                            st.image(img, channels="BGR", width='stretch')
                                            img_displayed = True
                                    except:
                                        pass
                                
                                # If still no image, create placeholder
                                if not img_displayed:
                                    placeholder = np.ones((300, 300, 3), dtype=np.uint8) * 200
                                    cv2.putText(placeholder, "No Image Data", (30, 120),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                    cv2.putText(placeholder, "Available", (60, 160),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                    det_count = len(frame.get("detections", []))
                                    cv2.putText(placeholder, f"Det: {det_count}", (80, 220),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)
                                    st.image(placeholder, channels="BGR", width='stretch')
                                
                                # Frame info
                                st.caption(f"⏰ {format_timestamp(frame.get('timestamp'))}")
                                st.caption(f"🎯 {len(frame.get('detections', []))} detection(s)")
        
        else:
            st.success("✅ No harmful video frames detected in this period - System is operating normally!")

    # --- FOOTER & REFRESH ---
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
