"""
Streamlit Dashboard for Real-time Harmful Content Detection Monitoring
Upgraded Version: Includes Image Evidence, Working Time Filters, and Grid Layout.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import time
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
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "🚨 Alerts Log", "📹 Video Evidence", "🎤 Audio Analysis"]
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
        st.subheader("📸 Harmful Content Gallery")

        harmful_frames = [d for d in detections if d.get("is_harmful") and "data" in d]

        if harmful_frames:
            st.warning(f"Found {len(harmful_frames)} frames with harmful content.")

            # Grid Layout (3 Columns)
            cols = st.columns(3)

            # Show latest 30 frames to avoid memory issues
            for idx, item in enumerate(harmful_frames[:30]):
                col = cols[idx % 3]  # Distribute items across columns

                with col:
                    with st.container(border=True):
                        # Decode & Display Image
                        try:
                            img = decode_base64_to_image(item["data"])
                            if img is not None:
                                # OpenCV is BGR, Streamlit needs RGB/BGR specified
                                # Use channels="BGR" to let Streamlit know format
                                st.image(img, channels="BGR", width="stretch")
                            else:
                                st.error("Image decode failed")
                        except Exception:
                            st.error("Image error")

                        # Meta info
                        ts = format_timestamp(item.get("timestamp"))
                        st.markdown(f"**Time:** {ts}")

                        # List Detections
                        for det in item.get("harmful_detections", []):
                            st.markdown(
                                f"🔴 **{det.get('class')}**: {det.get('confidence', 0):.1%}"
                            )
        else:
            st.success("No harmful video frames detected in this period.")

    # === TAB 4: AUDIO ANALYSIS ===
    with tab4:
        st.subheader("🎙️ Audio & Speech Analysis")

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

            for item in audio_events[:30]:  # Hiển thị 30 mẫu mới nhất
                # 1. Lấy thông tin từ DB
                timestamp = format_timestamp(item.get("timestamp"))

                # Thông tin Text (Toxic)
                text = item.get("transcribed_text", "")
                is_toxic = item.get("is_toxic", False)

                # Thông tin Âm thanh (Screaming, Explosion...)
                sound_label = item.get("sound_label", "Speech")
                sound_conf = item.get("sound_confidence", 0.0)
                is_screaming = item.get("is_screaming", False)  # Flag từ consumer

                # 2. Xử lý hiển thị

                # --- CASE A: ÂM THANH NGUY HIỂM (TIẾNG NỔ, SÚNG, HÉT) ---
                # Kiểm tra flag is_screaming hoặc check thủ công label
                harmful_sounds = [
                    "Screaming",
                    "Yelling",
                    "Explosion",
                    "Gunshot, gunfire",
                    "Bang",
                ]

                if is_screaming or (sound_label in harmful_sounds and sound_conf > 0.3):
                    st.markdown(
                        f"""
                    <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; border-left: 6px solid #f44336; margin-bottom: 15px;">
                        <h4 style="color: #c62828; margin:0;">🔊 DANGER SOUND: {sound_label}</h4>
                        <span style="font-size: 0.9em; color: #555;">Detected at: {timestamp}</span><br>
                        <strong style="color: #c62828;">Confidence:</strong> <span style="color: #c62828;"> {sound_conf:.1%} </span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # --- CASE B: LỜI NÓI ĐỘC HẠI ---
                if is_toxic:
                    st.error(
                        f'🤬 **Toxic Speech Detected** ({timestamp})\n\n> "{text}"'
                    )

                # --- CASE C: BÌNH THƯỜNG (Ẩn bớt để đỡ rối) ---
                # Chỉ hiện nếu không phải nguy hiểm và có text
                elif not is_screaming and not is_toxic:
                    with st.expander(f"ℹ️ Clean Audio Log - {timestamp}"):
                        st.markdown(f"**Sound:** {sound_label} ({sound_conf:.1%})")
                        st.markdown(f"**Transcript:** *{text}*")

        else:
            st.info("No audio analysis data found in the selected period.")

    # --- FOOTER & REFRESH ---
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
