"""
Waste Detection Streamlit App
"""

import os
import requests
from datetime import datetime, timedelta
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

API_URL = os.getenv('API_URL', 'http://api:8000')

def get_models():
    """Fetch available models from API"""
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        return resp.json()
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []

def get_history():
    """Fetch detection history"""
    try:
        resp = requests.get(f"{API_URL}/history", timeout=10)
        return resp.json()
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return []

def predict(image_data, lat, lon, model_name):
    """Send prediction request"""
    try:
        files = {'file': ('image.jpg', image_data, 'image/jpeg')}
        data = {
            'latitude': str(lat),
            'longitude': str(lon),
            'model_name': model_name
        }
        resp = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=30)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Page config
st.set_page_config(
    page_title="Waste Detection - Drone Patrol",
    page_icon="🗑️",
    layout="wide"
)

st.title("🗑️ Waste Detection - Drone Patrol System")
st.markdown("Upload images to detect waste deposits or view drone patrol data on the map.")

# Sidebar
st.sidebar.header("Settings")

# Model selection
models = get_models()
model_names = [m['name'] for m in models] if models else ['yolov8']
selected_model = st.sidebar.selectbox("Select Model", model_names)

# Show model info
if models:
    model_info = next((m for m in models if m['name'] == selected_model), None)
    if model_info:
        st.sidebar.info(f"Version: {model_info.get('version', 'N/A')}")

# Tabs
upload_tab, map_tab = st.tabs(["📤 Upload & Predict", "🗺️ Detection Map"])

# Upload tab
with upload_tab:
    st.subheader("Upload Image for Waste Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    with col2:
        st.markdown("### GPS Coordinates")
        lat = st.number_input("Latitude", value=48.8566, format="%.6f")
        lon = st.number_input("Longitude", value=2.3522, format="%.6f")
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Run Detection"):
            with st.spinner("Processing..."):
                image_data = uploaded_file.getvalue()
                result = predict(image_data, lat, lon, selected_model)
                
                if result:
                    st.success("Detection complete!")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Waste Detected", "Yes" if result.get('rubbish') else "No")
                    col2.metric("Confidence", f"{result.get('confiance', 0):.2%}")
                    col3.metric("Model Used", result.get('model_used', 'N/A'))

# Map tab
with map_tab:
    st.subheader("Detection Map")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        source_filter = st.selectbox("Source", ["All", "manual", "drone_patrol"])
    with col2:
        time_filter = st.selectbox("Time Period", ["All Time", "Last 24h", "Last 7 days", "Last 30 days"])
    with col3:
        model_filter = st.selectbox("Model", ["All"] + model_names)
    
    # Get history
    history = get_history()
    
    if history:
        # Apply filters
        filtered = history
        
        if source_filter != "All":
            filtered = [h for h in filtered if h.get('source') == source_filter]
        
        if time_filter != "All Time":
            now = datetime.utcnow()
            if time_filter == "Last 24h":
                cutoff = now - timedelta(days=1)
            elif time_filter == "Last 7 days":
                cutoff = now - timedelta(days=7)
            else:
                cutoff = now - timedelta(days=30)
            filtered = [h for h in filtered if datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) > cutoff]
        
        if model_filter != "All":
            filtered = [h for h in filtered if h.get('model_name') == model_filter]
        
        # Create map
        if filtered:
            center_lat = sum(h['latitude'] for h in filtered) / len(filtered)
            center_lon = sum(h['longitude'] for h in filtered) / len(filtered)
        else:
            center_lat, center_lon = 48.8566, 2.3522
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        marker_cluster = MarkerCluster().add_to(m)
        
        for detection in filtered:
            is_manual = detection.get('source') == 'manual'
            color = 'red' if is_manual else 'orange'
            icon = 'trash' if is_manual else 'plane'
            
            popup_text = f"""
            <b>Source:</b> {detection.get('source', 'N/A')}<br>
            <b>Model:</b> {detection.get('model_name', 'N/A')}<br>
            <b>Confidence:</b> {detection.get('confiance', 0):.2%}<br>
            <b>Time:</b> {detection.get('timestamp', 'N/A')}
            """
            
            folium.Marker(
                location=[detection['latitude'], detection['longitude']],
                popup=folium.Popup(popup_text, max_width=250),
                icon=folium.Icon(color=color, icon=icon, prefix='fa'),
                tooltip=f"{'Manual Upload' if is_manual else 'Drone Patrol'}"
            ).add_to(marker_cluster)
        
        st_folium(m, width=800, height=600)
        
        # Legend
        st.markdown("""
        <div style='display: flex; gap: 20px; align-items: center; margin-top: 10px;'>
            <div style='display: flex; align-items: center; gap: 5px;'>
                <div style='width: 15px; height: 15px; background: red; border-radius: 50%;'></div>
                <span>Manual Upload</span>
            </div>
            <div style='display: flex; align-items: center; gap: 5px;'>
                <div style='width: 15px; height: 15px; background: orange; border-radius: 50%;'></div>
                <span>Drone Patrol</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"Showing {len(filtered)} detections")
    else:
        st.info("No detection history available yet.")
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=6)
        st_folium(m, width=800, height=600)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    if health.get('status') == 'ok':
        st.sidebar.success("API: Connected")
    else:
        st.sidebar.warning("API: Degraded")
except:
    st.sidebar.error("API: Unreachable")
