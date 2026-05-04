import streamlit as st
from PIL import Image
import torch
import time
from ultralytics import YOLO

st.set_page_config(page_title="Elektro YOLO", page_icon="🔌", layout="centered")

# ------------------- YOLO Modell laden -------------------
@st.cache_resource(show_spinner="Lade YOLO-Modell für Elektro-Komponenten...")
def load_model():
    # Gutes öffentliches Modell für elektronische Bauteile
    # Du kannst hier später dein eigenes Roboflow-Modell eintragen
    model_path = "yolov8n.pt"  # Start mit nano (schnell)
    
    # Besser: Ein spezialisiertes Modell (falls verfügbar)
    # model = YOLO("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt")
    model = YOLO(model_path)
    return model

model = load_model()

st.title("🔌 Elektro-Komponenten Erkennung (YOLO)")
st.markdown("**Object Detection** — Zeigt Rahmen + mehrere Komponenten gleichzeitig")

# Bild hochladen
uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    col1, col2 = st.columns(2)
    with col1:
        confidence = st.slider("Mindest-Konfidenz", 0.1, 0.95, 0.5)
    with col2:
        iou = st.slider("IoU-Schwellwert", 0.1, 0.95, 0.45)

    if st.button("🔍 Jetzt erkennen", type="primary", use_container_width=True):
        with st.spinner("YOLO analysiert das Bild..."):
            start = time.time()
            
            # Inference
            results = model(image, conf=confidence, iou=iou)
            
            duration = time.time() - start
            
            # Ergebnisbild mit Bounding Boxes
            result_img = results[0].plot()
            result_pil = Image.fromarray(result_img)
            
            st.image(result_pil, caption="Erkennung mit Bounding Boxes", use_column_width=True)
            st.success(f"✅ Fertig in {duration:.2f} Sekunden")

            # Gefundene Objekte auflisten
            st.subheader("Erkannte Komponenten")
            detections = results[0].boxes
            
            if len(detections) > 0:
                for box in detections:
                    cls = int(box.cls.item())
                    name = model.names[cls]
                    conf = box.conf.item() * 100
                    st.metric(label=name.replace("_", " ").title(), value=f"{conf:.1f}%")
            else:
                st.warning("Keine Komponenten mit ausreichender Sicherheit gefunden.")

st.divider()
st.caption("Powered by Ultralytics YOLO + Roboflow • Gut für Kondensator, Widerstand, Diode, LED, Transistor etc.")
