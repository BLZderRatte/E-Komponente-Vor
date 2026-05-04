import streamlit as st
from PIL import Image
import time
from inference_sdk import InferenceHTTPClient

st.set_page_config(page_title="29class Elektro-KI", page_icon="🔌", layout="centered")

# ====================== 29class_final MODELL ======================
@st.cache_resource(show_spinner="Verbinde mit Roboflow 29class_final Modell...")
def load_client():
    # Ersetze mit deinem eigenen API-Key (kostenlos auf Roboflow)
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="DEIN_API_KEY_HIER_EINFÜGEN"   # ←←← HIER ÄNDERN!
    )
    return client

client = load_client()

# Modell-ID für genau dieses Projekt
MODEL_ID = "electronic-components-dataset-for-yolo/29class_final/1"   # Version 1

st.title("🔌 29class_final Elektro-Komponenten Erkennung")
st.markdown("**29 Klassen** – Speziell für elektronische Bauteile (inkl. viele Widerstandswerte, BJT, MOSFET, OP_AMP etc.)")

uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    confidence = st.slider("Mindest-Konfidenz", 0.1, 0.95, 0.5)

    if st.button("🔍 Jetzt erkennen", type="primary", use_container_width=True):
        with st.spinner("29class_final Modell analysiert..."):
            start = time.time()
            
            result = client.infer(
                image,
                model_id=MODEL_ID,
                confidence=confidence
            )
            
            duration = time.time() - start

        st.success(f"✅ Fertig in {duration:.2f} Sekunden")

        # Ergebnisbild mit Bounding Boxes
        if result.get("image"):
            st.image(result["image"], caption="Erkennung mit Bounding Boxes", use_column_width=True)
        else:
            # Fallback: Originalbild mit Overlay (falls kein Bild zurückkommt)
            st.image(image, caption="Ergebnis (Bounding Boxes nicht direkt verfügbar)")

        # Gefundene Objekte auflisten
        st.subheader("Erkannte Komponenten")
        predictions = result.get("predictions", [])

        if predictions:
            for pred in predictions:
                label = pred["class"].replace("_", " ").title()
                conf = pred["confidence"] * 100
                st.metric(label=label, value=f"{conf:.1f}%")
        else:
            st.warning("Keine Komponenten mit ausreichender Sicherheit gefunden.")

# Hinweis
st.divider()
st.info("Modell: **29class_final** von Roboflow\n\n29 Klassen elektronischer Bauteile")
st.caption("API-Key von https://roboflow.com erforderlich")
