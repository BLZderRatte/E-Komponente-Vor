import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import time
from roboflow import Roboflow

st.set_page_config(page_title="29class Elektro-KI", page_icon="🔌", layout="centered")

# ====================== 29class_final MODELL ======================
@st.cache_resource(show_spinner="Lade 29class_final Roboflow Modell...")
def load_model():
    rf = Roboflow(api_key="DEIN_API_KEY_HIER_EINFÜGEN")   # ←←← HIER DEINEN KEY EINTRAGEN!
    project = rf.workspace("electronic-components-dataset-for-yolo").project("29class_final")
    model = project.version(1).model
    return model

model = load_model()

st.title("🔌 29class_final Elektro-Komponenten Erkennung")
st.markdown("**29 Klassen** – Speziell trainiert für elektronische Bauteile")

uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    confidence = st.slider("Mindest-Konfidenz", 0.1, 0.95, 0.5, 0.01)

    if st.button("🔍 Jetzt erkennen", type="primary", use_container_width=True):
        with st.spinner("29class_final Modell läuft..."):
            start = time.time()
            
            # Inference
            prediction = model.predict(
                image,
                confidence=confidence * 100,
                overlap=30
            )
            
            duration = time.time() - start

        st.success(f"✅ Fertig in {duration:.2f} Sekunden")

        # Bild mit Bounding Boxes
        prediction.save("prediction.jpg")
        result_img = Image.open("prediction.jpg")
        st.image(result_img, caption="Erkennung mit Bounding Boxes", use_column_width=True)

        # Liste der erkannten Komponenten
        st.subheader("Erkannte Komponenten")
        predictions = prediction.json()["predictions"]

        if predictions:
            for pred in predictions:
                label = pred["class"].replace("_", " ").title()
                conf = pred["confidence"] * 100
                st.metric(label=label, value=f"{conf:.1f}%")
        else:
            st.warning("Keine Komponenten gefunden.")

# Footer
st.divider()
st.caption("Modell: electronic-components-dataset-for-yolo / 29class_final (Version 1)")
