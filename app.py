import streamlit as st
from PIL import Image
import time
from roboflow import Roboflow

st.set_page_config(page_title="Elektro-KI", page_icon="🔌", layout="centered")

# ====================== DEIN MODELL ======================
@st.cache_resource(show_spinner="Lade Electronic Components Model...")
def load_model():
    rf = Roboflow(api_key="DEIN_ROBOFLOW_API_KEY_HIER")   # ←←← HIER EINTRAGEN!
    
    project = rf.workspace("samu-drioq").project("electronic-components-d6uul")
    model = project.version(1).model          # Version 1 (meistens die aktuelle)
    return model

model = load_model()

st.title("🔌 Electronic Components Erkennung")
st.markdown("**Modell:** samu-drioq / electronic-components-d6uul  \n13 Klassen (Resistor, Capacitor, Transistor, LED, IC, Pot usw.)")

uploaded_file = st.file_uploader("Foto einer Komponente hochladen", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    confidence = st.slider("Mindest-Konfidenz (%)", 10, 95, 50)

    if st.button("🔍 Jetzt erkennen", type="primary", use_container_width=True):
        with st.spinner("Modell analysiert das Bild..."):
            start = time.time()
            
            prediction = model.predict(
                image,
                confidence=confidence,
                overlap=30
            )
            
            duration = time.time() - start

        st.success(f"✅ Fertig in {duration:.2f} Sekunden")

        # Bild mit Bounding Boxes speichern und anzeigen
        prediction.save("result.jpg")
        result_image = Image.open("result.jpg")
        st.image(result_image, caption="Ergebnis mit Bounding Boxes", use_column_width=True)

        # Erkannte Objekte auflisten
        st.subheader("Erkannte Komponenten")
        predictions = prediction.json()["predictions"]

        if predictions:
            for pred in sorted(predictions, key=lambda x: x["confidence"], reverse=True):
                label = pred["class"].replace("_", " ").title()
                conf = pred["confidence"] * 100
                st.metric(label=label, value=f"{conf:.1f}%")
        else:
            st.warning("Keine Komponenten mit ausreichender Sicherheit erkannt.")

# Info
st.divider()
st.info("**Modell-Info:** 13 Klassen elektronischer Bauteile (Resistor, Capacitor, Transistor, LED, IC, Potentiometer etc.)")
st.caption("Projekt: samu-drioq/electronic-components-d6uul")
