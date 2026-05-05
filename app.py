import streamlit as st
from PIL import Image
import time
import os
from roboflow import Roboflow

st.set_page_config(page_title="Elektro-KI", page_icon="🔌", layout="centered")

# ====================== BAUTEIL-ERKLÄRUNGEN ======================
component_info = {
    "resistor": "Widerstand – Begrenzt den Stromfluss in einer Schaltung. Wird in fast allen elektronischen Geräten verwendet.",
    "capacitor": "Kondensator – Speichert elektrische Energie kurzzeitig. Wichtig für Spannungsstabilisierung und Filter.",
    "transistor": "Transistor – Schaltet oder verstärkt Signale. Grundbaustein aller modernen Computer und Verstärker.",
    "led": "LED (Light Emitting Diode) – Leuchtdiode, die Licht erzeugt, wenn Strom fließt. Sehr energieeffizient.",
    "diode": "Diode – Lässt Strom nur in eine Richtung fließen. Wird als Gleichrichter oder zum Schutz verwendet.",
    "ic": "IC (Integrated Circuit) – Integrierte Schaltung (Chip). Enthält viele Transistoren auf kleinem Raum (z.B. Mikrocontroller).",
    "inductor": "Spule / Induktor – Speichert Energie in einem Magnetfeld. Wichtig in Schaltnetzteilen und Filtern.",
    "potentiometer": "Potentiometer – Veränderbarer Widerstand. Wird als Drehregler für Lautstärke, Helligkeit etc. verwendet.",
    "switch": "Schalter – Unterbricht oder schließt einen Stromkreis manuell.",
    "battery": "Batterie – Chemische Energiequelle, die elektrische Spannung liefert.",
    "relay": "Relais – Elektromagnetischer Schalter. Erlaubt es, hohe Ströme mit einem kleinen Signal zu schalten.",
    "connector": "Steckverbinder – Dient zur mechanischen und elektrischen Verbindung von Kabeln oder Platinen.",
    "fuse": "Sicherung – Schützt die Schaltung, indem sie bei Überstrom durchbrennt."
}

# ====================== MODELL LADEN ======================
@st.cache_resource(show_spinner="Lade Electronic Components Model...")
def load_model():
    rf = Roboflow(api_key="zza9zsVKAPFMWMKaebBo")
    project = rf.workspace("samu-drioq").project("electronic-components-d6uul")
    model = project.version(1).model
    return model

model = load_model()

st.title("🔌 Electronic Components Erkennung")
st.markdown("**Modell:** samu-drioq / electronic-components-d6uul • Mit Erklärungen")

uploaded_file = st.file_uploader("Foto einer Komponente hochladen", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    confidence = st.slider("Mindest-Konfidenz (%)", 10, 95, 50)

    if st.button("🔍 Jetzt erkennen", type="primary", use_container_width=True):
        with st.spinner("Modell analysiert das Bild..."):
            start = time.time()
            
            temp_path = "temp_upload.jpg"
            image.save(temp_path)
            
            prediction = model.predict(temp_path, confidence=confidence, overlap=30)
            duration = time.time() - start

        st.success(f"✅ Fertig in {duration:.2f} Sekunden")

        # Ergebnis anzeigen
        prediction.save("result.jpg")
        result_image = Image.open("result.jpg")
        st.image(result_image, caption="Erkennung mit Bounding Boxes", use_column_width=True)

        # Erkannte Komponenten + Erklärung
        st.subheader("Erkannte Komponenten")
        predictions = prediction.json()["predictions"]

        if predictions:
            for pred in sorted(predictions, key=lambda x: x["confidence"], reverse=True):
                class_name = pred["class"].lower()
                label = pred["class"].replace("_", " ").title()
                conf = pred["confidence"] * 100
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric(label=label, value=f"{conf:.1f}%")
                with col2:
                    description = component_info.get(class_name, "Keine Beschreibung verfügbar.")
                    st.write(f"**Verwendung:** {description}")
        else:
            st.warning("Keine Komponenten mit ausreichender Sicherheit erkannt.")

        if os.path.exists(temp_path):
            os.remove(temp_path)

st.divider()
st.caption("Modell: samu-drioq/electronic-components-d6uul | Mit Bauteil-Erklärungen")
