import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import time

st.set_page_config(page_title="Elektro-KI", page_icon="🔌", layout="centered")

# ------------------- Spezialisiertes Modell -------------------
@st.cache_resource(show_spinner="Lade spezialisiertes Elektro-Modell...")
def load_model():
    model_name = "qipchip31/electronic-components-model"
    
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        st.success("✅ Spezialisiertes Elektro-Modell geladen!")
        return processor, model, model_name
    except Exception as e:
        st.error(f"Fehler beim Laden des Spezialmodells: {e}")
        st.info("Versuche Fallback-Modell...")
        # Fallback
        fallback = "google/vit-base-patch16-224"
        processor = AutoImageProcessor.from_pretrained(fallback)
        model = AutoModelForImageClassification.from_pretrained(fallback)
        return processor, model, fallback

processor, model, model_name = load_model()

labels = model.config.id2label

# ------------------- UI -------------------
st.title("🔌 Elektro-Komponenten Erkennung")
st.markdown("**Spezialisiertes Modell** für Kondensatoren, Transformatoren, Dioden, LEDs, Widerstände usw.")

uploaded_file = st.file_uploader("Foto hochladen (Kondensator, Diode, LED, Transformator...)", 
                                 type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    if st.button("🔍 Jetzt erkennen", type="primary", use_container_width=True):
        with st.spinner("KI analysiert die elektrotechnische Komponente..."):
            start = time.time()
            
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
            
            top5_prob, top5_idx = torch.topk(probs, 5)
            duration = time.time() - start

        st.success(f"✅ Erkennung abgeschlossen in {duration:.2f} Sekunden")

        st.subheader("🔬 Ergebnis")
        for prob, idx in zip(top5_prob, top5_idx):
            label = labels[int(idx)].replace("_", " ").title()
            confidence = float(prob) * 100
            st.metric(label=label, value=f"{confidence:.1f}%")

        if model_name == "qipchip31/electronic-components-model":
            st.caption("Modell ist auf elektronische Bauteile spezialisiert.")
        else:
            st.warning("Fallback-Modell verwendet – Genauigkeit kann niedriger sein.")

# Info
st.divider()
st.info("""
**Unterstützte Komponenten (je nach Training):**
- Kondensator (Capacitor)
- Widerstand (Resistor)
- Diode
- LED
- Transformator / Induktor
- Transistor
- uvm.
""")

st.caption("Modell: qipchip31/electronic-components-model | Streamlit + Hugging Face")
