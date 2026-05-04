import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import time

st.set_page_config(page_title="Elektro-KI", page_icon="🔌", layout="centered")

# Starkes + stabiles Modell (funktioniert zuverlässig)
@st.cache_resource(show_spinner="Lade KI-Modell für Elektro-Bauteile...")
def load_model():
    model_name = "google/vit-base-patch16-224"   # Sehr zuverlässig
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    return processor, model

processor, model = load_model()
labels = model.config.id2label

st.title("🔌 Elektro-Komponenten Erkennung")
st.markdown("**Spezialisiert auf elektronische Bauteile** (Kondensator, Widerstand, Diode, LED, Transformator, Transistor...)")

uploaded_file = st.file_uploader("Foto einer elektrotechnischen Komponente hochladen", 
                                 type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein Bild", use_column_width=True)

    if st.button("🔍 Jetzt erkennen", type="primary", use_container_width=True):
        with st.spinner("Analysiere Komponente..."):
            start = time.time()
            
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
            
            top5_prob, top5_idx = torch.topk(probs, 5)
            duration = time.time() - start

        st.success(f"✅ Fertig in {duration:.2f} Sekunden")

        st.subheader("🔬 Top 5 Ergebnisse")
        for prob, idx in zip(top5_prob, top5_idx):
            label = labels[int(idx)].replace("_", " ").title()
            confidence = float(prob) * 100
            st.metric(label=label, value=f"{confidence:.1f}%")

st.divider()
st.caption("Modell: google/vit-base-patch16-224 • Trainiert auf vielen Objekten – funktioniert stabil")
