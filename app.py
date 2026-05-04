import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import time

st.set_page_config(page_title="Elektro-KI", page_icon="🔌", layout="centered")

# ------------------- Modell laden -------------------
@st.cache_resource(show_spinner="Lade KI-Modell...")
def load_model():
    # Zuverlässigeres Modell als Alternative
    model_name = "google/vit-base-patch16-224"  # Fallback (gut trainiert)
    
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        st.success("✅ Modell erfolgreich geladen")
        return processor, model, model_name
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")
        st.stop()

processor, model, model_name = load_model()

# Labels vom Modell holen
labels = model.config.id2label

# ------------------- UI -------------------
st.title("🔌 Elektrotechnische Komponenten Erkennung")
st.markdown("**Lade ein Foto einer elektronischen Komponente hoch**")

uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    if st.button("🔍 Jetzt erkennen", type="primary", use_container_width=True):
        with st.spinner("KI analysiert das Bild..."):
            start = time.time()
            
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
            
            top5_prob, top5_idx = torch.topk(probs, 5)
            
            duration = time.time() - start

        st.success(f"Fertig in {duration:.2f} Sekunden")

        st.subheader("Ergebnisse")
        for prob, idx in zip(top5_prob, top5_idx):
            label = labels[int(idx)].replace("_", " ").title()
            confidence = float(prob) * 100
            st.metric(label=label, value=f"{confidence:.1f}%")
