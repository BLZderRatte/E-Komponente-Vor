import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import time

st.set_page_config(page_title="Elektro-KI", page_icon="🔌", layout="centered")

# ====================== SPEZIALISIERTES ELEKTRO-MODELL ======================
@st.cache_resource(show_spinner="Lade spezialisiertes Elektro-Modell (nur Bauteile)...")
def load_model():
    model_name = "qipchip31/electronic-components-model"
    
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(
        model_name, 
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    )
    
    return processor, model

processor, model = load_model()

labels = model.config.id2label

# ====================== UI ======================
st.title("🔌 Elektro-Komponenten Erkennung")
st.markdown("**Spezialisiert nur auf elektronische Bauteile**")

uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    if st.button("🔍 Jetzt erkennen", type="primary", use_container_width=True):
        with st.spinner("Analysiere elektrotechnische Komponente..."):
            start = time.time()
            
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
            
            top5_prob, top5_idx = torch.topk(probs, 5)
            duration = time.time() - start

        st.success(f"✅ Fertig in {duration:.2f} Sekunden")

        st.subheader("🔬 Top-Ergebnisse")
        for prob, idx in zip(top5_prob, top5_idx):
            label = labels[int(idx)].replace("_", " ").title()
            confidence = float(prob) * 100
            st.metric(label=label, value=f"{confidence:.1f}%")
