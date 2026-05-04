import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import time

st.set_page_config(page_title="Elektro-KI", page_icon="🔌", layout="centered")

# ------------------- Modell (besser als vorher) -------------------
@st.cache_resource(show_spinner="Lade starkes KI-Modell...")
def load_model():
    # Starkes Modell – gut für feine visuelle Details
    model_name = "google/vit-large-patch16-224"
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    return processor, model, model_name

processor, model, model_name = load_model()

labels = model.config.id2label

# ------------------- UI -------------------
st.title("🔌 Elektrotechnische Komponenten Erkennung")
st.markdown("**Hochauflösendes Modell** – bessere Erkennung von Kondensatoren, Widerständen etc.")

col_info, col_img = st.columns([2, 1])
with col_info:
    st.info("**Tipp:** Foto möglichst nah und gut beleuchtet hochladen. Hintergrund sollte nicht zu unruhig sein.")

uploaded_file = st.file_uploader("Foto einer Komponente hochladen", 
                                 type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein Bild", use_column_width=True)

    if st.button("🔍 Jetzt klassifizieren", type="primary", use_container_width=True):
        with st.spinner("Analysiere Bild mit starkem ViT-Modell..."):
            start = time.time()
            
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits[0], dim=-1)
            
            top5_prob, top5_idx = torch.topk(probs, 5)
            duration = time.time() - start

        st.success(f"✅ Fertig in {duration:.2f} Sekunden")

        st.subheader("🔬 Top 5 Erkennungen")
        for prob, idx in zip(top5_prob, top5_idx):
            label = labels[int(idx)].replace("_", " ").title()
            confidence = float(prob) * 100
            st.metric(label=label, value=f"{confidence:.1f}%")
            
        st.caption("Hinweis: Das Modell ist allgemein trainiert. Für maximale Genauigkeit können wir es später auf Elektro-Bauteile fine-tunen.")

# Footer
st.divider()
st.caption("Modell: google/vit-large-patch16-224 | Streamlit + Hugging Face")
