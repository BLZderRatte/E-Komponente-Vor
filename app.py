import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import time

# ------------------- Konfiguration -------------------
st.set_page_config(
    page_title="Elektro-Komponenten Erkennung",
    page_icon="🔌",
    layout="centered"
)

# Modell laden (wird beim ersten Start heruntergeladen und dann gecacht)
@st.cache_resource
def load_model():
    model_name = "qipchip31/electronic-components-model"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

processor, model = load_model()

# Klassen (basierend auf dem Modell)
# Das Modell erkennt typischerweise: resistor, capacitor, inductor, transistor, diode usw.
labels = model.config.id2label

# ------------------- UI -------------------
st.title("🔌 Elektrotechnische Komponenten Erkennung")
st.markdown("""
**Lade ein Foto einer Komponente hoch** (Widerstand, Kondensator, Spule, Transistor, Diode etc.)
""")

# Tabs
tab1, tab2 = st.tabs(["📸 Erkennung", "ℹ️ Info"])

with tab1:
    uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
            
            if st.button("🔍 Jetzt klassifizieren", type="primary"):
                with st.spinner("KI analysiert das Bild..."):
                    start_time = time.time()
                    
                    # Vorverarbeitung
                    inputs = processor(images=image, return_tensors="pt")
                    
                    # Inference
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probabilities = torch.nn.functional.softmax(logits[0], dim=-1)
                    
                    # Top 5 Vorhersagen
                    top5_prob, top5_idx = torch.topk(probabilities, 5)
                    
                    end_time = time.time()
                    
                    st.success(f"Fertig in {end_time - start_time:.2f} Sekunden")
                    
                    # Ergebnisse anzeigen
                    st.subheader("Ergebnis")
                    for i, (prob, idx) in enumerate(zip(top5_prob, top5_idx)):
                        label = labels[int(idx)]
                        confidence = prob.item() * 100
                        st.metric(
                            label=label.replace("_", " ").title(),
                            value=f"{confidence:.1f}%"
                        )
    
    with col2:
        st.info("**Tipps für gute Ergebnisse:**\n"
                "- Gute Beleuchtung\n"
                "- Komponente gut sichtbar und zentriert\n"
                "- Möglichst wenig Hintergrundrauschen")

with tab2:
    st.markdown("""
    ### Verwendetes Modell
    - **Modell:** [qipchip31/electronic-components-model](https://huggingface.co/qipchip31/electronic-components-model)
    - **Architektur:** Vision Transformer (ViT) fine-tuned auf elektronische Bauteile
    - **Klassen:** Widerstände, Kondensatoren, Spulen, Transistoren, Dioden uvm.
    
    ### Deployment
    Die App läuft perfekt auf **Streamlit Community Cloud** (kostenlos).
    """)

# Footer
st.caption("🚀 Erstellt mit ❤️ für die Elektrotechnik-Community | Hugging Face + Streamlit")
