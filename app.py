import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import MBartForConditionalGeneration, MBartTokenizerimport, json
from khmernltk import word_tokenize
import torch
import difflib

# Set page config
st.set_page_config(
    page_title="Khmer Homophone Corrector",
    page_icon="✍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 1.2rem;
    }
    .result-text {
        font-size: 1.2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .correction {
        background-color: #ffd700;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
    .correction-details {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .header-image {
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
        display: block;
    }
    .model-info {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Display header image
st.image("header.png", use_column_width=True)

# Model configurations
MODEL_CONFIG = {
    "path": "SocheataSokhachan/khmerhomophonecorrector",
    "description": "Hosted on Hugging Face Hub"
}

def word_segment(text):
    return " ".join(word_tokenize(text)).replace("   ", " ▂ ")

def find_corrections(original, corrected):
    original_words = [w for w in word_tokenize(original) if w.strip()]
    corrected_words = [w for w in word_tokenize(corrected) if w.strip()]
    
    matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
    corrections = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            original_text = ' '.join(original_words[i1:i2])
            corrected_text = ' '.join(corrected_words[j1:j2])
            if original_text.strip() and corrected_text.strip() and original_text != corrected_text:
                corrections.append({
                    'original': original_text,
                    'corrected': corrected_text,
                    'position': i1
                })
    
    return corrections

@st.cache_resource
def load_model(model_path):
    try:
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": device
        }
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_text(text, model_components):
    if model_components is None:
        return "Error: Model not loaded properly"
        
    model = model_components["model"]
    tokenizer = model_components["tokenizer"]
    device = model_components["device"]
    
    segmented_text = word_segment(text)
    input_text = f"{segmented_text} </s> <2km>"
    
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
        add_special_tokens=True
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            num_beams=5,
            early_stopping=True,
            do_sample=False,
            no_repeat_ngram_size=3,
            forced_bos_token_id=32000,
            forced_eos_token_id=32001,
            length_penalty=1.0,
            temperature=1.0
        )
    
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    corrected = corrected.replace("</s>", "").replace("<2km>", "").replace("▂", " ").strip()
    
    return corrected

# Header
st.title("✍️ Khmer Homophone Corrector")

# Simple instruction
st.markdown("Type or paste your Khmer text below to correct homophones.")

# Create two columns for input and output
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Text")
    user_input = st.text_area(
        "Enter Khmer text with homophones:",
        height=200,
        placeholder="Type or paste your Khmer text here...",
        key="input_text"
    )

    correct_button = st.button("🔄 Correct Text", type="primary", use_container_width=True)

with col2:
    st.subheader("Results")
    if correct_button and user_input:
        with st.spinner("Processing..."):
            try:
                # Load model
                model_components = load_model(MODEL_CONFIG["path"])
                
                # Process the text
                corrected = process_text(user_input, model_components)
                
                # Find corrections
                corrections = find_corrections(user_input, corrected)
                
                # Display results
                st.markdown("**Corrected Text:**")
                st.markdown(f'<div class="result-text">{corrected}</div>', unsafe_allow_html=True)
                
                # Show corrections if any were made
                if corrections:
                    st.success(f"Found {len(corrections)} corrections!")
                    st.markdown("**Corrections made:**")
                    for i, correction in enumerate(corrections, 1):
                        st.markdown(f"""
                            <div class="correction-details">
                                {i}. Changed "{correction['original']}" to "{correction['corrected']}"
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No corrections were made.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    elif correct_button:
        st.warning("Please enter text first!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <a href='https://sites.google.com/paragoniu.edu.kh/khmerhomophonecorrector/home' 
           target='_blank' 
           style='text-decoration: none; color: #1f77b4; font-size: 16px;'>
           📚 Learn more about this project
        </a>
    </div>
""", unsafe_allow_html=True) 