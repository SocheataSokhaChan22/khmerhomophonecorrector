import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBartForConditionalGeneration
import json
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
    .homophone-group {
        background-color: #f0f8ff;
        padding: 0.5rem;
        margin: 0.25rem 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #1f77b4;
        white-space: nowrap;
        display: inline-block;
        min-width: 200px;
    }
    .homophone-examples {
        max-height: 400px;
        overflow-y: auto;
        overflow-x: auto;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #fafafa;
        margin: 1rem 0;
    }
    .homophone-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        min-width: max-content;
    }
    .homophone-section {
        margin: 2rem 0;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
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

@st.cache_data
def load_homophone_examples():
    """Load homophone examples from JSON file"""
    try:
        with open('homophone_test.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('homophones', [])
    except Exception as e:
        st.error(f"Error loading homophone examples: {str(e)}")
        return []

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

# Load homophone examples
homophone_examples = load_homophone_examples()

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

# Homophone Examples Section (before footer)
st.markdown("---")
st.markdown("## 📚 List of Homophones")
st.markdown("**Common Khmer homophones that can be corrected:**")

if homophone_examples:
    # Create 3 columns for homophone groups
    col1, col2, col3 = st.columns(3)
    
    # Distribute homophone groups across 3 columns
    groups_per_column = len(homophone_examples) // 3
    remainder = len(homophone_examples) % 3
    
    start_idx = 0
    
    with col1:
        end_idx = start_idx + groups_per_column + (1 if remainder > 0 else 0)
        for i in range(start_idx, end_idx):
            if i < len(homophone_examples) and len(homophone_examples[i]) >= 2:
                group_text = " | ".join(homophone_examples[i])
                st.markdown(f"""
                    <div class="homophone-group">
                        <strong>Group {i+1}:</strong> {group_text}
                    </div>
                """, unsafe_allow_html=True)
        start_idx = end_idx
        remainder = max(0, remainder - 1)
    
    with col2:
        end_idx = start_idx + groups_per_column + (1 if remainder > 0 else 0)
        for i in range(start_idx, end_idx):
            if i < len(homophone_examples) and len(homophone_examples[i]) >= 2:
                group_text = " | ".join(homophone_examples[i])
                st.markdown(f"""
                    <div class="homophone-group">
                        <strong>Group {i+1}:</strong> {group_text}
                    </div>
                """, unsafe_allow_html=True)
        start_idx = end_idx
        remainder = max(0, remainder - 1)
    
    with col3:
        for i in range(start_idx, len(homophone_examples)):
            if len(homophone_examples[i]) >= 2:
                group_text = " | ".join(homophone_examples[i])
                st.markdown(f"""
                    <div class="homophone-group">
                        <strong>Group {i+1}:</strong> {group_text}
                    </div>
                """, unsafe_allow_html=True)
else:
    st.warning("Homophone examples not available.")

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