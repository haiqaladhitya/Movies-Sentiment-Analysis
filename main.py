import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #222831;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk membersihkan teks
def clean_text(text):
    """Membersihkan teks dari HTML tags dan karakter khusus"""
    # Hapus HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Hapus karakter non-alfabetik kecuali spasi
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Ubah ke lowercase
    text = text.lower()
    # Hapus spasi berlebih
    text = ' '.join(text.split())
    return text

# Fungsi untuk memuat model dan preprocessing objects
@st.cache_resource
def load_model_and_preprocessing():
    """Memuat model dan preprocessing objects yang sudah dilatih"""
    try:
        # Load model
        model = load_model('best_clstm_model.keras')
        
        # Load preprocessing objects
        with open('preprocessing_objects.pkl', 'rb') as f:
            preprocessing_objects = pickle.load(f)
        
        return model, preprocessing_objects
    except FileNotFoundError as e:
        st.error(f"❌ File tidak ditemukan: {e}")
        st.error("Pastikan file 'best_clstm_model.keras' dan 'preporcessing_objects.pkl' ada di direktori yang sama.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error saat memuat model: {e}")
        return None, None

# Fungsi prediksi sentimen
def predict_sentiment(text, model, preprocessing_objects):
    """Memprediksi sentimen dari teks yang diberikan"""
    try:
        # Ambil tokenizer dan parameter dari preprocessing objects
        tokenizer = preprocessing_objects['tokenizer']
        max_length = preprocessing_objects.get('max_length', 500)
        
        # Bersihkan teks
        cleaned_text = clean_text(text)
        
        # Tokenisasi dan padding
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_length)
        
        # Prediksi
        prediction = model.predict(padded_sequence, verbose=0)[0][0]
        
        return prediction
        
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")
        return 0.5

# Sidebar untuk navigasi
st.sidebar.markdown("# Bar Navigation")
st.sidebar.markdown("---")

# Inisialisasi session state untuk halaman yang dipilih
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Model Test"

# Tombol navigasi
if st.sidebar.button("🧪 Model Test", use_container_width=True, 
                     type="primary" if st.session_state.selected_page == "Model Test" else "secondary"):
    st.session_state.selected_page = "Model Test"

if st.sidebar.button("📊 Overview Model", use_container_width=True,
                     type="primary" if st.session_state.selected_page == "Overview Model" else "secondary"):
    st.session_state.selected_page = "Overview Model"

st.sidebar.markdown("---")

# TIPS DIPINDAH KE SIDEBAR (DI BAWAH NAVIGASI)
st.sidebar.markdown("### 💡 Tips:")
st.sidebar.info("""
- Masukkan review film dalam bahasa Inggris
- Semakin panjang review, semakin akurat prediksi
- Model dapat mendeteksi kata-kata emosional
- Score mendekati 1.0 = Positif
- Score mendekati 0.0 = Negatif
""")

st.sidebar.markdown("---")

# Informasi tambahan di sidebar
st.sidebar.markdown("### 📋 Info")
st.sidebar.info("""
This project was created by:
Steven, Aqil and Haiqal
""")

selected_page = st.session_state.selected_page

# Header utama
st.markdown('<h1 class="main-header">🎬 IMDB Movie Review Sentiment Analysis</h1>', unsafe_allow_html=True)
st.markdown('')

# Load model dan preprocessing objects
model, preprocessing_objects = load_model_and_preprocessing()

# Halaman Model Test
if selected_page == "Model Test":
    # CONTOH REVIEW DIPINDAH KE BAWAH HEADER (SEBELUM INPUT)
    st.markdown("### 📋 Contoh Review:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Review Positif", use_container_width=True):
            st.session_state.example_text = "This movie was absolutely fantastic! The cinematography was breathtaking, the acting was superb, and the storyline kept me engaged throughout. I would definitely recommend this to anyone looking for a great film experience."
    
    with col2:
        if st.button("Review Negatif", use_container_width=True):
            st.session_state.example_text = "This movie was a complete disappointment. The plot was confusing, the acting was terrible, and I felt like I wasted my time watching it. I wouldn't recommend this to anyone."
    
    # Tampilkan contoh teks jika ada
    if 'example_text' in st.session_state:
        st.text_area("Contoh:", value=st.session_state.example_text, height=100, key="example")
        
        col_use, col_clear = st.columns([1, 4])
        with col_use:
            if st.button("Gunakan Contoh Ini"):
                # Copy contoh ke input utama
                st.session_state.user_input = st.session_state.example_text
                st.rerun()
    
    st.markdown("---")
    
    st.markdown("## 🧪 Test Model Sentiment Analysis")
    st.markdown("Masukkan review film untuk menganalisis sentimennya (positif atau negatif)")
    
    if model is not None and preprocessing_objects is not None:
        # Area input teks (full width)
        user_input = st.text_area(
            "📝 Masukkan Review Film:",
            placeholder="Contoh: This movie was absolutely amazing! The acting was superb and the plot was engaging...",
            height=150,
            value=st.session_state.get('user_input', '')
        )
        
        # Tombol prediksi
        if st.button("🔍 Analisis Sentimen", type="primary"):
            if user_input.strip():
                with st.spinner("Menganalisis sentimen..."):
                    # Prediksi sentimen
                    confidence = predict_sentiment(user_input, model, preprocessing_objects)
                    
                    # Tentukan sentimen
                    if confidence > 0.5:
                        sentiment = "Positif"
                        emoji = "😊"
                        color_class = "sentiment-positive"
                        confidence_text = f"{confidence*100:.1f}%"
                    else:
                        sentiment = "Negatif"
                        emoji = "😞"
                        color_class = "sentiment-negative"
                        confidence_text = f"{(1-confidence)*100:.1f}%"
                    
                    # Tampilkan hasil
                    st.markdown(f"""
                    <div class="{color_class}">
                        <h3>{emoji} Hasil Analisis Sentimen</h3>
                        <p><strong>Sentimen:</strong> {sentiment}</p>
                        <p><strong>Tingkat Keyakinan:</strong> {confidence_text}</p>
                        <p><strong>Raw Score:</strong> {confidence:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar untuk confidence
                    progress_value = confidence if confidence > 0.5 else (1-confidence)
                    st.progress(progress_value)
                    
                    # Interpretasi hasil
                    if confidence > 0.8:
                        st.success("✅ Model sangat yakin dengan prediksinya!")
                    elif confidence < 0.2:
                        st.success("✅ Model sangat yakin dengan prediksinya!")
                    elif 0.4 < confidence < 0.6:
                        st.warning("⚠️ Model ragu-ragu. Sentimen mungkin netral atau ambigu.")
                    else:
                        st.info("ℹ️ Model cukup yakin dengan prediksinya.")
                    
            else:
                st.warning("⚠️ Silakan masukkan teks review terlebih dahulu!")
    
    else:
        st.error("❌ Model belum berhasil dimuat. Periksa file model dan preprocessing objects.")

# Halaman Overview Model
elif selected_page == "Overview Model":
    st.markdown("## 📊 Overview Model")
    
    # Informasi model
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Informasi Model</h3>
            <ul>
                <li><strong>Dataset:</strong> IMDB Movie Reviews (50,000 reviews)</li>
                <li><strong>Arsitektur:</strong> CNN-LSTM Hybrid</li>
                <li><strong>Framework:</strong> TensorFlow/Keras</li>
                <li><strong>Input:</strong> Text sequences</li>
                <li><strong>Output:</strong> Binary classification (0-1)</li>
                <li><strong>Model File:</strong> best_clstm_model.keras</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Performa Model</h3>
            <ul>
                <li><strong>Akurasi:</strong> ~87-90%</li>
                <li><strong>Arsitektur:</strong> CNN + LSTM</li>
                <li><strong>Preprocessing:</strong> Tokenization + Padding</li>
                <li><strong>Training:</strong> Early Stopping</li>
                <li><strong>Optimasi:</strong> Best model saved</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Status model
    if model is not None and preprocessing_objects is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("✅ Status Model", "Loaded", "Ready")
        
        with col2:
            vocab_size = len(preprocessing_objects['tokenizer'].word_index) + 1
            st.metric("📚 Vocabulary Size", f"{vocab_size:,}")
        
        with col3:
            max_len = preprocessing_objects.get('max_length', 500)
            st.metric("📏 Max Sequence Length", max_len)
        
        with col4:
            st.metric("🎯 Model Type", "CNN-LSTM", "Binary")
    
    else:
        st.error("❌ Model belum dimuat. Informasi detail tidak tersedia.")
    
    # Visualisasi simulasi
    st.markdown("### 📈 Visualisasi Performa Model")
    
    tab1, tab2, tab3 = st.tabs(["Model Architecture", "Training Simulation", "Data Distribution"])
    
    with tab1:
        st.markdown("""
        ### 🤖 CNN-LSTM
        
        **CNN-LSTM** adalah arsitektur hybrid yang menggabungkan kekuatan **Convolutional Neural Networks (CNN)** dan **Long Short-Term Memory (LSTM)**. Model ini dirancang khusus untuk pemrosesan data sekuensial seperti teks, di mana:
        
        - **CNN** berfungsi untuk **ekstraksi fitur lokal** dan mendeteksi pola-pola pendek dalam teks seperti n-gram dan frasa penting
        - **LSTM** berfungsi untuk **memahami konteks jangka panjang** dan dependensi sekuensial dalam kalimat
        
        Kombinasi ini memungkinkan model untuk menangkap baik pola lokal maupun konteks global dalam analisis sentimen, menghasilkan performa yang superior dibandingkan menggunakan CNN atau LSTM secara terpisah.
        """)

        st.markdown("### 🏗️ Arsitektur CNN-LSTM Model")
        
        st.markdown("""
        **Layer Structure:**
        1. **Embedding Layer** - Convert tokens to dense vectors
        2. **Conv1D Layer** - Local feature extraction
        3. **MaxPooling1D** - Dimensionality reduction
        4. **LSTM Layer** - Sequential pattern learning
        5. **Dropout Layer** - Regularization
        6. **Dense Layer** - Final classification
        
        **Model Flow:**
        ```
        Text Input → Tokenization → Embedding → Conv1D → MaxPool → LSTM → Dense → Prediction
        ```
        """)
        
        # Diagram sederhana
        architecture_data = {
            'Layer': ['Input', 'Embedding', 'Conv1D', 'MaxPool', 'LSTM', 'Dense', 'Output'],
            'Output Shape': ['(batch, seq_len)', '(batch, seq_len, embed)', '(batch, seq_len, filters)', 
                           '(batch, new_len, filters)', '(batch, lstm_units)', '(batch, 1)', '(batch, 1)'],
            'Parameters': ['0', '~1M', '~100K', '0', '~200K', '~65', '1']
        }
        
        df_arch = pd.DataFrame(architecture_data)
        st.dataframe(df_arch, use_container_width=True)
    
    with tab2:
        # Simulasi training history
        epochs = list(range(1, 21))
        train_acc = [0.6 + 0.014*i + np.random.normal(0, 0.01) for i in epochs]
        val_acc = [0.58 + 0.013*i + np.random.normal(0, 0.015) for i in epochs]
        train_loss = [0.6 - 0.025*i + np.random.normal(0, 0.02) for i in epochs]
        val_loss = [0.62 - 0.023*i + np.random.normal(0, 0.025) for i in epochs]
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Loss'))
        
        fig.add_trace(go.Scatter(x=epochs, y=train_acc, name='Train Accuracy', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=val_acc, name='Validation Accuracy', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Train Loss', line=dict(color='blue')), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validation Loss', line=dict(color='red')), row=1, col=2)
        
        fig.update_layout(title="Training History Simulation")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Distribusi data
        labels = ['Positive Reviews', 'Negative Reviews']
        values = [25000, 25000]
        
        fig = px.pie(values=values, names=labels, title="IMDB Dataset Distribution")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Informasi file
    st.markdown("### 📁 File Information")
    
    file_info = {
        'File': ['best_clstm_model.keras', 'preprocessing_objects.pkl'],
        'Type': ['Keras Model', 'Pickle Object'],
        'Contains': ['Trained CNN-LSTM Model', 'Tokenizer + Parameters'],
        'Status': ['✅ Required' if model else '❌ Missing', 
                  '✅ Required' if preprocessing_objects else '❌ Missing']
    }   
    
    df_files = pd.DataFrame(file_info)
    st.dataframe(df_files, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    🎬 IMDB Sentiment Analysis | CNN-LSTM Model | Built with TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)
