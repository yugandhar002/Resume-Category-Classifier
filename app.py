import streamlit as st
import pickle
import re
import nltk
from PIL import Image
import time

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Custom styling
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTitle {
        color: #2e4057;
        font-size: 40px;
        text-align: center;
        margin-bottom: 30px;
    }
    .upload-section {
        background-color: rgba(49, 51, 63, 0.7);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        color: #ffffff;
    }
    .result-section {
        margin-top: 2rem;
        padding: 2rem;
        border-radius: 10px;
        background-color: rgba(49, 51, 63, 0.7);
        color: white;
    }
    .big-category-text {
        color: #00ff88;
        font-size: 48px !important;
        font-weight: bold;
        text-align: center;
        padding: 30px;
        margin: 30px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        background: rgba(0, 0, 0, 0.2);
        border-radius: 15px;
        display: block;
    }
    .prediction-header {
        color: #ffffff;
        font-size: 28px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

#load model
knn = pickle.load(open('knn.pkl','rb'))
#load vectorizer
tfidf = pickle.load(open('tfidf.pkl','rb'))
le = pickle.load(open('encoder.pkl','rb'))

category_mapping = dict(zip(le.transform(le.classes_), le.classes_))

cat = ['Advocate', 'Arts', 'Automation Testing', 'Blockchain', 'Business Analyst', 'Civil Engineer', 'Data Science', 'Database', 'DevOps Engineer', 'DotNet Developer', 'ETL Developer', 'Electrical Engineering', 'HR', 'Hadoop', 'Health and fitness', 'Java Developer', 'Mechanical Engineer', 'Network Security Engineer', 'Operations Manager', 'PMO', 'Python Developer', 'SAP Developer', 'Sales', 'Testing', 'Web Designing']

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove everything except letters and spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    return text.strip()

def main():
    # Header section
    st.markdown("<h1 class='stTitle'>📄 Resume Category Classifier</h1>", unsafe_allow_html=True)
    
    # Info section
    st.info("📌 This tool analyzes resumes and predicts the most suitable job category.")
    
    # Upload section
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.markdown("### Upload Your Resume")
    uploaded_file = st.file_uploader("Choose a file (PDF or TXT)", type=['txt', 'pdf'], 
                                   help="Supported formats: PDF, TXT")
    
    if uploaded_file:
        with st.spinner('Analyzing your resume...'):
            try:
                # Show file details
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": f"{uploaded_file.size/1024:.2f} KB"
                }
                st.write("**File Details:**")
                for key, value in file_details.items():
                    st.write(f"- {key}: {value}")
                
                # Process the file
                resume_bytes = uploaded_file.read()
                try:
                    resume_text = resume_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    resume_text = resume_bytes.decode('latin-1')
                
                # Clean and predict
                cleaned_text = clean_text(resume_text)
                input_features = tfidf.transform([cleaned_text])
                prediction_id = knn.predict(input_features)[0]
                
                # Show prediction with animation
                time.sleep(1)  # Add slight delay for effect
                st.success("Analysis Complete! 🎉")
                
                # Display result
                st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                st.markdown("<h2 class='prediction-header'>Predicted Category</h2>", unsafe_allow_html=True)
                st.markdown(f"<div class='big-category-text'>{cat[prediction_id]}</div>", unsafe_allow_html=True)
                
                # Show confidence info
                st.info("""
                    💡 This prediction is based on the content analysis of your resume. 
                    Consider this as a guide for your job search and application process.
                """)
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")
    
    # Add footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
        Made with ❤️ 
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()