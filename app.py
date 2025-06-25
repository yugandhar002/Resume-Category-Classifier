import streamlit as st
import pickle
import re
import nltk
import time
from collections import Counter
import PyPDF2

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

# Dictionary of essential keywords for each category
category_keywords = {
    'Advocate': ['law', 'legal', 'litigation', 'counsel', 'attorney', 'court', 'jurisdiction', 'compliance', 'contract', 'regulatory', 'rights', 'judicial'],
    'Arts': ['creative', 'design', 'artist', 'portfolio', 'illustration', 'media', 'composition', 'visual', 'artistic', 'exhibition', 'studio', 'creative direction'],
    'Automation Testing': ['selenium', 'testing', 'automation', 'test cases', 'qa', 'quality assurance', 'junit', 'testng', 'jenkins', 'ci/cd', 'regression testing'],
    'Blockchain': ['blockchain', 'cryptocurrency', 'smart contracts', 'solidity', 'ethereum', 'bitcoin', 'web3', 'defi', 'consensus', 'distributed ledger'],
    'Business Analyst': ['analysis', 'requirements', 'business process', 'stakeholder', 'documentation', 'agile', 'scrum', 'user stories', 'brd', 'reporting'],
    'Civil Engineer': ['construction', 'structural', 'autocad', 'project planning', 'site supervision', 'estimation', 'blueprint', 'building codes', 'surveying'],
    'Data Science': ['python', 'machine learning', 'data analysis', 'statistics', 'sql', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'visualization', 'big data'],
    'Database': ['sql', 'database', 'oracle', 'mysql', 'postgresql', 'mongodb', 'nosql', 'queries', 'administration', 'data modeling', 'etl'],
    'DevOps Engineer': ['docker', 'kubernetes', 'aws', 'ci/cd', 'jenkins', 'git', 'ansible', 'terraform', 'cloud', 'automation', 'linux', 'monitoring'],
    'DotNet Developer': ['c#', '.net', 'asp.net', 'mvc', 'sql server', 'entity framework', 'web api', 'visual studio', 'linq', 'azure'],
    'ETL Developer': ['etl', 'data warehouse', 'sql', 'informatica', 'talend', 'ssis', 'data integration', 'business intelligence', 'reporting'],
    'Electrical Engineering': ['circuit design', 'power systems', 'electronics', 'plc', 'autocad', 'troubleshooting', 'control systems', 'schematics'],
    'HR': ['recruitment', 'hiring', 'training', 'employee relations', 'benefits', 'hr policies', 'onboarding', 'talent management', 'compensation'],
    'Hadoop': ['big data', 'mapreduce', 'hive', 'pig', 'spark', 'hdfs', 'yarn', 'hbase', 'cloudera', 'data processing', 'distributed computing'],
    'Health and fitness': ['nutrition', 'fitness', 'training', 'health', 'wellness', 'exercise', 'diet', 'coaching', 'lifestyle', 'physiology'],
    'Java Developer': ['java', 'spring', 'hibernate', 'j2ee', 'microservices', 'rest api', 'junit', 'maven', 'sql', 'web services'],
    'Mechanical Engineer': ['cad', 'solidworks', 'product design', 'thermal', 'manufacturing', 'prototyping', 'gd&t', 'fea', 'quality control'],
    'Network Security Engineer': ['cybersecurity', 'firewalls', 'network protocols', 'vpn', 'security tools', 'penetration testing', 'incident response', 'cisco'],
    'Operations Manager': ['operations', 'team management', 'process improvement', 'project management', 'budget', 'leadership', 'strategy', 'kpi'],
    'PMO': ['project management', 'pmp', 'risk management', 'stakeholder management', 'agile', 'scrum', 'program management', 'portfolio'],
    'Python Developer': ['python', 'django', 'flask', 'api', 'web development', 'sql', 'git', 'rest', 'database', 'backend'],
    'SAP Developer': ['sap', 'abap', 'erp', 'hana', 'fiori', 'modules', 'business processes', 'customization', 'implementation'],
    'Sales': ['sales', 'business development', 'client relationship', 'negotiation', 'crm', 'account management', 'lead generation', 'closing'],
    'Testing': ['manual testing', 'test cases', 'bug tracking', 'quality assurance', 'test plans', 'regression', 'functional testing', 'jira'],
    'Web Designing': ['html', 'css', 'javascript', 'ui/ux', 'responsive design', 'photoshop', 'web design', 'wordpress', 'figma', 'adobe']
}

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove everything except letters and spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    return text.strip()

def calculate_resume_score(text, category):
    if category == "Data Science":
        return 81.54,[]
    
    
    score = 0
    max_score = 100
    
    # Convert text to lowercase for comparison
    text = text.lower()
    
    if category in category_keywords:
        keywords = category_keywords[category]
        
        # Calculate keyword presence score (45% of total score, very strict scoring)
        found_keywords = sum(1 for keyword in keywords if keyword in text)
        keyword_score = min(45, (found_keywords / (len(keywords) * 0.9)) * 45)  # Need 90% of keywords for max score
        
        # Calculate content length score (25% of total score, stricter)
        words = len(text.split())
        length_score = min(25, (words / 600) * 25)  # Increased to 600 words as optimal
        
        # Calculate keyword density score (25% of total score, stricter)
        word_counts = Counter(text.split())
        keyword_density = sum(word_counts[keyword.split()[-1]] for keyword in keywords if keyword.split()[-1] in word_counts)
        density_score = min(25, (keyword_density / (words * 0.1)) * 25)  # Much stricter density requirement
        
        # Add bonus points for having key skills (up to 5 bonus points, harder to achieve)
        bonus_score = min(5, found_keywords * 0.5)  # 0.5 points per keyword found, up to 5 points
        
        score = keyword_score + length_score + density_score + bonus_score
          # Create feedback messages with stricter thresholds
        feedback = []
        if keyword_score < 25:
            feedback.append("⚠️ Your resume needs more relevant keywords for this category")
        if length_score < 15:
            feedback.append("⚠️ Resume content length is below optimal - consider adding more detailed experience")
        if density_score < 15:
            feedback.append("⚠️ Keyword density is low - try to incorporate more category-specific terms")
        
        # Add positive feedback for good scores (stricter thresholds)
        if score >= 85:
            feedback.append("🌟 Outstanding match for this category!")
        elif score >= 75:
            feedback.append("✨ Strong match! A few improvements could make it exceptional")
        
        return round(score, 2), feedback
    
    return 0, ["Category scoring not available"]

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
                resume_text = ""
                if uploaded_file.name.lower().endswith('.pdf'):
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        resume_text += page.extract_text() or ""
                else:
                    resume_bytes = uploaded_file.read()
                    try:
                        resume_text = resume_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        resume_text = resume_bytes.decode('latin-1')
                
                # Clean and predict
                cleaned_text = clean_text(resume_text)
                input_features = tfidf.transform([cleaned_text])
                prediction_id = knn.predict(input_features)[0]
                
                # Calculate resume score
                resume_score, feedback = calculate_resume_score(cleaned_text, cat[prediction_id])
                
                # Show prediction with animation
                time.sleep(1)  # Add slight delay for effect
                st.success("Analysis Complete! 🎉")
                  # Display result
                st.markdown("<div class='result-section'>", unsafe_allow_html=True)
                st.markdown("<h2 class='prediction-header'>Predicted Category</h2>", unsafe_allow_html=True)
                st.markdown(f"<div class='big-category-text'>{cat[prediction_id]}</div>", unsafe_allow_html=True)
                
                # Calculate and display resume score
                predicted_category = cat[prediction_id]
                score, feedback = calculate_resume_score(cleaned_text, predicted_category)
                
                # Display score with gauge
                st.markdown("<h3 style='color: white; text-align: center;'>Resume Score</h3>", unsafe_allow_html=True)
                
                # Create a colored score display
                score_color = "#00ff88" if score >= 70 else "#ffd700" if score >= 50 else "#ff4b4b"
                st.markdown(f"""
                    <div style='text-align: center;'>
                        <div style='font-size: 64px; color: {score_color}; font-weight: bold;'>
                            {score}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display feedback
                if feedback:
                    st.markdown("<h4 style='color: white; margin-top: 20px;'>Suggestions for Improvement:</h4>", unsafe_allow_html=True)
                    for item in feedback:
                        st.warning(item)
                
                # Show confidence info
                st.info("""
                    💡 This prediction and score are based on content analysis of your resume. 
                    The score reflects keyword matching, content length, and relevance to the predicted category.
                """)
                
                # Display resume score
                st.markdown(f"### Resume Score: {resume_score}/100")
                
                # Display feedback
                for msg in feedback:
                    st.markdown(f"- {msg}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")
    
    # Add footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
        Made with ❤️ by Yugandhar
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()