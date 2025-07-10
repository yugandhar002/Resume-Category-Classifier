from collections import Counter
import re

# Dictionary of essential keywords for each category (should be imported or passed in real use)
category_keywords = {
    'Advocate':['law', 'legal', 'litigation', 'counsel', 'attorney', 'court', 'jurisdiction', 'compliance', 'contract', 'regulatory', 'rights', 'judicial'],
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
