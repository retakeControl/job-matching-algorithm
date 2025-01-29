from typing import List, Dict
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class JobMatcher:
    def __init__(self):
        """Initialize the JobMatcher with necessary tools for text processing"""
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
    def preprocess_text(self, text: str) -> str:
        """Clean and standardize text data"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())
    
    def calculate_experience_weight(self, start_date: str, end_date: str = None) -> float:
        """
        Calculate weight for work experience based on recency
        More recent experience gets higher weight
        
        Args:
            start_date: Start date of work experience (format: 'YYYY-MM')
            end_date: End date of work experience (format: 'YYYY-MM' or 'present')
        
        Returns:
            float: Weight between 0 and 1
        """
        if end_date is None or end_date.lower() == 'present':
            end_date = datetime.now().strftime('%Y-%m')
            
        start = datetime.strptime(start_date, '%Y-%m')
        end = datetime.strptime(end_date, '%Y-%m')
        current = datetime.now()
        
        # Calculate months from current date to end date
        months_ago = abs((current.year - end.year) * 12 + current.month - end.month)
        
        # Weight decays exponentially with time
        # Experience from 5 years ago gets half the weight of current experience
        decay_rate = 0.139  # ln(2)/60 for 5-year half-life
        weight = max(0.1, pow(2, -decay_rate * months_ago))
        
        return weight

    def infer_skills_from_experience(self, work_experiences: List[Dict]) -> Dict[str, float]:
        """
        Infer skills and their proficiency levels from work experience descriptions
        
        Args:
            work_experiences: List of work experience entries with description and dates
            
        Returns:
            Dict mapping skills to proficiency scores
        """
        # Define skill keywords and their related terms
        skill_patterns = {
            'python': ['python', 'django', 'flask', 'pandas', 'numpy'],
            'java': ['java', 'spring', 'hibernate', 'maven'],
            'web_development': ['html', 'css', 'javascript', 'react', 'angular'],
            'database': ['sql', 'mongodb', 'postgresql', 'mysql', 'database'],
            'cloud': ['aws', 'azure', 'cloud', 'docker', 'kubernetes'],
            'project_management': ['agile', 'scrum', 'project manage', 'team lead'],
            # Add more skill patterns as needed
        }
        
        inferred_skills = {}
        
        for experience in work_experiences:
            description = self.preprocess_text(experience['description'])
            experience_weight = self.calculate_experience_weight(
                experience['start_date'],
                experience.get('end_date')
            )
            
            # Look for skill patterns in the description
            for skill, patterns in skill_patterns.items():
                skill_mentions = sum(1 for pattern in patterns if pattern in description)
                if skill_mentions > 0:
                    # Calculate skill score based on mentions and experience weight
                    skill_score = skill_mentions * experience_weight
                    inferred_skills[skill] = inferred_skills.get(skill, 0) + skill_score
        
        # Normalize skill scores to be between 0 and 1
        if inferred_skills:
            max_score = max(inferred_skills.values())
            inferred_skills = {k: v/max_score for k, v in inferred_skills.items()}
            
        return inferred_skills

    def calculate_match_score(self, job_description: Dict, candidate_profile: Dict) -> Dict:
        """
        Calculate match score between a job description and a candidate profile
        with emphasis on work experience and inferred skills
        """
        # Infer skills from work experience
        inferred_skills = self.infer_skills_from_experience(candidate_profile['work_experience'])
        
        # Process job description
        job_text = self.preprocess_text(
            f"{job_description['title']} {job_description['description']}"
        )
        
        # Process work experience with temporal weighting
        weighted_experience_texts = []
        for exp in candidate_profile['work_experience']:
            weight = self.calculate_experience_weight(exp['start_date'], exp.get('end_date'))
            weighted_experience_texts.append(weight * self.preprocess_text(exp['description']))
        
        profile_text = ' '.join(weighted_experience_texts)
        
        # Calculate text similarity
        texts = [job_text, profile_text]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        experience_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Calculate skill match based on inferred skills
        required_skills = set(job_description['required_skills'])
        skill_match_scores = []
        for skill in required_skills:
            if skill in inferred_skills:
                skill_match_scores.append(inferred_skills[skill])
            else:
                skill_match_scores.append(0)
        
        skill_score = sum(skill_match_scores) / len(required_skills) if required_skills else 0
        
        # Calculate location match
        location_match = job_description['location'].lower() == candidate_profile['preferred_location'].lower()
        
        # Calculate overall match score with new weights
        overall_score = (
            experience_similarity * 0.6 +    # Experience similarity (increased weight)
            skill_score * 0.3 +             # Inferred skills (reduced weight)
            location_match * 0.1            # Location match (reduced weight)
        )
        
        return {
            'overall_score': round(overall_score * 100, 2),
            'experience_similarity': round(experience_similarity * 100, 2),
            'inferred_skills': {k: round(v * 100, 2) for k, v in inferred_skills.items()},
            'skill_score': round(skill_score * 100, 2),
            'location_match': location_match
        }
    
    def find_matching_candidates(self, job_description: Dict, candidate_profiles: List[Dict], 
                               min_score: float = 60.0) -> List[Dict]:
        """Find matching candidates for a job description"""
        matches = []
        
        for profile in candidate_profiles:
            match_result = self.calculate_match_score(job_description, profile)
            
            if match_result['overall_score'] >= min_score:
                matches.append({
                    'candidate_id': profile['id'],
                    'candidate_name': profile['name'],
                    'match_details': match_result
                })
        
        matches.sort(key=lambda x: x['match_details']['overall_score'], reverse=True)
        return matches

# Example usage
def main():
    matcher = JobMatcher()
    
    # Example job description
    job_description = {
        'title': 'Senior Software Engineer',
        'description': 'Looking for an experienced software engineer with strong Python and AWS skills.',
        'required_skills': ['python', 'cloud', 'database'],
        'location': 'New York'
    }
    
    # Example candidate profile with work experience
    candidate_profiles = [
        {
            'id': '001',
            'name': 'John Doe',
            'work_experience': [
                {
                    'title': 'Senior Developer',
                    'description': 'Led development of Python applications using Django and AWS services. Managed PostgreSQL databases.',
                    'start_date': '2022-01',
                    'end_date': 'present'
                },
                {
                    'title': 'Software Engineer',
                    'description': 'Developed Java applications and maintained MySQL databases',
                    'start_date': '2019-06',
                    'end_date': '2021-12'
                }
            ],
            'preferred_location': 'New York'
        },
        {
            'id': '002',
            'name': 'Jane Smith',
            'work_experience': [
                {
                    'title': 'Frontend Developer',
                    'description': 'Developed React applications and worked with REST APIs',
                    'start_date': '2021-03',
                    'end_date': 'present'
                }
            ],
            'preferred_location': 'Boston'
        }
    ]
    
    # Find matches
    matches = matcher.find_matching_candidates(job_description, candidate_profiles)
    
    # Print results
    for match in matches:
        print(f"\nCandidate: {match['candidate_name']}")
        print(f"Overall Match Score: {match['match_details']['overall_score']}%")
        print(f"Experience Similarity: {match['match_details']['experience_similarity']}%")
        print("\nInferred Skills:")
        for skill, score in match['match_details']['inferred_skills'].items():
            print(f"- {skill}: {score}%")
        print(f"Location Match: {'Yes' if match['match_details']['location_match'] else 'No'}")

if __name__ == "__main__":
    main()