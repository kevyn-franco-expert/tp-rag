import os
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
from .embeddings import OpenAIEmbeddingManager

load_dotenv()
logger = logging.getLogger(__name__)

class TherapistRAGEngine:
    def __init__(self):
        self.embedding_manager = OpenAIEmbeddingManager()
        self.is_loaded = True
    
    def search_similar_cases(
        self, 
        query: str, 
        top_k: int = 5, 
        min_similarity: float = 0.7,
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        return self.embedding_manager.search_similar(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            category_filter=category_filter
        )
    
    def generate_guidance(
        self, 
        patient_context: str, 
        therapist_question: str,
        top_k: int = 3
    ) -> Dict:
        search_query = f"{patient_context} {therapist_question}"
        
        similar_cases = self.search_similar_cases(
            query=search_query,
            top_k=top_k,
            min_similarity=0.5
        )
        
        guidance = self._generate_guidance_text(
            patient_context, 
            therapist_question, 
            similar_cases
        )
        
        warnings = self._extract_warnings(patient_context)
        recommendations = self._generate_recommendations(similar_cases)
        
        return {
            'guidance': guidance,
            'similar_cases': similar_cases,
            'warnings': warnings,
            'recommendations': recommendations,
            'confidence_score': self._calculate_confidence(similar_cases)
        }
    
    def _generate_guidance_text(
        self, 
        patient_context: str, 
        therapist_question: str, 
        similar_cases: List[Dict]
    ) -> str:
        if not similar_cases:
            return (
                "No similar cases found in the database. Consider consulting with "
                "a supervisor or referring to established therapeutic frameworks "
                "for this situation."
            )
        
        approaches = []
        for case in similar_cases:
            response = case['metadata']['Response']
            if 'CBT' in response or 'cognitive' in response.lower():
                approaches.append('Cognitive Behavioral Therapy (CBT)')
            if 'therapy' in response.lower() and 'recommend' in response.lower():
                approaches.append('Professional therapy referral')
            if 'support' in response.lower():
                approaches.append('Support system engagement')
        
        guidance_parts = []
        
        guidance_parts.append(
            f"Based on {len(similar_cases)} similar cases in our database, "
            "here are some therapeutic considerations:"
        )
        
        if approaches:
            unique_approaches = list(set(approaches))
            guidance_parts.append(
                f"\nCommon approaches in similar cases: {', '.join(unique_approaches[:3])}"
            )
        
        guidance_parts.append("\nKey considerations:")
        
        if any('depression' in case['metadata'].get('category', '') for case in similar_cases):
            guidance_parts.append("• Assess for depression symptoms and consider screening tools")
        
        if any('anxiety' in case['metadata'].get('category', '') for case in similar_cases):
            guidance_parts.append("• Evaluate anxiety levels and coping mechanisms")
        
        guidance_parts.append("• Build therapeutic rapport and establish trust")
        guidance_parts.append("• Consider the client's readiness for change")
        guidance_parts.append("• Explore support systems and resources")
        
        guidance_parts.append(
            "\nRemember: This guidance is based on similar cases and should be "
            "adapted to your specific client's needs. Always use your professional "
            "judgment and consider consultation when needed."
        )
        
        return '\n'.join(guidance_parts)
    
    def _extract_warnings(self, patient_context: str) -> List[str]:
        warnings = []
        context_lower = patient_context.lower()
        
        risk_keywords = {
            'suicide': 'Suicide risk indicators detected - immediate assessment needed',
            'self-harm': 'Self-harm indicators present - safety assessment required',
            'abuse': 'Abuse indicators mentioned - consider safety and reporting requirements',
            'crisis': 'Crisis situation indicated - immediate intervention may be needed'
        }
        
        for keyword, warning in risk_keywords.items():
            if keyword in context_lower:
                warnings.append(f"WARNING: {warning}")
        
        if not warnings:
            warnings.append("OK: No immediate risk indicators detected in provided context")
        
        return warnings
    
    def _generate_recommendations(self, similar_cases: List[Dict]) -> List[str]:
        recommendations = [
            "Review similar cases for therapeutic approach patterns",
            "Consider client's individual circumstances and preferences",
            "Evaluate need for additional assessments or referrals",
            "Plan follow-up and progress monitoring strategies"
        ]
        
        if similar_cases:
            categories = [case['metadata'].get('category') for case in similar_cases]
            if 'depression' in categories:
                recommendations.append("Consider depression-specific interventions (CBT, behavioral activation)")
            if 'anxiety' in categories:
                recommendations.append("Explore anxiety management techniques (relaxation, exposure)")
            if 'relationships' in categories:
                recommendations.append("Consider couples/family therapy approaches if appropriate")
        
        return recommendations
    
    def _calculate_confidence(self, similar_cases: List[Dict]) -> float:
        if not similar_cases:
            return 0.0
        
        avg_similarity = sum(case['similarity'] for case in similar_cases) / len(similar_cases)
        case_count_factor = min(len(similar_cases) / 5.0, 1.0)
        
        confidence = avg_similarity * case_count_factor
        return min(confidence, 0.95)
    
    def get_system_stats(self) -> Dict:
        return self.embedding_manager.get_stats()
    
    def health_check(self) -> Dict:
        try:
            stats = self.get_system_stats()
            return {
                'status': 'healthy' if stats.get('status') == 'loaded' else 'not_loaded',
                'embeddings_loaded': stats.get('status') == 'loaded',
                'total_cases': stats.get('total_conversations', 0)
            }
        except Exception as e:
            return {
                'status': 'error',
                'embeddings_loaded': False,
                'total_cases': 0,
                'error': str(e)
            }

def main():
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        logger.error("Please set your OPENAI_API_KEY in the .env file")
        return
    
    rag_engine = TherapistRAGEngine()
    
    logger.info("TESTING RAG ENGINE")
    
    test_query = "Patient feeling depressed and anxious, having trouble sleeping"
    results = rag_engine.search_similar_cases(test_query, top_k=3)
    
    logger.info(f"Query: {test_query}")
    logger.info(f"Found {len(results)} similar cases")
    
    for i, result in enumerate(results, 1):
        logger.info(f"{i}. Similarity: {result['similarity']:.3f}, Category: {result['metadata']['category']}")
    
    logger.info("TESTING GUIDANCE GENERATION")
    
    guidance = rag_engine.generate_guidance(
        patient_context="Patient reports feeling depressed for several weeks, trouble sleeping",
        therapist_question="What therapeutic approach would be most effective?"
    )
    
    logger.info("Guidance generated successfully")
    logger.info(f"Confidence: {guidance['confidence_score']:.2f}")
    logger.info(f"Warnings: {guidance['warnings']}")
    
    

if __name__ == "__main__":
    main()