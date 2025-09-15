import pandas as pd
import re
import os
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TherapyDataProcessor:
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.df = None
        self.stats = {}
    
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.raw_data_path}")
        self.df = pd.read_csv(self.raw_data_path)
        logger.info(f"Loaded {len(self.df)} conversations")
        return self.df
    
    def clean_text(self, text: str) -> str:
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-"\']', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def calculate_quality_score(self, context: str, response: str) -> float:
        score = 100.0
        
        if len(context) < 50:
            score -= 30
        if len(response) < 30:
            score -= 20
        
        if 100 <= len(context) <= 2000:
            score += 10
        if 50 <= len(response) <= 1500:
            score += 10
        
        if len(context) > 3000:
            score -= 20
        if len(response) > 2000:
            score -= 15
        
        return max(0, min(100, score))
    
    def detect_categories(self, context: str) -> str:
        context_lower = context.lower()
        
        categories = {
            'depression': ['depress', 'sad', 'hopeless', 'worthless', 'empty'],
            'anxiety': ['anxious', 'panic', 'worry', 'fear', 'nervous'],
            'relationships': ['marriage', 'relationship', 'partner', 'spouse', 'family'],
            'trauma': ['abuse', 'trauma', 'ptsd', 'assault'],
            'self_esteem': ['self-esteem', 'confidence', 'worth', 'value'],
            'therapy': ['therapy', 'counseling', 'therapist', 'counselor']
        }
        
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in context_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return 'general'
    
    def process_data(self) -> pd.DataFrame:
        """Main processing pipeline."""
        logger.info("Starting data processing...")
        
        # Load data
        self.load_data()
        initial_count = len(self.df)
        
        # Clean text columns
        logger.info("Cleaning text data...")
        self.df['Context'] = self.df['Context'].apply(self.clean_text)
        self.df['Response'] = self.df['Response'].apply(self.clean_text)
        
        # Remove rows with empty content
        self.df = self.df[
            (self.df['Context'].str.len() > 20) & 
            (self.df['Response'].str.len() > 10)
        ]
        
        # Remove exact duplicates
        self.df = self.df.drop_duplicates()
        
        # Calculate quality scores
        logger.info("Calculating quality scores...")
        self.df['quality_score'] = self.df.apply(
            lambda row: self.calculate_quality_score(row['Context'], row['Response']), 
            axis=1
        )
        
        # Filter by quality (keep scores >= 40)
        self.df = self.df[self.df['quality_score'] >= 40]
        
        # Add categories
        logger.info("Adding categories...")
        self.df['category'] = self.df['Context'].apply(self.detect_categories)
        
        # Add metadata
        self.df['context_length'] = self.df['Context'].str.len()
        self.df['response_length'] = self.df['Response'].str.len()
        self.df['id'] = range(1, len(self.df) + 1)
        
        # Reorder columns
        self.df = self.df[[
            'id', 'Context', 'Response', 'category', 
            'quality_score', 'context_length', 'response_length'
        ]]
        
        # Store processing stats
        final_count = len(self.df)
        self.stats = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_count': initial_count - final_count,
            'removal_percentage': (initial_count - final_count) / initial_count * 100,
            'categories': self.df['category'].value_counts().to_dict(),
            'avg_quality_score': self.df['quality_score'].mean(),
            'avg_context_length': self.df['context_length'].mean(),
            'avg_response_length': self.df['response_length'].mean()
        }
        
        logger.info(f"Processing complete: {initial_count} → {final_count} conversations")
        logger.info(f"Categories found: {list(self.stats['categories'].keys())}")
        
        return self.df
    
    def save_processed_data(self):
        """Save processed data to CSV."""
        logger.info(f"Saving processed data to {self.processed_data_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        
        # Save main data
        self.df.to_csv(self.processed_data_path, index=False)
        
        # Save stats
        stats_path = self.processed_data_path.replace('.csv', '_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("Data Processing Statistics\n")
            f.write("=" * 30 + "\n\n")
            for key, value in self.stats.items():
                f.write(f"{key}: {value}\n")
        
        logger.info("Data saved successfully!")

def main():
    base_dir = Path(__file__).parent.parent
    raw_data_path = base_dir / "data" / "raw" / "train-base.csv"
    processed_data_path = base_dir / "data" / "processed" / "cleaned_conversations.csv"
    
    processor = TherapyDataProcessor(str(raw_data_path), str(processed_data_path))
    processor.process_data()
    processor.save_processed_data()
    
    logger.info("DATA PROCESSING COMPLETE")
    logger.info(f"Processed {processor.stats['final_count']} conversations")
    logger.info(f"Average quality score: {processor.stats['avg_quality_score']:.1f}")
    logger.info(f"Categories: {', '.join(processor.stats['categories'].keys())}")

if __name__ == "__main__":
    main()