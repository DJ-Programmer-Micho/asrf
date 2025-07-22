"""
Generate vocabulary for Kurdish Sorani ASR from transcript data.
Creates vocab.json file needed for custom tokenizer.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import List, Dict, Set
import argparse

from normalizer import KurdishSoraniNormalizer


class KurdishVocabularyGenerator:
    def __init__(self):
        self.normalizer = KurdishSoraniNormalizer()
        
        # Special tokens for ASR
        self.special_tokens = {
            "[PAD]": 0,      # Padding token
            "[UNK]": 1,      # Unknown token  
            "[CLS]": 2,      # Classification token (if needed)
            "[SEP]": 3,      # Separator token (if needed)
            "[MASK]": 4,     # Mask token (if needed)
            "|": 5,          # Word boundary token (important for Wav2Vec2)
        }
        
        # Minimum frequency for character inclusion
        self.min_char_frequency = 1
        
    def extract_characters_from_texts(self, texts: List[str]) -> Counter:
        """
        Extract and count characters from list of texts.
        
        Args:
            texts: List of transcript texts
            
        Returns:
            Counter object with character frequencies
        """
        char_counter = Counter()
        
        for text in texts:
            if pd.isna(text) or not isinstance(text, str):
                continue
                
            # Normalize text first
            normalized_text = self.normalizer.normalize_text(text)
            
            # Count characters
            for char in normalized_text:
                if char != ' ':  # Don't count spaces in vocab (handled by word boundary)
                    char_counter[char] += 1
        
        return char_counter
    
    def load_transcripts_from_csv(self, csv_path: str, transcript_column: str = 'transcript') -> List[str]:
        """
        Load transcripts from CSV file.
        
        Args:
            csv_path: Path to CSV file
            transcript_column: Name of transcript column
            
        Returns:
            List of transcript texts
        """
        try:
            df = pd.read_csv(csv_path)
            if transcript_column not in df.columns:
                raise ValueError(f"Column '{transcript_column}' not found in CSV")
            
            transcripts = df[transcript_column].tolist()
            print(f"Loaded {len(transcripts)} transcripts from {csv_path}")
            return transcripts
            
        except Exception as e:
            print(f"Error loading CSV {csv_path}: {e}")
            return []
    
    def create_vocabulary(self, texts: List[str], min_frequency: int = None) -> Dict[str, int]:
        """
        Create vocabulary mapping from texts.
        
        Args:
            texts: List of transcript texts
            min_frequency: Minimum character frequency (overrides class default)
            
        Returns:
            Dictionary mapping characters to indices
        """
        if min_frequency is None:
            min_frequency = self.min_char_frequency
        
        # Extract character frequencies
        char_counter = self.extract_characters_from_texts(texts)
        
        # Filter by minimum frequency
        filtered_chars = {char: count for char, count in char_counter.items() 
                         if count >= min_frequency}
        
        print(f"Found {len(char_counter)} unique characters")
        print(f"After filtering (min_freq={min_frequency}): {len(filtered_chars)} characters")
        
        # Create vocabulary starting with special tokens
        vocab = dict(self.special_tokens)
        
        # Add characters sorted by frequency (most frequent first)
        sorted_chars = sorted(filtered_chars.items(), key=lambda x: x[1], reverse=True)
        
        current_idx = len(self.special_tokens)
        for char, freq in sorted_chars:
            vocab[char] = current_idx
            current_idx += 1
        
        return vocab
    
    def save_vocabulary(self, vocab: Dict[str, int], output_path: str) -> None:
        """
        Save vocabulary to JSON file.
        
        Args:
            vocab: Vocabulary dictionary
            output_path: Path to save JSON file
        """
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        print(f"Saved vocabulary with {len(vocab)} tokens to {output_path}")
    
    def print_vocabulary_stats(self, vocab: Dict[str, int], char_counter: Counter = None) -> None:
        """
        Print vocabulary statistics.
        
        Args:
            vocab: Vocabulary dictionary
            char_counter: Character frequency counter (optional)
        """
        print(f"\nVocabulary Statistics:")
        print(f"Total tokens: {len(vocab)}")
        print(f"Special tokens: {len(self.special_tokens)}")
        print(f"Character tokens: {len(vocab) - len(self.special_tokens)}")
        
        print(f"\nSpecial tokens:")
        for token, idx in self.special_tokens.items():
            print(f"  {idx}: '{token}'")
        
        if char_counter:
            print(f"\nMost frequent characters:")
            # Get characters (excluding special tokens)
            chars_in_vocab = [char for char in vocab.keys() if char not in self.special_tokens]
            for i, char in enumerate(chars_in_vocab[:20]):  # Top 20
                freq = char_counter.get(char, 0)
                print(f"  {vocab[char]}: '{char}' (freq: {freq})")
    
    def validate_vocabulary(self, vocab: Dict[str, int], texts: List[str]) -> Dict:
        """
        Validate vocabulary coverage on texts.
        
        Args:
            vocab: Vocabulary dictionary
            texts: List of texts to validate against
            
        Returns:
            Validation statistics
        """
        vocab_chars = set(vocab.keys()) - set(self.special_tokens.keys())
        
        total_chars = 0
        covered_chars = 0
        uncovered_chars = set()
        
        for text in texts[:100]:  # Sample validation on first 100 texts
            if pd.isna(text) or not isinstance(text, str):
                continue
                
            normalized = self.normalizer.normalize_text(text)
            for char in normalized:
                if char != ' ':  # Skip spaces
                    total_chars += 1
                    if char in vocab_chars:
                        covered_chars += 1
                    else:
                        uncovered_chars.add(char)
        
        coverage = covered_chars / total_chars if total_chars > 0 else 0
        
        return {
            'total_characters': total_chars,
            'covered_characters': covered_chars,
            'coverage_ratio': coverage,
            'uncovered_characters': list(uncovered_chars)
        }


def main():
    parser = argparse.ArgumentParser(description='Generate Kurdish Sorani vocabulary for ASR')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file with transcripts')
    parser.add_argument('--output_path', type=str, default='./dataset/vocab/vocab.json',
                       help='Output path for vocabulary JSON')
    parser.add_argument('--transcript_column', type=str, default='transcript',
                       help='Name of transcript column in CSV')
    parser.add_argument('--min_frequency', type=int, default=1,
                       help='Minimum character frequency for inclusion')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation on generated vocabulary')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = KurdishVocabularyGenerator()
    generator.min_char_frequency = args.min_frequency
    
    # Load transcripts
    print(f"Loading transcripts from {args.csv_path}")
    texts = generator.load_transcripts_from_csv(args.csv_path, args.transcript_column)
    
    if not texts:
        print("No transcripts loaded. Exiting.")
        return
    
    # Generate vocabulary
    print("Generating vocabulary...")
    char_counter = generator.extract_characters_from_texts(texts)
    vocab = generator.create_vocabulary(texts, args.min_frequency)
    
    # Save vocabulary
    generator.save_vocabulary(vocab, args.output_path)
    
    # Print statistics
    generator.print_vocabulary_stats(vocab, char_counter)
    
    # Validate if requested
    if args.validate:
        print("\nValidating vocabulary coverage...")
        validation = generator.validate_vocabulary(vocab, texts)
        print(f"Coverage: {validation['coverage_ratio']:.3f}")
        print(f"Covered: {validation['covered_characters']}/{validation['total_characters']} characters")
        if validation['uncovered_characters']:
            print(f"Uncovered characters: {validation['uncovered_characters'][:10]}...")


if __name__ == "__main__":
    import sys
    
    # Example usage when run directly
    if len(sys.argv) == 1:
        print("Example usage:")
        print("python generate_vocab.py --csv_path ./dataset/preprocessed_data/metadata.csv --output_path ./dataset/vocab/vocab.json")
        
        # Demo with sample data
        generator = KurdishVocabularyGenerator()
        
        sample_texts = [
            "سڵاو، چۆنی؟ باشم.",
            "ئەم کتێبە زۆر باشە.",  
            "٤٢ ساڵەم.",
            "کوردستان جوانە.",
            "بەیانی باش.",
        ]
        
        vocab = generator.create_vocabulary(sample_texts)
        generator.print_vocabulary_stats(vocab)
        
        # Save sample vocab
        output_path = "./dataset/vocab/sample_vocab.json"
        generator.save_vocabulary(vocab, output_path)
    else:
        main()