"""
Kurdish Sorani text normalization for ASR training.
Handles character standardization, spelling variations, and cleaning.
"""

import re
import unicodedata
from typing import Dict, List


class KurdishSoraniNormalizer:
    def __init__(self):
        # Kurdish Sorani specific character mappings
        self.char_mappings = {
            # Standardize Kurdish letters
            'ك': 'ک',  # Arabic kaf to Kurdish kaf
            'ي': 'ی',  # Arabic yeh to Kurdish yeh
            'ى': 'ی',  # Alef maksura to Kurdish yeh
            'ة': 'ە',  # Teh marbuta to Kurdish schwa
            'أ': 'ا',  # Alef with hamza above
            'إ': 'ا',  # Alef with hamza below
            'آ': 'ا',  # Alef with madda above
            
            # English digits to Kurdish/Arabic digits (optional)
            '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
            '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩',
            
            # Common variations
            'ئێ': 'ێ',  # Simplify some combinations
            'وو': 'و',  # Reduce doubled waw
        }
        
        # Valid Kurdish Sorani characters
        self.valid_chars = {
            # Kurdish specific letters
            'ئ', 'ا', 'ب', 'پ', 'ت', 'ج', 'چ', 'ح', 'خ', 'د', 'ر', 'ڕ', 'ز', 'ژ',
            'س', 'ش', 'ع', 'غ', 'ف', 'ڤ', 'ق', 'ک', 'گ', 'ل', 'ڵ', 'م', 'ن', 'ه',
            'و', 'ۆ', 'ی', 'ێ', 'ە',
            
            # Diacritics and marks
            'َ', 'ِ', 'ُ', 'ً', 'ٍ', 'ٌ', 'ْ', 'ّ',
            
            # Numbers
            '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩',
            
            # Punctuation and spaces
            ' ', '.', '،', '؟', '!', ':', ';', '"', "'", '(', ')', '[', ']',
            '{', '}', '-', '–', '—', '/', '\\', '|', '@', '#', '%', '&', '*',
            '+', '=', '<', '>', '~', '`',
        }
        
        # Common Kurdish words that might have spelling variations
        self.word_mappings = {
            'دەستور': 'دەستوور',  # Constitution
            'پێشکەش': 'پێشکەشی',  # Presentation variants
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Kurdish Sorani text for ASR training.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Convert to lowercase (if needed for your model)
        # Note: Kurdish has case distinctions, so be careful
        # text = text.lower()
        
        # 3. Apply character mappings
        for old_char, new_char in self.char_mappings.items():
            text = text.replace(old_char, new_char)
        
        # 4. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 5. Apply word-level corrections
        words = text.split()
        normalized_words = []
        for word in words:
            # Remove punctuation for word lookup
            clean_word = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', '', word)
            if clean_word in self.word_mappings:
                # Replace the clean part but keep punctuation
                normalized_word = word.replace(clean_word, self.word_mappings[clean_word])
                normalized_words.append(normalized_word)
            else:
                normalized_words.append(word)
        
        text = ' '.join(normalized_words)
        
        # 6. Handle specific Kurdish patterns
        text = self._handle_kurdish_patterns(text)
        
        # 7. Final cleanup
        text = self._final_cleanup(text)
        
        return text
    
    def _handle_kurdish_patterns(self, text: str) -> str:
        """Handle Kurdish-specific linguistic patterns."""
        
        # Remove or standardize certain diacritics for ASR
        # (depends on your ASR approach - with or without diacritics)
        # text = re.sub(r'[ًٌٍَُِّْ]', '', text)  # Remove all diacritics
        
        # Standardize some common prefixes/suffixes
        # Example: handle definite article variations
        text = re.sub(r'\bئه\b', 'ئە', text)  # Definite article
        
        # Handle doubled consonants
        text = re.sub(r'([بپتجچحخدرڕزژسشعغفڤقکگلڵمنهوۆیێ])\1+', r'\1\1', text)
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final text cleanup."""
        
        # Remove characters not in valid set (optional - be careful!)
        # filtered_chars = [c for c in text if c in self.valid_chars]
        # text = ''.join(filtered_chars)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.،؟!:;])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.،؟!:;])([^\s])', r'\1 \2', text)  # Add space after punctuation
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_character_set(self, texts: List[str]) -> set:
        """
        Extract all unique characters from a list of texts.
        Useful for building vocabulary.
        
        Args:
            texts: List of text strings
            
        Returns:
            Set of unique characters
        """
        all_chars = set()
        for text in texts:
            normalized = self.normalize_text(text)
            all_chars.update(set(normalized))
        
        return all_chars
    
    def validate_text(self, text: str) -> Dict:
        """
        Validate text and return statistics.
        
        Args:
            text: Text to validate
            
        Returns:
            Dictionary with validation results
        """
        normalized = self.normalize_text(text)
        
        char_count = len(normalized)
        word_count = len(normalized.split())
        unique_chars = set(normalized)
        
        # Check for non-Kurdish characters
        invalid_chars = unique_chars - self.valid_chars
        
        return {
            'original_length': len(text),
            'normalized_length': char_count,
            'word_count': word_count,
            'unique_chars': len(unique_chars),
            'invalid_chars': list(invalid_chars),
            'is_valid': len(invalid_chars) == 0
        }


def normalize_transcript_file(input_file: str, output_file: str) -> None:
    """
    Normalize all transcripts in a file.
    
    Args:
        input_file: Path to input file (CSV with transcript column)
        output_file: Path to output normalized file
    """
    import pandas as pd
    
    normalizer = KurdishSoraniNormalizer()
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Normalize transcripts
    print("Normalizing transcripts...")
    df['transcript'] = df['transcript'].apply(normalizer.normalize_text)
    
    # Filter out empty transcripts
    df = df[df['transcript'].str.len() > 0]
    
    # Save normalized data
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} normalized transcripts to {output_file}")


if __name__ == "__main__":
    # Example usage
    normalizer = KurdishSoraniNormalizer()
    
    # Test normalization
    test_texts = [
        "سڵاو، چۆنی؟ باشم.",  # Hello, how are you? I'm fine.
        "ئەم کتێبە زۆر باشە.",  # This book is very good.
        "٤٢ ساڵەم.",  # I am 42 years old.
        "دەستور   و    یاسا.",  # Constitution and law (with extra spaces)
    ]
    
    for text in test_texts:
        normalized = normalizer.normalize_text(text)
        print(f"Original:  {text}")
        print(f"Normalized: {normalized}")
        print("---")