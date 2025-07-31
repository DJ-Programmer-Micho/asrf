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

            # English digits to Arabic digits
            '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
            '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩',

            # Common variations
            'ئێ': 'ێ',
            'وو': 'و',
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

            # Punctuation
            ' ', '.', '،', '؟', '!', ':', ';', '"', "'", '(', ')', '[', ']',
            '{', '}', '-', '–', '—', '/', '\\', '|', '@', '#', '%', '&', '*',
            '+', '=', '<', '>', '~', '`',
        }

        # Word-level mappings (optional spelling fixes)
        self.word_mappings = {
            'دەستور': 'دەستوور',
            'پێشکەش': 'پێشکەشی',
        }

    def normalize_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        # Step 0: Unicode normalization
        text = unicodedata.normalize('NFKC', text)

        # Step 1: Remove invisible or problematic characters
        invisible_chars = ['\u200c', '\u200d', '\u200e', '\u200f', '\u202a', '\u202b', '\u202c']
        for ch in invisible_chars:
            text = text.replace(ch, '')

        # Step 2: Apply character mappings
        for old_char, new_char in self.char_mappings.items():
            text = text.replace(old_char, new_char)

        # Step 3: Apply word-level corrections
        words = text.split()
        normalized_words = []
        for word in words:
            clean_word = re.sub(r'[^\u0600-\u06FF]', '', word)
            if clean_word in self.word_mappings:
                normalized_word = word.replace(clean_word, self.word_mappings[clean_word])
                normalized_words.append(normalized_word)
            else:
                normalized_words.append(word)
        text = ' '.join(normalized_words)

        # Step 4: Handle Kurdish-specific linguistic patterns
        text = self._handle_kurdish_patterns(text)

        # Step 5: Final cleanup
        text = self._final_cleanup(text)

        return text

    def _handle_kurdish_patterns(self, text: str) -> str:
        # Example: replace standalone definite article
        text = re.sub(r'\bئه\b', 'ئە', text)

        # Remove repeated letters (limit to 2 for emphasis)
        text = re.sub(r'([بپتجچحخدرڕزژسشعغفڤقکگلڵمنهوۆیێ])\1{2,}', r'\1\1', text)

        return text

    def _final_cleanup(self, text: str) -> str:
        # Normalize spaces around punctuation
        text = re.sub(r'\s+([.،؟!:;])', r'\1', text)
        text = re.sub(r'([.،؟!:;])([^\s])', r'\1 \2', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def get_character_set(self, texts: List[str]) -> set:
        all_chars = set()
        for text in texts:
            normalized = self.normalize_text(text)
            all_chars.update(set(normalized))
        return all_chars

    def validate_text(self, text: str) -> Dict:
        normalized = self.normalize_text(text)
        unique_chars = set(normalized)
        invalid_chars = unique_chars - self.valid_chars
        return {
            'original_length': len(text),
            'normalized_length': len(normalized),
            'word_count': len(normalized.split()),
            'unique_chars': len(unique_chars),
            'invalid_chars': list(invalid_chars),
            'is_valid': len(invalid_chars) == 0
        }


def normalize_transcript_file(input_file: str, output_file: str) -> None:
    import pandas as pd
    normalizer = KurdishSoraniNormalizer()
    df = pd.read_csv(input_file)

    print("Normalizing transcripts...")
    df['transcript'] = df['transcript'].apply(normalizer.normalize_text)
    if "English Translation" in df.columns:
        df = df.drop(columns=["English Translation"])
    df = df[df['transcript'].str.len() > 0]
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} normalized transcripts to {output_file}")


if __name__ == "__main__":
    # Example test
    normalizer = KurdishSoraniNormalizer()
    test_texts = [
        "سڵاو‌، چۆنی؟",        # includes ZWNJ
        "٤٢ ساڵم.",             # Arabic digits
        "دەستور   و    یاسا",   # extra spacing
        "ئه‌و کتێبه‌",          # definite article normalization
    ]
    for text in test_texts:
        norm = normalizer.normalize_text(text)
        print(f"Original:   {text}")
        print(f"Normalized: {norm}")
        print("---")
