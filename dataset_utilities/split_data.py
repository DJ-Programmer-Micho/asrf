"""
Split Kurdish Sorani ASR dataset into train/validation/test sets.
Ensures proper stratification and balanced distribution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetSplitter:
    def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """
        Initialize dataset splitter.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set  
            test_ratio: Ratio for test set
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
    def analyze_dataset(self, df: pd.DataFrame, audio_path_col: str = 'path', 
                       transcript_col: str = 'transcript') -> Dict:
        """
        Analyze dataset characteristics for informed splitting.
        
        Args:
            df: DataFrame with audio paths and transcripts
            audio_path_col: Column name for audio file paths
            transcript_col: Column name for transcripts
            
        Returns:
            Dictionary with dataset statistics
        """
        import librosa
        
        print("Analyzing dataset characteristics...")
        
        # Basic statistics
        total_samples = len(df)
        
        # Transcript length analysis
        df['transcript_length'] = df[transcript_col].str.len()
        df['word_count'] = df[transcript_col].str.split().str.len()
        
        # Audio duration analysis (sample a subset for speed)
        sample_size = min(1000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        durations = []
        for _, row in sample_df.iterrows():
            try:
                audio_path = row[audio_path_col]
                if Path(audio_path).exists():
                    # Get duration without loading full audio
                    duration = librosa.get_duration(path=audio_path)
                    durations.append(duration)
            except Exception as e:
                print(f"Warning: Could not get duration for {audio_path}: {e}")
        
        # Statistics
        stats = {
            'total_samples': total_samples,
            'transcript_stats': {
                'min_length': df['transcript_length'].min(),
                'max_length': df['transcript_length'].max(),
                'mean_length': df['transcript_length'].mean(),
                'median_length': df['transcript_length'].median(),
            },
            'word_count_stats': {
                'min_words': df['word_count'].min(),
                'max_words': df['word_count'].max(), 
                'mean_words': df['word_count'].mean(),
                'median_words': df['word_count'].median(),
            },
            'audio_duration_stats': {
                'sample_size': len(durations),
                'min_duration': np.min(durations) if durations else 0,
                'max_duration': np.max(durations) if durations else 0,
                'mean_duration': np.mean(durations) if durations else 0,
                'median_duration': np.median(durations) if durations else 0,
            }
        }
        
        return stats
    
    def create_duration_bins(self, df: pd.DataFrame, audio_path_col: str = 'path', 
                           num_bins: int = 5) -> pd.Series:
        """
        Create duration bins for stratified splitting.
        
        Args:
            df: DataFrame with audio paths
            audio_path_col: Column name for audio paths
            num_bins: Number of duration bins
            
        Returns:
            Series with bin labels
        """
        import librosa
        
        print(f"Creating {num_bins} duration bins for stratified splitting...")
        
        durations = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            try:
                audio_path = row[audio_path_col]
                if Path(audio_path).exists():
                    duration = librosa.get_duration(path=audio_path)
                    durations.append(duration)
                    valid_indices.append(idx)
                else:
                    print(f"Warning: Audio file not found: {audio_path}")
            except Exception as e:
                print(f"Warning: Could not process {audio_path}: {e}")
        
        # Create bins
        duration_bins = pd.cut(durations, bins=num_bins, labels=[f'bin_{i}' for i in range(num_bins)])
        
        # Create series for all indices
        all_bins = pd.Series(index=df.index, dtype='object')
        all_bins.loc[valid_indices] = duration_bins
        all_bins.fillna('unknown', inplace=True)
        
        print(f"Duration bin distribution:")
        print(all_bins.value_counts())
        
        return all_bins
    
    def split_dataset(self, df: pd.DataFrame, stratify_by: str = None, 
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            df: DataFrame to split
            stratify_by: Column name to stratify by (optional)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"Splitting dataset of {len(df)} samples...")
        print(f"Train: {self.train_ratio:.1%}, Val: {self.val_ratio:.1%}, Test: {self.test_ratio:.1%}")
        
        # First split: train vs (val + test)
        train_size = self.train_ratio
        temp_size = self.val_ratio + self.test_ratio
        
        stratify_col = df[stratify_by] if stratify_by else None
        
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            stratify=stratify_col,
            random_state=random_state,
            shuffle=True
        )
        
        # Second split: val vs test from temp
        val_size = self.val_ratio / temp_size
        
        if stratify_by:
            temp_stratify = temp_df[stratify_by]
        else:
            temp_stratify = None
            
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            stratify=temp_stratify,
            random_state=random_state,
            shuffle=True
        )
        
        print(f"Split results:")
        print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df):.1%})")
        print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df):.1%})")
        print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df):.1%})")
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   test_df: pd.DataFrame, output_dir: str) -> None:
        """
        Save dataset splits to separate CSV files.
        
        Args:
            train_df: Training set DataFrame
            val_df: Validation set DataFrame
            test_df: Test set DataFrame
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        train_path = output_path / "train.csv"
        val_path = output_path / "validation.csv"
        test_path = output_path / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Saved dataset splits to {output_dir}:")
        print(f"  {train_path} ({len(train_df)} samples)")
        print(f"  {val_path} ({len(val_df)} samples)")
        print(f"  {test_path} ({len(test_df)} samples)")
    
    def plot_split_analysis(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, stratify_by: str = None, 
                           output_dir: str = None) -> None:
        """
        Create visualizations of the dataset splits.
        
        Args:
            train_df: Training set DataFrame
            val_df: Validation set DataFrame
            test_df: Test set DataFrame
            stratify_by: Column used for stratification
            output_dir: Directory to save plots (optional)
        """
        try:
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Dataset Split Analysis', fontsize=16)
            
            # Split sizes
            sizes = [len(train_df), len(val_df), len(test_df)]
            labels = ['Train', 'Validation', 'Test']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Dataset Split Distribution')
            
            # Transcript length distribution
            bins = 50
            axes[0, 1].hist(train_df['transcript_length'], bins=bins, alpha=0.7, label='Train', color=colors[0])
            axes[0, 1].hist(val_df['transcript_length'], bins=bins, alpha=0.7, label='Val', color=colors[1])
            axes[0, 1].hist(test_df['transcript_length'], bins=bins, alpha=0.7, label='Test', color=colors[2])
            axes[0, 1].set_xlabel('Transcript Length (characters)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Transcript Length Distribution')
            axes[0, 1].legend()
            
            # Word count distribution
            axes[1, 0].hist(train_df['word_count'], bins=bins, alpha=0.7, label='Train', color=colors[0])
            axes[1, 0].hist(val_df['word_count'], bins=bins, alpha=0.7, label='Val', color=colors[1])
            axes[1, 0].hist(test_df['word_count'], bins=bins, alpha=0.7, label='Test', color=colors[2])
            axes[1, 0].set_xlabel('Word Count')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Word Count Distribution')
            axes[1, 0].legend()
            
            # Stratification distribution (if applicable)
            if stratify_by and stratify_by in train_df.columns:
                # Combine all splits for comparison
                all_data = []
                for df, split_name in [(train_df, 'Train'), (val_df, 'Val'), (test_df, 'Test')]:
                    for _, row in df.iterrows():
                        all_data.append({'split': split_name, stratify_by: row[stratify_by]})
                
                plot_df = pd.DataFrame(all_data)
                sns.countplot(data=plot_df, x=stratify_by, hue='split', ax=axes[1, 1])
                axes[1, 1].set_title(f'Stratification by {stratify_by}')
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, 'No stratification\ncolumn provided', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Stratification Analysis')
            
            plt.tight_layout()
            
            if output_dir:
                plot_path = Path(output_dir) / "split_analysis.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Saved split analysis plot to {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
    
    def validate_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                       test_df: pd.DataFrame, audio_path_col: str = 'path') -> Dict:
        """
        Validate the dataset splits.
        
        Args:
            train_df: Training set DataFrame
            val_df: Validation set DataFrame  
            test_df: Test set DataFrame
            audio_path_col: Column name for audio paths
            
        Returns:
            Dictionary with validation results
        """
        print("Validating dataset splits...")
        
        # Check for overlapping audio files
        train_files = set(train_df[audio_path_col])
        val_files = set(val_df[audio_path_col])
        test_files = set(test_df[audio_path_col])
        
        train_val_overlap = train_files.intersection(val_files)
        train_test_overlap = train_files.intersection(test_files)
        val_test_overlap = val_files.intersection(test_files)
        
        # Check for missing files
        all_files = list(train_files) + list(val_files) + list(test_files)
        missing_files = []
        for file_path in all_files[:100]:  # Check first 100 files
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        validation_results = {
            'total_files': len(all_files),
            'unique_files': len(set(all_files)),
            'overlaps': {
                'train_val': len(train_val_overlap),
                'train_test': len(train_test_overlap),
                'val_test': len(val_test_overlap)
            },
            'missing_files': len(missing_files),
            'missing_files_sample': missing_files[:10]
        }
        
        # Print validation results
        print(f"Validation Results:")
        print(f"  Total files: {validation_results['total_files']}")
        print(f"  Unique files: {validation_results['unique_files']}")
        print(f"  Overlaps:")
        print(f"    Train-Val: {validation_results['overlaps']['train_val']}")
        print(f"    Train-Test: {validation_results['overlaps']['train_test']}")
        print(f"    Val-Test: {validation_results['overlaps']['val_test']}")
        print(f"  Missing files (from sample): {validation_results['missing_files']}")
        
        if any(validation_results['overlaps'].values()):
            print("WARNING: Found overlapping files between splits!")
        if validation_results['missing_files']:
            print("WARNING: Found missing audio files!")
        
        return validation_results


def main():
    parser = argparse.ArgumentParser(description='Split Kurdish Sorani ASR dataset')
    parser.add_argument('--input_csv', type=str, default='dataset/preprocessed_data/metadata.csv',
                       help='Input CSV file with dataset')
    # parser.add_argument('--input_csv', type=str, required=True,
    #                    help='Input CSV file with dataset')
    parser.add_argument('--output_dir', type=str, default='dataset/prepared_splits',
                       help='Output directory for split files')
    # parser.add_argument('--output_dir', type=str, required=True,
    #                    help='Output directory for split files')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--stratify_by_duration', action='store_true',
                       help='Stratify by audio duration bins')
    parser.add_argument('--duration_bins', type=int, default=5,
                       help='Number of duration bins for stratification')
    parser.add_argument('--audio_path_col', type=str, default='path',
                       help='Column name for audio file paths')
    parser.add_argument('--transcript_col', type=str, default='transcript',
                       help='Column name for transcripts')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--create_plots', action='store_true',
                       help='Create analysis plots')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples")
    
    # Initialize splitter
    splitter = DatasetSplitter(args.train_ratio, args.val_ratio, args.test_ratio)
    
    # Analyze dataset
    stats = splitter.analyze_dataset(df, args.audio_path_col, args.transcript_col)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create stratification column if requested
    stratify_by = None
    if args.stratify_by_duration:
        df['duration_bin'] = splitter.create_duration_bins(df, args.audio_path_col, args.duration_bins)
        stratify_by = 'duration_bin'
    
    # Split dataset
    train_df, val_df, test_df = splitter.split_dataset(df, stratify_by, args.random_state)
    
    # Save splits
    splitter.save_splits(train_df, val_df, test_df, args.output_dir)
    
    # Validate splits
    validation = splitter.validate_splits(train_df, val_df, test_df, args.audio_path_col)
    
    # Create plots if requested
    if args.create_plots:
        splitter.plot_split_analysis(train_df, val_df, test_df, stratify_by, args.output_dir)


if __name__ == "__main__":
    main()