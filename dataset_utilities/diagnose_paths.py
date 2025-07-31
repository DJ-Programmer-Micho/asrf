"""
Diagnostic script to identify and fix audio path issues
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config_loader import load_config

def diagnose_audio_paths():
    config = load_config()
    
    # Load the metadata
    csv_path = config["dataset"]["preprocessed_data_path_csv"]
    df = pd.read_csv(csv_path)
    
    base_dir = Path(config["dataset"]["audio_path"])
    print(f"Base audio directory: {base_dir}")
    print(f"Base directory exists: {base_dir.exists()}")
    
    if base_dir.exists():
        print(f"Contents of base directory:")
        for item in base_dir.iterdir():
            print(f"  {item}")
    
    print(f"\nChecking first 10 audio file paths:")
    
    missing_count = 0
    found_count = 0
    
    for i, (_, row) in enumerate(df.head(10).iterrows()):
        original_path = row['path']
        
        # Try different path constructions
        path_variations = [
            base_dir / original_path,
            base_dir / original_path.lstrip('/'),
            base_dir / original_path.lstrip('./'),
            Path(original_path),
            Path(original_path.replace('\\', '/')),
            Path(original_path.replace('/', '\\')),
        ]
        
        print(f"\n{i+1}. Original path: '{original_path}'")
        
        found = False
        for j, path_var in enumerate(path_variations):
            exists = path_var.exists()
            print(f"   Variation {j+1}: {path_var} -> {'EXISTS' if exists else 'MISSING'}")
            if exists and not found:
                found = True
        
        if found:
            found_count += 1
        else:
            missing_count += 1
    
    print(f"\nSummary of first 10 files:")
    print(f"  Found: {found_count}")
    print(f"  Missing: {missing_count}")
    
    # Check if there are common patterns
    print(f"\nPath patterns in dataset:")
    sample_paths = df['path'].head(5).tolist()
    for path in sample_paths:
        print(f"  '{path}'")
    
    return df, base_dir

def suggest_path_fix(df, base_dir):
    """Suggest the correct path construction method"""
    
    print(f"\nTesting path construction methods...")
    
    methods = {
        'method1': lambda p: base_dir / p,
        'method2': lambda p: base_dir / p.lstrip('/'),
        'method3': lambda p: base_dir / p.lstrip('./'),
        'method4': lambda p: Path(p),
        'method5': lambda p: base_dir / Path(p).name,  # Just filename
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        found = 0
        for _, row in df.head(20).iterrows():
            try:
                path = method_func(row['path'])
                if path.exists():
                    found += 1
            except:
                pass
        results[method_name] = found
        print(f"  {method_name}: {found}/20 files found")
    
    best_method = max(results, key=results.get)
    print(f"\nBest method: {best_method} with {results[best_method]}/20 files found")
    
    return best_method, results

if __name__ == "__main__":
    df, base_dir = diagnose_audio_paths()
    best_method, results = suggest_path_fix(df, base_dir)