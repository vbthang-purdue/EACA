# extract_features.py
import os
import sys
import argparse
from config import Config
from src.batch_feature_extractor import BatchFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description='Extract multimodal features from MELD dataset')
    parser.add_argument('--split', type=str, default='all', 
                       choices=['all', 'train', 'dev', 'test'],
                       help='Which split to process (default: all)')
    parser.add_argument('--format', type=str, default='pickle',
                       choices=['pickle', 'numpy'],
                       help='Save format (default: pickle)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.SAVE_FORMAT = args.format
    config.VERBOSE = args.verbose
    
    # Initialize feature extractor
    extractor = BatchFeatureExtractor(config)
    
    if args.split == 'all':
        # Process all splits
        extractor.extract_all_features()
    else:
        # Process specific split
        features = extractor.extract_features_for_split(args.split)
        extractor.save_features(features, args.split)
        extractor.print_feature_summary(features, args.split)
    
    print("\n🎉 Feature extraction completed!")

if __name__ == "__main__":
    main()