#!/usr/bin/env python3
"""
Quick script to clean all contents under data/processed directory.
Useful for debugging and testing the pipeline.
"""

import shutil
import os
from pathlib import Path

def clean_processed_directory():
    """Clean all contents under data/processed directory."""
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        print("❌ data/processed directory does not exist")
        return
    
    print("🧹 Cleaning data/processed directory...")
    
    # List what will be cleaned
    print("📋 Contents to be removed:")
    for item in processed_dir.rglob("*"):
        if item.is_file():
            print(f"   📄 {item}")
        elif item.is_dir():
            print(f"   📁 {item}")
    
    # Confirm with user
    response = input("\n❓ Are you sure you want to delete all contents? (y/N): ")
    if response.lower() != 'y':
        print("❌ Operation cancelled")
        return
    
    try:
        # Remove all contents but keep the directory structure
        for item in processed_dir.iterdir():
            if item.is_file():
                item.unlink()
                print(f"🗑️  Deleted file: {item}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"🗑️  Deleted directory: {item}")
        
        # Recreate the directory structure
        print("\n📁 Recreating directory structure...")
        
        # Create main directories
        (processed_dir / "fluent_interviews").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviews" / "full_fluent_interviews").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviews" / "interviewer_fluent_interviews").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviews" / "interviewee_fluent_interviews").mkdir(exist_ok=True)
        
        # Create CSV directories
        (processed_dir / "fluent_interviewer_sentence_csv").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviewer_sentence_csv" / "validated_sentences").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviewer_sentence_csv" / "discarded_sentences").mkdir(exist_ok=True)
        
        (processed_dir / "fluent_interviewee_sentence_csv").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviewee_sentence_csv" / "validated_sentences").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviewee_sentence_csv" / "discarded_sentences").mkdir(exist_ok=True)
        
        print("✅ Successfully cleaned data/processed directory")
        print("📁 Directory structure recreated")
        
    except Exception as e:
        print(f"❌ Error cleaning directory: {e}")

def quick_clean():
    """Quick clean without confirmation - useful for automated testing."""
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        print("❌ data/processed directory does not exist")
        return
    
    print("🧹 Quick cleaning data/processed directory...")
    
    try:
        # Remove all contents but keep the directory structure
        for item in processed_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        # Recreate the directory structure
        (processed_dir / "fluent_interviews").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviews" / "full_fluent_interviews").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviews" / "interviewer_fluent_interviews").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviews" / "interviewee_fluent_interviews").mkdir(exist_ok=True)
        
        (processed_dir / "fluent_interviewer_sentence_csv").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviewer_sentence_csv" / "validated_sentences").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviewer_sentence_csv" / "discarded_sentences").mkdir(exist_ok=True)
        
        (processed_dir / "fluent_interviewee_sentence_csv").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviewee_sentence_csv" / "validated_sentences").mkdir(exist_ok=True)
        (processed_dir / "fluent_interviewee_sentence_csv" / "discarded_sentences").mkdir(exist_ok=True)
        
        print("✅ Quick clean completed")
        
    except Exception as e:
        print(f"❌ Error during quick clean: {e}")

def show_status():
    """Show current status of data/processed directory."""
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        print("❌ data/processed directory does not exist")
        return
    
    print("📊 Current status of data/processed:")
    
    # Count files in each directory
    fluent_interviews = processed_dir / "fluent_interviews"
    if fluent_interviews.exists():
        full_fluent = fluent_interviews / "full_fluent_interviews"
        interviewer_fluent = fluent_interviews / "interviewer_fluent_interviews"
        interviewee_fluent = fluent_interviews / "interviewee_fluent_interviews"
        
        print(f"   📁 fluent_interviews/")
        print(f"      📁 full_fluent_interviews/: {len(list(full_fluent.glob('*'))) if full_fluent.exists() else 0} files")
        print(f"      📁 interviewer_fluent_interviews/: {len(list(interviewer_fluent.glob('*'))) if interviewer_fluent.exists() else 0} files")
        print(f"      📁 interviewee_fluent_interviews/: {len(list(interviewee_fluent.glob('*'))) if interviewee_fluent.exists() else 0} files")
    
    # CSV directories
    interviewer_csv = processed_dir / "fluent_interviewer_sentence_csv"
    interviewee_csv = processed_dir / "fluent_interviewee_sentence_csv"
    
    if interviewer_csv.exists():
        validated = interviewer_csv / "validated_sentences"
        discarded = interviewer_csv / "discarded_sentences"
        print(f"   📁 fluent_interviewer_sentence_csv/")
        print(f"      📁 validated_sentences/: {len(list(validated.glob('*.csv'))) if validated.exists() else 0} CSV files")
        print(f"      📁 discarded_sentences/: {len(list(discarded.glob('*.csv'))) if discarded.exists() else 0} CSV files")
    
    if interviewee_csv.exists():
        validated = interviewee_csv / "validated_sentences"
        discarded = interviewee_csv / "discarded_sentences"
        print(f"   📁 fluent_interviewee_sentence_csv/")
        print(f"      📁 validated_sentences/: {len(list(validated.glob('*.csv'))) if validated.exists() else 0} CSV files")
        print(f"      📁 discarded_sentences/: {len(list(discarded.glob('*.csv'))) if discarded.exists() else 0} CSV files")

def main():
    """Main function with command line interface."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "clean":
            clean_processed_directory()
        elif command == "quick":
            quick_clean()
        elif command == "status":
            show_status()
        else:
            print("❌ Unknown command. Use: clean, quick, or status")
    else:
        # Interactive mode
        print("🧹 Data/Processed Directory Cleaner")
        print("=" * 40)
        print("1. Clean with confirmation")
        print("2. Quick clean (no confirmation)")
        print("3. Show status")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ")
        
        if choice == "1":
            clean_processed_directory()
        elif choice == "2":
            quick_clean()
        elif choice == "3":
            show_status()
        elif choice == "4":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
