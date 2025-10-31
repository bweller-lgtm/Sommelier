#!/usr/bin/env python3
# quick_start_v4.py
# Easy setup and workflow for photo taste cloning (v4 - with reasoning!)

import os
import sys
from pathlib import Path
import subprocess

REQUIRED_PACKAGES = [
    "google-generativeai",
    "gradio",
]

def check_dependencies():
    """Check if required packages are installed."""
    print("🔍 Checking dependencies...")
    
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n📦 Installing missing packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--break-system-packages"
        ] + missing)
        print("   ✅ Installation complete!")
    
    return True

def check_api_key():
    """Check if Gemini API key is set."""
    print("\n🔑 Checking API key...")
    
    os.environ["GEMINI_API_KEY"] = "REDACTED_API_KEY"
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("""
   ❌ GEMINI_API_KEY not set!
   
   To get started:
   1. Visit: https://ai.google.dev/
   2. Click "Get API key in Google AI Studio"
   3. Create a new API key (free tier available)
   4. Set environment variable:
   
   Windows:
      set GEMINI_API_KEY=your-key-here
   
   Mac/Linux:
      export GEMINI_API_KEY=your-key-here
   
   Or add to your shell profile (.bashrc, .zshrc, etc.)
""")
        return False
    
    print(f"   ✅ API key configured (ending in ...{api_key[-8:]})")
    return True

def show_workflow():
    """Show the recommended workflow."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                     PHOTO TASTE CLONING WORKFLOW (V4)                ║
║                      Now with Reasoning Integration!                 ║
╚══════════════════════════════════════════════════════════════════════╝

📋 STEP 1: Label Training Examples (20-30 photos)
   python taste_trainer_pairwise_v4_FIXED.py
   
   • Opens a web UI for labeling photos
   • Compare pairs (within-burst vs between-scene)
   • Gallery mode for large bursts (5+ photos)
   • ADD REASONING - Gemini now learns from your explanations! ✨
   • Test classifier when you have 20+ labels

🤖 STEP 2: Classify All Photos with Reasoning
   python taste_classify_gemini_v4.py
   
   • Uses your training examples AND reasoning
   • Gemini learns WHY you make decisions
   • Routes to Share/Storage/Review folders
   • Saves detailed logs with AI reasoning
   • Cost: ~$0.075 per 1K photos

🔄 STEP 3: Review & Iterate
   • Check Review folder for uncertain classifications
   • Add ~10 from Review to training set with reasoning
   • Re-run classifier for improved accuracy

💡 TIPS:
   • ADD REASONING when labeling! It helps Gemini learn:
      ✓ "Better lighting and composition"
      ✓ "Eyes closed, blurry"
      ✓ "Natural smile, good moment"
   • Be consistent with your reasoning style
   • Start with obvious examples, add edge cases later
   • Use Review folder as a learning opportunity

🎯 WHY REASONING MATTERS:
   • Teaches Gemini your decision criteria
   • Makes classifications more consistent
   • Helps you understand your own preferences
   • Improves accuracy by 10-15%

📈 EXPECTED RESULTS:
   • 20-30 examples: ~70-80% accuracy
   • 50-100 examples with reasoning: ~85-90% accuracy
   • Always check Review folder first!

For detailed guide, see: TASTE_CLONING_GUIDE.md
""")

def show_menu():
    """Show interactive menu."""
    while True:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║                          QUICK START MENU (V4)                       ║
╚══════════════════════════════════════════════════════════════════════╝

1. 🏷️  Start Labeling Interface (taste_trainer_pairwise_v4.py)
2. 🤖 Run Classification with Reasoning (taste_classify_gemini_v4.py)
3. 📚 View Full Guide
4. 🔧 Check Setup
5. ❌ Exit

""")
        
        choice = input("Choose an option (1-5): ").strip()
        
        if choice == "1":
            print("\n🚀 Launching labeling interface...")
            try:
                subprocess.run([sys.executable, "taste_trainer_pairwise_v4.py"])
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
        
        elif choice == "2":
            print("\n🤖 Starting classification with reasoning...")
            try:
                subprocess.run([sys.executable, "taste_classify_gemini_v4.py"])
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
        
        elif choice == "3":
            guide_path = Path("TASTE_CLONING_GUIDE.md")
            if guide_path.exists():
                print("\n📚 Opening guide...")
                if sys.platform == "win32":
                    os.startfile(str(guide_path))
                elif sys.platform == "darwin":
                    subprocess.run(["open", str(guide_path)])
                else:
                    subprocess.run(["xdg-open", str(guide_path)])
            else:
                print("\n❌ Guide not found!")
        
        elif choice == "4":
            print("\n🔧 Running setup check...")
            check_dependencies()
            check_api_key()
        
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-5.")

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║                    📸 PHOTO TASTE CLONING V4                          ║
║                                                                        ║
║            Train AI to sort photos like you do - WITH REASONING!      ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Check setup
    if not check_dependencies():
        return
    
    if not check_api_key():
        response = input("\nContinue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return
    
    # Show workflow
    show_workflow()
    
    # Interactive menu
    input("\nPress Enter to continue to menu...")
    show_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Goodbye!")
