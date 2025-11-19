"""
Setup script to help configure your API key for RSBench-V3 demos.
"""

import os

def setup_api_key():
    """Interactive setup for API key."""
    print("🔑 RSBench-V3 API Key Setup")
    print("=" * 40)
    
    print("\nChoose your setup method:")
    print("1. Create .env file (recommended)")
    print("2. Set environment variable")
    print("3. Just show instructions")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        api_key = input("\nEnter your OpenAI API key: ").strip()
        if api_key:
            try:
                with open('.env', 'w', encoding='utf-8') as f:
                    f.write(f"OPENAI_API_KEY={api_key}\n")
                print("✅ .env file created successfully!")
                print("You can now run the demos:")
                print("  python examples/test_demo.py")
                print("  python examples/simple_nsga2_demo.py")
            except Exception as e:
                print(f"❌ Error creating .env file: {e}")
        else:
            print("❌ No API key provided")
    
    elif choice == "2":
        api_key = input("\nEnter your OpenAI API key: ").strip()
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            print("✅ Environment variable set for this session")
            print("Note: This will only work for the current terminal session")
        else:
            print("❌ No API key provided")
    
    elif choice == "3":
        print("\n📋 Manual Setup Instructions:")
        print("1. Get your API key from: https://platform.openai.com/api-keys")
        print("2. Create a .env file in the project root with:")
        print("   OPENAI_API_KEY=your_actual_api_key_here")
        print("3. Or edit examples/test_demo.py and set the api_key variable")
        print("4. Run: python examples/test_demo.py")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    setup_api_key()
