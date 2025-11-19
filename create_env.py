"""
Create .env file with proper UTF-8 encoding
"""

def create_env_file():
    """Create .env file with proper encoding."""
    print("🔧 Creating .env file with proper encoding...")
    
    # Get API key from user
    api_key = input("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return
    
    try:
        # Create .env file with UTF-8 encoding
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        
        print("✅ .env file created successfully with UTF-8 encoding!")
        print("You can now run the demos:")
        print("  python examples/basic_example.py")
        print("  python examples/simple_nsga2_demo.py")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

if __name__ == "__main__":
    create_env_file()
