from dotenv import load_dotenv
import os
from pathlib import Path

def load_environment():
    """Load environment variables from .env file."""
    # Find the project root (where .env should be located)
    current_dir = Path(__file__).resolve()
    project_root = current_dir.parent.parent.parent  # Go up three levels to project root
    env_path = project_root / '.env'

    if not env_path.exists():
        raise EnvironmentError(
            f"No .env file found at {env_path}. Please create one with your GROQ_API_KEY. "
            "Example: GROQ_API_KEY=your-api-key-here"
        )

    # Load the environment variables
    load_dotenv(env_path)

    # Verify required variables
    required_vars = ['GROQ_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}. "
            f"Please add them to your .env file at {env_path}"
        )
    
    # Print confirmation
    print(f"Loaded environment variables from {env_path}")
    print(f"GROQ_API_KEY is {'set' if os.getenv('GROQ_API_KEY') else 'not set'}") 