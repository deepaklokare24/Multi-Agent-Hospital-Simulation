from hospital_simulation.interface.gradio_app import main
from hospital_simulation.utils.env_loader import load_environment

if __name__ == "__main__":
    try:
        print("Initializing Hospital Simulation System...")
        
        # Load environment variables first
        load_environment()
        
        # Start the application
        print("Starting Gradio interface...")
        main()
    except EnvironmentError as e:
        print(f"\nEnvironment Error: {e}")
        print("\nPlease set up your environment variables and try again.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nIf this is an API key error, make sure you have set your GROQ_API_KEY in the .env file.") 