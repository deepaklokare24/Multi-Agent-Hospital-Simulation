# Multi-Agent Hospital Simulation System

A sophisticated healthcare simulation system demonstrating the practical application of multiple AI agents working together to process and manage patient care. This system showcases the integration of modern AI technologies with medical workflows to create a realistic and interactive hospital experience.

## 🌟 Key Features

- **Multi-Agent System Architecture**
  - Front Desk Agent: Patient intake and urgency assessment
  - Physician Agent: Medical examination and diagnosis
  - Radiologist Agent: Medical imaging analysis and reporting
  - Seamless inter-agent communication and workflow management

- **Advanced Medical Processing**
  - Real-time medical image analysis
  - Comprehensive patient assessments
  - Dynamic medical report generation
  - Structured medical knowledge representation

- **Interactive User Interface**
  - Modern Gradio-based web interface
  - Real-time patient profile generation
  - Medical image visualization
  - Detailed processing logs
  - Professional medical reporting

## 🛠️ Technical Stack

- **Core Technologies**
  - LangGraph: Agent orchestration and workflow management
  - Groq LLM: llama-3.3-70b-versatile model for agent intelligence
  - Gradio: Interactive web interface
  - ChromaDB: Medical knowledge storage
  - Transformers: Vision analysis for medical imaging

- **Data Sources**
  - Kaggle disease-symptoms dataset
  - Chest X-ray classification dataset
  - Synthetic patient identity generation
  - Custom medical knowledge base

## 📋 Prerequisites

- Python 3.8+
- GROQ API key
- Internet connection for external API access
- Sufficient storage for medical datasets

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MultiAgentHospitalSimulation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with:
```
GROQ_API_KEY=your_groq_api_key_here
```

## 🎮 Usage

1. Start the application:
```bash
python -m hospital_simulation
```

2. Access the web interface:
   - Open your browser and navigate to http://localhost:7860
   - The interface will guide you through the patient simulation process

3. Workflow Steps:
   - Generate or input patient information
   - Upload or generate medical images
   - Process through the hospital workflow
   - Review comprehensive medical assessments

## 📁 Project Structure

```
hospital_simulation/
├── agents/                 # AI agent implementations
│   ├── base_agent.py      # Base agent functionality
│   ├── front_desk_agent.py
│   ├── physician_agent.py
│   ├── radiologist_agent.py
│   └── agent_graph.py     # Workflow orchestration
├── data/                  # Data management
│   ├── prepare_data.py    # Dataset preparation
│   └── preprocess_data.py # Data preprocessing
├── database/             # Medical knowledge storage
│   └── medical_db.py     # Database operations
├── interface/            # User interface
│   └── gradio_app.py     # Web interface
├── vision/               # Image processing
│   └── patient_analysis.py # Medical image analysis
└── utils/               # Utility functions
    └── env_loader.py    # Environment management
```

## 🔄 Workflow

1. **Patient Intake**
   - Generate random patient or input details
   - Initial symptom assessment
   - Urgency level determination

2. **Medical Examination**
   - Comprehensive patient evaluation
   - Preliminary diagnosis
   - Treatment recommendations

3. **Imaging Analysis**
   - X-ray image processing
   - Professional radiology reporting
   - Follow-up recommendations

4. **Report Generation**
   - Structured medical assessments
   - Detailed agent processing logs
   - Professional medical documentation

## 🎯 Data Preparation

Before running the main application, you need to prepare the initial dataset. Our automated data preparation pipeline handles everything from data download to synthetic data generation.

### 1. Run Data Preparation

Simply run the data preparation script:
```bash
python -m hospital_simulation.data.prepare_data
```

This script automatically performs the following tasks:

1. **Dataset Initialization**
   - Automatically downloads required datasets using Kaggle API
   - Initializes necessary directory structures
   - Sets up data processing pipeline

2. **Medical Dataset Processing**
   - Loads disease-symptoms dataset
   - Preprocesses and structures medical data
   - Creates standardized symptom profiles
   - Generates preprocessed_patients.csv

3. **Synthetic Patient Generation**
   - Leverages Groq LLM for generating realistic patient identities
   - Ensures demographic diversity
   - Creates consistent patient IDs and medical records
   - Maintains data consistency with medical profiles

4. **X-ray Dataset Integration**
   - Downloads and processes chest X-ray classification dataset
   - Categorizes images by condition (Normal/Pneumonia)
   - Prepares vision analysis pipeline
   - Sets up example image repository

The script creates the following directory structure:
```bash
hospital_simulation/data/
├── processed/              # Processed patient data
│   └── preprocessed_patients.csv
├── example_images/        # Example X-ray images
│   ├── xray_normal_*.jpg
│   └── xray_pneumonia_*.jpg
└── chroma/               # Vector database
    └── medical_knowledge.db
```

### 2. Monitor Progress

During data preparation, you'll see progress indicators for:
- Dataset downloads and processing
- Patient identity generation
- X-ray image processing
- Database initialization

The entire process typically takes 10-15 minutes, depending on your system and internet connection. The script handles all necessary downloads and preprocessing automatically.

### 3. Verification

After the preparation is complete, verify the setup by checking:
1. The existence of `preprocessed_patients.csv` in the processed directory
2. Example X-ray images in the example_images directory
3. Initialized ChromaDB database

If any step fails, the script will provide detailed error messages and suggestions for resolution.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Kaggle for providing medical datasets
- Groq for LLM API access
- LangChain & LangGraph communities
- Medical professionals for workflow insights 