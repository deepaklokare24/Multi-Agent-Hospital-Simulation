from typing import Dict, Any
from hospital_simulation.agents.base_agent import BaseAgent
from langchain.prompts import PromptTemplate

class RadiologistAgent(BaseAgent):
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        super().__init__(model_name)
        self._setup_prompt()

    def _setup_prompt(self):
        """Set up the radiologist's prompt."""
        prompt_template = """You are a specialized radiologist analyzing medical imaging results. Your role is to:
1. Review patient information
2. Analyze imaging findings
3. Provide detailed interpretation
4. Make clinical recommendations
5. Suggest follow-up imaging if needed

Patient Information:
- Name: {patient_info[name]}
- ID: {patient_info[patient_id]}
- Age: {patient_info[age]}
- Gender: {patient_info[gender]}

Clinical History:
{clinical_history}

Imaging Analysis Results:
{imaging_request}

Please provide a comprehensive radiology report including:
1. Examination Details
   - Type of imaging performed
   - Image quality assessment
   - Patient positioning

2. Findings
   - Anatomical structures
   - Abnormalities detected
   - Comparison with normal findings

3. Interpretation
   - Clinical significance
   - Correlation with symptoms
   - Differential considerations

4. Recommendations
   - Additional views needed
   - Follow-up imaging timeline
   - Clinical correlation advised

5. Impression
   - Summary of key findings
   - Level of concern
   - Critical findings (if any)

Provide your report in a clear, structured format using standard radiological terminology while ensuring clarity for other healthcare providers.
"""
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["patient_info", "imaging_request", "clinical_history"]
        )

    def analyze_imaging(self, patient_info: Dict[str, Any], imaging_request: str, clinical_history: str) -> Dict[str, str]:
        """Analyze imaging results and provide professional interpretation."""
        print("\n=== Radiology Analysis ===")
        print(f"Analyzing imaging for patient: {patient_info['name']}")
        
        # Add X-ray analysis results to the request
        if "PNEUMONIA" in imaging_request:
            imaging_request = f"X-ray analysis indicates high probability of pneumonia. Detailed results: {imaging_request}"
        elif "NORMAL" in imaging_request:
            imaging_request = f"X-ray analysis suggests normal findings. Detailed results: {imaging_request}"
        
        result = self.run({
            "patient_info": patient_info,
            "imaging_request": imaging_request,
            "clinical_history": clinical_history
        })
        
        print("Radiology analysis completed")
        return result 