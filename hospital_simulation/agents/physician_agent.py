from typing import Dict, Any
from hospital_simulation.agents.base_agent import BaseAgent
from langchain.prompts import PromptTemplate

class PhysicianAgent(BaseAgent):
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        super().__init__(model_name)
        self._setup_prompt()

    def _setup_prompt(self):
        """Set up the physician's prompt."""
        prompt_template = """You are an experienced physician conducting a patient examination. Your role is to:
1. Review patient information thoroughly
2. Analyze symptoms and history
3. Provide initial diagnosis
4. Recommend necessary tests/imaging
5. Outline treatment plan

Patient Information:
- Name: {patient_info[name]}
- ID: {patient_info[patient_id]}
- Age: {patient_info[age]}
- Gender: {patient_info[gender]}

Current Symptoms:
{symptoms}

Medical History:
{medical_records}

Please provide a comprehensive assessment including:
1. Initial Evaluation
   - Review of symptoms
   - Physical examination findings
   - Vital signs needed

2. Preliminary Diagnosis
   - Primary concerns
   - Differential diagnoses
   - Risk factors

3. Recommended Tests
   - Laboratory tests needed
   - Imaging requirements (if any)
   - Specialist consultations (if needed)

4. Treatment Plan
   - Immediate interventions
   - Medications (if needed)
   - Follow-up requirements

5. Patient Instructions
   - Care instructions
   - Warning signs to watch for
   - Follow-up timeline

Provide your assessment in a clear, structured format, using medical terminology while ensuring patient understanding.
"""
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["patient_info", "symptoms", "medical_records"]
        )

    def examine_patient(self, patient_info: Dict[str, Any], symptoms: str, medical_records: str) -> Dict[str, str]:
        """Examine a patient and provide medical recommendations."""
        print("\n=== Physician Examination ===")
        print(f"Examining patient: {patient_info['name']}")
        print(f"Symptoms: {symptoms}")
        
        result = self.run({
            "patient_info": patient_info,
            "symptoms": symptoms,
            "medical_records": medical_records
        })
        
        print("Physician examination completed")
        return result 