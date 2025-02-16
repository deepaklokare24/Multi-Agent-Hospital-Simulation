from typing import Dict, Any
from hospital_simulation.agents.base_agent import BaseAgent
from langchain.prompts import PromptTemplate

class FrontDeskAgent(BaseAgent):
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        super().__init__(model_name)
        self._setup_prompt()

    def _setup_prompt(self):
        """Set up the front desk agent's prompt."""
        prompt_template = """You are a professional hospital front desk agent. Your role is to:
1. Greet patients professionally
2. Review their information and symptoms
3. Assess urgency level
4. Direct them to appropriate department
5. Provide clear instructions

Patient Information:
- Name: {patient_info[name]}
- ID: {patient_info[patient_id]}
- Age: {patient_info[age]}
- Gender: {patient_info[gender]}

Primary Complaints/Symptoms:
{complaint}

Based on the information provided, please:
1. Greet the patient professionally
2. Assess the urgency level (Low/Medium/High/Critical)
3. Recommend the appropriate department
4. Provide clear next steps
5. Note any special considerations

Respond in a professional, empathetic manner, structured with clear sections.
"""
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["patient_info", "complaint"]
        )

    def process_patient(self, patient_info: Dict[str, Any], complaint: str) -> Dict[str, str]:
        """Process a new patient and provide recommendations."""
        print("\n=== Front Desk Processing ===")
        print(f"Processing patient: {patient_info['name']}")
        print(f"Symptoms: {complaint}")
        
        result = self.run({
            "patient_info": patient_info,
            "complaint": complaint
        })
        
        print("Front desk assessment completed")
        return result 