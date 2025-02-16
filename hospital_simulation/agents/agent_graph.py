from typing import Dict, Any, TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from hospital_simulation.agents.front_desk_agent import FrontDeskAgent
from hospital_simulation.agents.physician_agent import PhysicianAgent
from hospital_simulation.agents.radiologist_agent import RadiologistAgent

class Assessment(BaseModel):
    """Base model for agent assessments"""
    response: str = Field(default="")

class PatientState(BaseModel):
    """Type definition for patient state in the workflow using Pydantic."""
    patient_info: Dict[str, Any] = Field(default_factory=dict)
    complaint: str = Field(default="")
    medical_records: str = Field(default="")
    
    # Agent assessments with proper initialization
    front_desk_assessment: Optional[Assessment] = Field(default_factory=Assessment)
    physician_assessment: Optional[Assessment] = Field(default_factory=Assessment)
    radiology_report: Optional[Assessment] = Field(default_factory=Assessment)
    
    # Agent logs with proper initialization
    front_desk_logs: List[str] = Field(default_factory=list)
    physician_logs: List[str] = Field(default_factory=list)
    radiologist_logs: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True

class HospitalGraph:
    def __init__(self):
        self.front_desk = FrontDeskAgent()
        self.physician = PhysicianAgent()
        self.radiologist = RadiologistAgent()
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create the workflow graph connecting all agents."""
        # Create a new graph with Pydantic model
        workflow = StateGraph(PatientState)

        # Define the nodes
        workflow.add_node("front_desk", self._front_desk_node)
        workflow.add_node("physician", self._physician_node)
        workflow.add_node("radiologist", self._radiologist_node)

        # Define the edges
        workflow.add_edge("front_desk", "physician")
        workflow.add_conditional_edges(
            "physician",
            self._needs_imaging,
            {
                True: "radiologist",
                False: END
            }
        )
        workflow.add_edge("radiologist", END)

        # Set the entry point
        workflow.set_entry_point("front_desk")
        
        return workflow.compile()

    def _front_desk_node(self, state: PatientState) -> Dict[str, PatientState]:
        """Process patient at front desk."""
        try:
            result = self.front_desk.process_patient(
                state.patient_info,
                state.complaint
            )
            # Create a new Assessment with the response
            assessment = Assessment(response=result.get("response", ""))
            # Update state with the new assessment and logs
            state.front_desk_assessment = assessment
            state.front_desk_logs = result.get("logs", [])
            print(f"Front desk logs: {len(state.front_desk_logs)} entries")
            print(f"Front desk response length: {len(state.front_desk_assessment.response)}")
            return {"front_desk": state}
        except Exception as e:
            print(f"Error in front desk node: {e}")
            state.front_desk_logs = [f"Error in front desk processing: {str(e)}"]
            return {"front_desk": state}

    def _physician_node(self, state: PatientState) -> Dict[str, PatientState]:
        """Process patient with physician."""
        try:
            result = self.physician.examine_patient(
                state.patient_info,
                state.complaint,
                state.medical_records
            )
            # Create a new Assessment with the response
            assessment = Assessment(response=result.get("response", ""))
            # Update state with the new assessment and logs
            state.physician_assessment = assessment
            state.physician_logs = result.get("logs", [])
            print(f"Physician logs: {len(state.physician_logs)} entries")
            print(f"Physician response length: {len(state.physician_assessment.response)}")
            return {"physician": state}
        except Exception as e:
            print(f"Error in physician node: {e}")
            state.physician_logs = [f"Error in physician processing: {str(e)}"]
            return {"physician": state}

    def _radiologist_node(self, state: PatientState) -> Dict[str, PatientState]:
        """Process patient with radiologist if imaging is needed."""
        try:
            result = self.radiologist.analyze_imaging(
                state.patient_info,
                state.physician_assessment.response,
                state.medical_records
            )
            # Create a new Assessment with the response
            assessment = Assessment(response=result.get("response", ""))
            # Update state with the new assessment and logs
            state.radiology_report = assessment
            state.radiologist_logs = result.get("logs", [])
            print(f"Radiologist logs: {len(state.radiologist_logs)} entries")
            print(f"Radiologist response length: {len(state.radiology_report.response)}")
            return {"radiologist": state}
        except Exception as e:
            print(f"Error in radiologist node: {e}")
            state.radiologist_logs = [f"Error in radiologist processing: {str(e)}"]
            return {"radiologist": state}

    def _needs_imaging(self, state: PatientState) -> bool:
        """Determine if patient needs imaging based on physician assessment."""
        return "imaging" in state.physician_assessment.response.lower()

    def process_patient(self, 
                       patient_info: Dict[str, Any], 
                       complaint: str,
                       medical_records: str = "") -> Dict[str, Any]:
        """Process a patient through the hospital workflow."""
        try:
            # Initialize responses and logs dictionary
            responses = {
                "front_desk_assessment": "",
                "physician_assessment": "",
                "radiology_report": "",
                "front_desk_logs": [],
                "physician_logs": [],
                "radiologist_logs": []
            }
            
            # Process through front desk
            print("\n=== Front Desk Processing ===")
            front_desk_result = self.front_desk.process_patient(
                patient_info,
                complaint
            )
            responses["front_desk_assessment"] = front_desk_result.get("response", "")
            responses["front_desk_logs"] = front_desk_result.get("logs", [])
            
            # Process through physician
            print("\n=== Physician Examination ===")
            physician_result = self.physician.examine_patient(
                patient_info,
                complaint,
                medical_records
            )
            responses["physician_assessment"] = physician_result.get("response", "")
            responses["physician_logs"] = physician_result.get("logs", [])
            
            # Check if imaging is needed based on physician's response
            if "imaging" in physician_result.get("response", "").lower():
                print("\n=== Radiology Analysis ===")
                radiology_result = self.radiologist.analyze_imaging(
                    patient_info,
                    physician_result.get("response", ""),
                    medical_records
                )
                responses["radiology_report"] = radiology_result.get("response", "")
                responses["radiologist_logs"] = radiology_result.get("logs", [])
            
            # Format medical assessment in markdown
            medical_assessment = f"""
# Medical Assessment Report

## 1. Front Desk Assessment
{responses['front_desk_assessment']}

## 2. Physician Assessment
{responses['physician_assessment']}
"""
            if responses["radiology_report"]:
                medical_assessment += f"""
## 3. Radiology Report
{responses['radiology_report']}
"""

            # Format logs in markdown
            processing_logs = f"""
# Processing Logs

## Front Desk Logs
```
{chr(10).join(responses['front_desk_logs'])}
```

## Physician Logs
```
{chr(10).join(responses['physician_logs'])}
```
"""
            if responses["radiologist_logs"]:
                processing_logs += f"""
## Radiologist Logs
```
{chr(10).join(responses['radiologist_logs'])}
```
"""
            
            # Print debug information
            print("\nCollected Responses:")
            print(f"Front Desk: {len(responses['front_desk_assessment'])} chars")
            print(f"Physician: {len(responses['physician_assessment'])} chars")
            print(f"Radiology: {len(responses['radiology_report'])} chars")
            
            # Return formatted responses
            return {
                "front_desk_assessment": responses["front_desk_assessment"],
                "physician_assessment": responses["physician_assessment"],
                "radiology_report": responses["radiology_report"],
                "front_desk_logs": responses["front_desk_logs"],
                "physician_logs": responses["physician_logs"],
                "radiologist_logs": responses["radiologist_logs"],
                "formatted_assessment": medical_assessment,
                "formatted_logs": processing_logs
            }
            
        except Exception as e:
            print(f"\nError processing patient: {e}")
            error_msg = f"Error in processing: {str(e)}"
            error_assessment = f"""
# Error in Processing

An error occurred while processing the patient:
```
{error_msg}
```
"""
            return {
                "front_desk_assessment": error_msg,
                "physician_assessment": error_msg,
                "radiology_report": error_msg,
                "front_desk_logs": [error_msg],
                "physician_logs": [error_msg],
                "radiologist_logs": [error_msg],
                "formatted_assessment": error_assessment,
                "formatted_logs": error_assessment
            } 