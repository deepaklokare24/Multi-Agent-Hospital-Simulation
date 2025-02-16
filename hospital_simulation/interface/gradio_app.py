import gradio as gr
import json
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import random
from pathlib import Path
import requests
from io import BytesIO
from datasets import load_dataset

from hospital_simulation.agents.agent_graph import HospitalGraph
from hospital_simulation.data.preprocess_data import DataPreprocessor
from hospital_simulation.vision.patient_analysis import PatientImageAnalysis

class HospitalInterface:
    def __init__(self):
        self.hospital = HospitalGraph()
        self.data_prep = DataPreprocessor()
        self.vision_analysis = PatientImageAnalysis()
        
        # Set up example images directory
        self.example_images_dir = Path(__file__).parent.parent / "data" / "example_images"
        self.example_images_dir.mkdir(exist_ok=True)
        
        # Load X-ray dataset
        print("Loading X-ray dataset...")
        self.x_ray_dataset = load_dataset("keremberke/chest-xray-classification", name="full")
        
        # Initialize interface
        self.interface = self._create_interface()
        self.current_patient = None

    def _get_random_xray(self, condition: str = None) -> Optional[Image.Image]:
        """Get a random X-ray image from the dataset."""
        try:
            if condition:
                # Filter by condition
                filtered_images = [
                    img for img in self.x_ray_dataset["train"] 
                    if img["disease"] == ("PNEUMONIA" if condition.upper() == "POSITIVE" else "NORMAL")
                ]
                if filtered_images:
                    return random.choice(filtered_images)["image"]
            
            # If no condition or no matching images, return random image
            random_index = random.randint(0, len(self.x_ray_dataset["train"]) - 1)
            return self.x_ray_dataset["train"][random_index]["image"]
        except Exception as e:
            print(f"Error getting random X-ray: {e}")
            print(f"Dataset structure: {self.x_ray_dataset['train'].features}")
        return None

    def _get_random_person_image(self, gender: str = None) -> Optional[Image.Image]:
        """Fetch a random person image."""
        try:
            # Using thispersondoesnotexist.com directly as it doesn't support parameters
            response = requests.get("https://thispersondoesnotexist.com/", timeout=10)
            if response.status_code == 200:
                print("Successfully fetched random person image")
                return Image.open(BytesIO(response.content))
            else:
                print(f"Failed to fetch image: {response.status_code}")
        except Exception as e:
            print(f"Error fetching random person image: {e}")
        return None

    def _format_output(self, result: Dict[str, Any]) -> str:
        """Format the output for display."""
        output = []
        
        if "front_desk_assessment" in result:
            output.append("=== Front Desk Assessment ===")
            output.append(result["front_desk_assessment"])
            output.append("")
        
        if "physician_assessment" in result:
            output.append("=== Physician Assessment ===")
            output.append(result["physician_assessment"])
            output.append("")
        
        if "radiology_report" in result:
            output.append("=== Radiology Report ===")
            output.append(result["radiology_report"])
        
        return "\n".join(output)

    def _format_logs(self, result: Dict[str, Any]) -> str:
        """Format the logs for display."""
        logs = []
        
        # Add front desk logs
        if "front_desk_logs" in result:
            logs.append("\n=== Front Desk Agent Logs ===")
            logs.extend(result["front_desk_logs"])
            
        # Add physician logs
        if "physician_logs" in result:
            logs.append("\n=== Physician Agent Logs ===")
            logs.extend(result["physician_logs"])
            
        # Add radiologist logs
        if "radiologist_logs" in result:
            logs.append("\n=== Radiologist Agent Logs ===")
            logs.extend(result["radiologist_logs"])
            
        return "\n".join(logs) if logs else "No processing logs available."

    def _create_interface(self) -> gr.Blocks:
        """Create the Gradio interface for the hospital simulation."""
        
        def load_random_patient(age_min: int, age_max: int, gender: str, ethnicity: str, condition: str) -> Tuple[str, str, str, str, str, str, Optional[str], Optional[str]]:
            """Load a random patient matching the criteria."""
            conditions = {
                "Gender": gender,
                "Age": (age_min, age_max)
            }
            if condition:
                conditions["Outcome Variable"] = condition

            print(f"Loading patient with conditions: {conditions}")
            patient = self.data_prep.get_random_patient(conditions)
            
            # Create a dictionary with patient data including ethnicity
            self.current_patient = {
                "First_Name": patient["First_Name"],
                "Last_Name": patient["Last_Name"],
                "Patient_ID": patient["Patient_ID"],
                "Age": patient["Age"],
                "Gender": patient["Gender"],
                "ethnicity": ethnicity  # Store the selected ethnicity
            }
            
            # Get example images if available
            patient_photo = None
            xray_image = None
            
            # Try to get matching example images
            try:
                if condition == "Positive":
                    xray_files = list(self.example_images_dir.glob("xray_pneumonia_*.jpg"))
                    if xray_files:
                        xray_image = str(random.choice(xray_files))
                        print(f"Loaded pneumonia X-ray: {xray_image}")
                elif condition == "Negative":
                    xray_files = list(self.example_images_dir.glob("xray_normal_*.jpg"))
                    if xray_files:
                        xray_image = str(random.choice(xray_files))
                        print(f"Loaded normal X-ray: {xray_image}")
            except Exception as e:
                print(f"Error loading example images: {e}")

            # Return values in the order expected by Gradio outputs
            name = f"{patient['First_Name']} {patient['Last_Name']}"
            patient_id = patient["Patient_ID"]
            age = str(patient["Age"])
            gender = patient["Gender"]
            symptoms = ", ".join([
                col for col in patient.index 
                if patient[col] == "Yes" and col not in 
                ["Gender", "First_Name", "Last_Name", "Patient_ID", "Age", "Outcome Variable"]
            ])

            print(f"Loaded patient: {name} (ID: {patient_id})")
            print(f"Symptoms: {symptoms}")
            print(f"Ethnicity: {ethnicity}")
            
            return name, patient_id, age, gender, ethnicity, symptoms, patient_photo, xray_image

        def process_patient(name: str, patient_id: str, age: str, gender: str, ethnicity: str, symptoms: str,
                          patient_photo: Optional[Image.Image] = None,
                          xray_image: Optional[Image.Image] = None) -> Tuple[str, Dict, Dict, str]:
            """Process patient information and return results."""
            if not all([name, patient_id, age, gender, ethnicity, symptoms]):
                return "Error: Please load a patient first", {}, {}, "No logs available - patient not processed"

            # Prepare patient info
            patient_info = {
                "name": name,
                "patient_id": patient_id,
                "age": age,
                "gender": gender,
                "ethnicity": ethnicity
            }

            try:
                # Analyze patient photo if provided
                photo_analysis = {}
                if patient_photo is not None:
                    print("Analyzing patient photo...")
                    photo_analysis = self.vision_analysis.analyze_patient_photo(patient_photo)
                    print("Photo analysis results:", photo_analysis)

                # Analyze X-ray if provided
                xray_analysis = {}
                if xray_image is not None:
                    print("Analyzing X-ray image...")
                    xray_analysis = self.vision_analysis.analyze_xray(xray_image)
                    print("X-ray analysis results:", xray_analysis)

                # Process through hospital workflow
                print("Processing through hospital workflow...")
                result = self.hospital.process_patient(
                    patient_info=patient_info,
                    complaint=symptoms,
                    medical_records=""
                )
                
                # Return formatted outputs
                return (
                    result.get("formatted_assessment", "Error: No assessment available"),
                    photo_analysis,
                    xray_analysis,
                    result.get("formatted_logs", "Error: No logs available")
                )
                
            except Exception as e:
                error_msg = f"Error processing patient: {str(e)}"
                print(error_msg)
                return error_msg, photo_analysis, xray_analysis, f"Error in processing: {str(e)}"

        def generate_person_photo(gender: str, ethnicity: str) -> Optional[Image.Image]:
            """Generate a random person photo based on gender and ethnicity."""
            # Map ethnicity to API format
            ethnicity_map = {
                "Asian": "asian",
                "Black": "black",
                "White": "white",
                "Indian": "indian",
                "Middle Eastern": "middle",
                "Latino": "latino"
            }
            self.current_patient["ethnicity"] = ethnicity_map.get(ethnicity, "asian")
            return self._get_random_person_image(gender)

        def generate_xray(condition: str) -> Optional[Image.Image]:
            """Generate a random X-ray based on condition."""
            return self._get_random_xray(condition)

        # Create the interface
        with gr.Blocks(title="Multi-Agent Hospital System") as blocks:
            with gr.Tabs():
                with gr.Tab("Patient Assessment"):
                    gr.Markdown("""
                    # Multi-Agent Hospital System
                    
                    ## Instructions:
                    1. Use the controls below to set patient criteria
                    2. Click 'Load Random Patient' to get a matching patient
                    3. Use 'Generate' buttons to create random images or upload your own
                    4. Click 'Process Patient' to run the analysis
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            age_min = gr.Slider(minimum=0, maximum=100, value=20, label="Minimum Age")
                            age_max = gr.Slider(minimum=0, maximum=100, value=29, label="Maximum Age")
                            gender = gr.Radio(choices=["Male", "Female"], value="Female", label="Gender")
                            ethnicity = gr.Radio(
                                choices=["Asian", "Black", "White", "Indian", "Middle Eastern", "Latino"],
                                value="Asian",
                                label="Ethnicity"
                            )
                            condition = gr.Radio(choices=["Positive", "Negative", ""], value="", label="Condition")
                            load_btn = gr.Button("Load Random Patient", variant="primary")

                        with gr.Column():
                            name = gr.Textbox(label="Patient Name", interactive=False)
                            patient_id = gr.Textbox(label="Patient ID", interactive=False)
                            age = gr.Textbox(label="Age", interactive=False)
                            gender_display = gr.Textbox(label="Gender", interactive=False)
                            ethnicity_display = gr.Textbox(label="Ethnicity", interactive=False)
                            symptoms = gr.Textbox(label="Symptoms", interactive=False)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Patient Photo")
                            patient_photo = gr.Image(label="Patient photo", type="pil")
                            generate_photo_btn = gr.Button("Generate Random Photo", variant="secondary")
                        
                        with gr.Column():
                            gr.Markdown("### X-Ray Image")
                            xray_image = gr.Image(label="X-ray image", type="pil")
                            generate_xray_btn = gr.Button("Generate Random X-ray", variant="secondary")

                    process_btn = gr.Button("Process Patient", variant="primary")

                    with gr.Row():
                        with gr.Column():
                            assessment = gr.Markdown(
                                label="Medical Assessment",
                                value="No assessment available yet. Click 'Process Patient' to begin.",
                                elem_id="medical_assessment"
                            )
                        with gr.Column():
                            with gr.Row():
                                photo_analysis = gr.JSON(
                                    label="Photo Analysis",
                                    elem_id="photo_analysis"
                                )
                            with gr.Row():
                                xray_analysis = gr.JSON(
                                    label="X-Ray Analysis",
                                    elem_id="xray_analysis"
                                )

                with gr.Tab("Processing Logs"):
                    gr.Markdown("""
                    # Processing Logs
                    
                    This tab shows detailed logs of the processing steps, including:
                    - Model information
                    - Agent reasoning steps
                    - Input/output for each agent
                    - Error handling and retries
                    """)
                    logs_display = gr.Markdown(
                        value="No logs available yet. Process a patient to see the logs.",
                        elem_id="processing_logs"
                    )

            # Set up event handlers
            load_btn.click(
                load_random_patient,
                inputs=[age_min, age_max, gender, ethnicity, condition],
                outputs=[name, patient_id, age, gender_display, ethnicity_display, symptoms, patient_photo, xray_image]
            )

            generate_photo_btn.click(
                generate_person_photo,
                inputs=[gender_display, ethnicity_display],
                outputs=[patient_photo]
            )

            generate_xray_btn.click(
                generate_xray,
                inputs=[condition],
                outputs=[xray_image]
            )

            process_btn.click(
                process_patient,
                inputs=[name, patient_id, age, gender_display, ethnicity_display, symptoms, patient_photo, xray_image],
                outputs=[assessment, photo_analysis, xray_analysis, logs_display]
            )

            # Add some CSS for better formatting
            gr.Markdown("""
            <style>
            #medical_assessment {
                padding: 20px;
                border-radius: 8px;
                background-color: #f8f9fa;
                margin-bottom: 20px;
            }
            #medical_assessment h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            #medical_assessment h2 {
                color: #34495e;
                margin-top: 20px;
                padding: 10px 0;
                border-left: 4px solid #3498db;
                padding-left: 15px;
            }
            #processing_logs {
                padding: 20px;
                border-radius: 8px;
                background-color: #f8f9fa;
            }
            #processing_logs h1 {
                color: #2c3e50;
                border-bottom: 2px solid #e74c3c;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            #processing_logs h2 {
                color: #34495e;
                margin-top: 20px;
            }
            #processing_logs pre {
                background-color: #2c3e50;
                color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }
            #photo_analysis, #xray_analysis {
                padding: 15px;
                border-radius: 8px;
                background-color: #f8f9fa;
                margin-bottom: 15px;
            }
            </style>
            """)

        return blocks

    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        self.interface.launch(**kwargs)

def main():
    """Main entry point for the hospital interface."""
    app = HospitalInterface()
    app.launch(share=True)

if __name__ == "__main__":
    main() 