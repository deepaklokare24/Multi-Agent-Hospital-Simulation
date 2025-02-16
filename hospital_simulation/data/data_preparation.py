import pandas as pd
import kagglehub
from typing import List, Dict, Tuple
from langchain_groq import ChatGroq
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
import json
from datasets import load_dataset
from langchain.schema import HumanMessage
import time
import backoff  # For exponential backoff

class DataPreparation:
    def __init__(self, model_name: str = 'llama-3.3-70b-versatile'):
        print(f"\nInitializing DataPreparation with model: {model_name}")
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.patient_df = None
        self.all_patients_name_id = None
        self.x_ray_dataset = None
        self.max_retries = 5
        self.batch_size = 50  # Generate in smaller batches

    def prepare_medical_dataset(self) -> pd.DataFrame:
        """Load and prepare the disease symptoms dataset."""
        try:
            print("\n=== Loading Medical Dataset ===")
            print("Downloading dataset from Kaggle...")
            path = kagglehub.dataset_download("uom190346a/disease-symptoms-and-patient-profile-dataset")
            file_name = '/Disease_symptom_and_patient_profile_dataset.csv'
            self.patient_df = pd.read_csv(path + file_name)
            
            # Print dataset statistics
            print(f"\nDataset Statistics:")
            print(f"Total records: {len(self.patient_df)}")
            print(f"Gender distribution:")
            print(self.patient_df['Gender'].value_counts().to_string())
            print(f"\nColumns: {', '.join(self.patient_df.columns)}")
            
            return self.patient_df
            
        except Exception as e:
            print(f"\nError loading dataset: {e}")
            print("Creating sample dataset for testing...")
            self.patient_df = pd.DataFrame({
                'Gender': ['Male', 'Female'] * 5,
                'Age': [25, 30, 45, 35, 50, 28, 42, 33, 55, 40],
                'Difficulty Breathing': ['Yes', 'No'] * 5,
                'Outcome Variable': ['Positive', 'Negative'] * 5
            })
            return self.patient_df

    def load_xray_dataset(self):
        """Load chest X-ray dataset."""
        try:
            print("Loading X-ray dataset...")
            self.x_ray_dataset = load_dataset("keremberke/chest-xray-classification", name="full")
            print("X-ray dataset loaded successfully")
            return self.x_ray_dataset
        except Exception as e:
            print(f"Error loading X-ray dataset: {e}")
            return None

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=5,
        giveup=lambda e: "Invalid API key" in str(e)
    )
    def _generate_batch(self, num_records: int, ratio: float) -> List[Dict]:
        """Generate a batch of patient records with retry logic."""
        prompt = f"""Generate {num_records} Indian patient records in JSON array format.
The ratio of Female to Male should be {ratio}:1.
Each record must have these exact fields:
1. First_Name: An Indian first name matching the gender
2. Last_Name: An Indian last name
3. Patient_ID: A unique 13-character alphanumeric ID starting with 'IN'
4. G_Gender: The gender ('Male' or 'Female') matching the first name

Example format:
[
  {{
    "First_Name": "Priya",
    "Last_Name": "Patel",
    "Patient_ID": "IN123456789AB",
    "G_Gender": "Female"
  }}
]

Return ONLY the JSON array with {num_records} records."""

        print(f"\nGenerating batch of {num_records} records...")
        messages = [HumanMessage(content=prompt)]
        
        try:
            output = self.llm.invoke(messages)
            text = str(output.content)
            
            # Extract JSON from the response
            start = text.find('[')
            end = text.rfind(']') + 1
            
            if start == -1 or end == 0:
                print("No valid JSON found in response. Response preview:")
                print(text[:200])
                return []
                
            json_str = text[start:end]
            return json.loads(json_str)
            
        except Exception as e:
            print(f"Error in batch generation: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
            return []

    def generate_patient_identities(self) -> pd.DataFrame:
        """Generate synthetic patient identities with batching and retries."""
        if self.patient_df is None:
            raise ValueError("Please load medical dataset first using prepare_medical_dataset()")

        print("\n=== Generating Patient Identities ===")
        
        # Calculate gender ratio
        female_count = self.patient_df[self.patient_df['Gender'] == 'Female'].shape[0]
        male_count = self.patient_df[self.patient_df['Gender'] == 'Male'].shape[0]
        ratio = female_count / male_count
        
        print(f"Target gender distribution:")
        print(f"- Female: {female_count}")
        print(f"- Male: {male_count}")
        print(f"- Ratio (F/M): {ratio:.2f}")

        start_time = time.time()
        all_records = []
        total_records = len(self.patient_df)
        records_generated = 0

        while records_generated < total_records:
            remaining = total_records - records_generated
            current_batch_size = min(self.batch_size, remaining)
            
            print(f"\nProgress: {records_generated}/{total_records} records")
            batch = self._generate_batch(current_batch_size, ratio)
            
            if batch:
                all_records.extend(batch)
                records_generated = len(all_records)
                print(f"Successfully generated {len(batch)} records")
            else:
                print("Batch generation failed, retrying...")
                time.sleep(2)  # Add delay before retry
            
            # Print progress
            elapsed_time = time.time() - start_time
            if records_generated > 0:
                avg_time_per_record = elapsed_time / records_generated
                estimated_remaining = avg_time_per_record * (total_records - records_generated)
                print(f"Elapsed time: {elapsed_time:.1f}s, Estimated remaining time: {estimated_remaining:.1f}s")

        # Convert to DataFrame and validate
        print("\nProcessing generated records...")
        self.all_patients_name_id = pd.DataFrame(all_records)
        self.all_patients_name_id.rename(columns={"G_Gender": "Gender"}, inplace=True)
        
        # Print generation statistics
        print(f"\nGeneration completed in {time.time() - start_time:.1f} seconds")
        print(f"Generated {len(self.all_patients_name_id)} patient identities")
        print("\nGenerated gender distribution:")
        print(self.all_patients_name_id['Gender'].value_counts().to_string())
        
        # Verify data quality
        print("\nVerifying data quality...")
        missing_fields = self.all_patients_name_id.isnull().sum()
        if missing_fields.any():
            print("Warning: Found missing values:")
            print(missing_fields[missing_fields > 0].to_string())
        
        # Print some example records
        print("\nExample generated records:")
        print(self.all_patients_name_id.head(3).to_string())
        
        return self.all_patients_name_id

    def merge_patient_data(self) -> pd.DataFrame:
        """Merge medical data with generated patient identities."""
        if self.patient_df is None or self.all_patients_name_id is None:
            raise ValueError("Please load medical dataset and generate patient identities first")

        print("\n=== Merging Patient Data ===")
        
        # Split by gender
        print("Splitting data by gender...")
        gender_counts = self.patient_df['Gender'].value_counts()
        unique_males = self.all_patients_name_id[self.all_patients_name_id['Gender'] == 'Male'].drop_duplicates().head(gender_counts['Male'])
        unique_females = self.all_patients_name_id[self.all_patients_name_id['Gender'] == 'Female'].drop_duplicates().head(gender_counts['Female'])

        patient_male = self.patient_df[self.patient_df['Gender'] == 'Male'].reset_index(drop=True)
        patient_female = self.patient_df[self.patient_df['Gender'] == 'Female'].reset_index(drop=True)

        print("\nMerging records...")
        # Merge data
        updated_male_patients = pd.concat([
            patient_male.reset_index(drop=True),
            unique_males[0:patient_male.shape[0]].reset_index(drop=True)
        ], axis=1)

        updated_female_patients = pd.concat([
            patient_female.reset_index(drop=True),
            unique_females[0:patient_female.shape[0]].reset_index(drop=True)
        ], axis=1)

        # Combine and clean
        print("Combining and cleaning data...")
        updated_patient_df = pd.concat([updated_male_patients, updated_female_patients], axis=0)
        updated_patient_df = updated_patient_df.loc[:, ~updated_patient_df.columns.duplicated()]
        
        print(f"\nMerge completed:")
        print(f"- Total records: {len(updated_patient_df)}")
        print(f"- Columns: {', '.join(updated_patient_df.columns)}")
        print("\nExample merged record:")
        print(updated_patient_df.iloc[0].to_string())
        
        return updated_patient_df

    def get_random_patient(self, conditions: dict = None) -> pd.Series:
        """Get a random patient matching the specified conditions."""
        if self.patient_df is None:
            self.prepare_medical_dataset()
            self.generate_patient_identities()
            self.patient_df = self.merge_patient_data()

        if conditions:
            mask = pd.Series(True, index=self.patient_df.index)
            for column, value in conditions.items():
                if isinstance(value, tuple) and len(value) == 2:
                    mask &= self.patient_df[column].between(value[0], value[1])
                else:
                    mask &= self.patient_df[column] == value
            filtered_df = self.patient_df[mask]
        else:
            filtered_df = self.patient_df

        if filtered_df.empty:
            raise ValueError("No patients match the specified conditions")

        return filtered_df.sample(n=1).iloc[0] 