from hospital_simulation.data.data_preparation import DataPreparation
import pandas as pd
from pathlib import Path

class DataPreprocessor(DataPreparation):
    """Data preprocessor that handles data preparation and saving."""
    
    def __init__(self, model_name: str = 'llama-3.3-70b-versatile'):
        super().__init__(model_name)
        self.output_dir = Path(__file__).parent / "processed"
        self.output_dir.mkdir(exist_ok=True)
        self.preprocessed_file = self.output_dir / "preprocessed_patients.csv"

    def preprocess_and_save(self) -> pd.DataFrame:
        """Run the complete preprocessing pipeline and save results."""
        try:
            # Load and process medical dataset
            patient_df = self.prepare_medical_dataset()
            print(f"Loaded medical dataset with {len(patient_df)} records")
            
            # Generate patient identities
            identities_df = self.generate_patient_identities()
            print(f"Generated {len(identities_df)} patient identities")
            
            # Merge the data
            final_df = self.merge_patient_data()
            print(f"Final dataset contains {len(final_df)} records")
            
            # Save processed data
            final_df.to_csv(self.preprocessed_file, index=False)
            print(f"\nSaved processed data to {self.preprocessed_file}")
            
            return final_df
            
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            raise

    def load_preprocessed_data(self) -> pd.DataFrame:
        """Load preprocessed data if available, otherwise run preprocessing."""
        if not self.preprocessed_file.exists():
            print("Preprocessed data not found. Running preprocessing...")
            return self.preprocess_and_save()
        
        print(f"Loading preprocessed data from {self.preprocessed_file}")
        return pd.read_csv(self.preprocessed_file)

    def get_random_patient(self, conditions: dict = None) -> pd.Series:
        """Get a random patient matching the specified conditions."""
        df = self.load_preprocessed_data()
        
        if conditions:
            mask = pd.Series(True, index=df.index)
            for column, value in conditions.items():
                if isinstance(value, tuple) and len(value) == 2:
                    mask &= df[column].between(value[0], value[1])
                else:
                    mask &= df[column] == value
            filtered_df = df[mask]
        else:
            filtered_df = df

        if filtered_df.empty:
            raise ValueError("No patients match the specified conditions")

        return filtered_df.sample(n=1).iloc[0]

def main():
    """Run the preprocessing script."""
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_and_save()

if __name__ == "__main__":
    main() 