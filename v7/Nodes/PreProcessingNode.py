"""
- preprocessing node: 
    -- processes dataset
    -- tokenizes data
    --- note not LLM scale tokenization, likely just small numerical representation of categorical data unique values for now

"""
import pandas as pd

class PreProcessingNode:
    def __init__(self, logging_enabled=False):
        self.logging_enabled = logging_enabled
        self.df = None
    
    def display(self, message):
        if self.logging_enabled:
            print(f"[PreProcessingNode] {message}")
    
    def process(self, file_path):
        self.display(f"Starting preprocessing on dataset: {file_path}")
        self.load_csv(file_path)
        if self.df is None:
            self.display("No dataset loaded; aborting preprocessing.")
            return None
        self.tokenize_categorical()
        self.remove_nan()
        self.normalize_data()
        self.display("Preprocessing complete.")
        self.display(f"Processed DataFrame info: {self.df.info()}")
        return self.df

    def load_csv(self, file_path):
        import pandas as pd
        self.display(f"Loading CSV file from: {file_path}")
        self.df = pd.read_csv(file_path)
        self.display(f"CSV loaded with shape: {self.df.shape}")
        return self.df
    
    def tokenize_categorical(self, target_col="exam_score", smoothing=10):
        if self.df is None:
            self.display("No dataset loaded to preprocess.")
            return None

        import numpy as np

        self.display("Starting target mean encoding with evidence triples.")

        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        self.display(f"Categorical columns identified: {categorical_cols.tolist()}")

        # Ensure target column is numeric
        if not pd.api.types.is_numeric_dtype(self.df[target_col]):
            self.display(f"Target column '{target_col}' is not numeric. Converting to numeric.")
            self.df[target_col] = pd.to_numeric(self.df[target_col], errors='coerce')
        
        global_mean = self.df[target_col].mean()
        global_var = self.df[target_col].var()
        
        # Handle case where variance is NaN or not numeric
        if pd.isna(global_var) or not isinstance(global_var, (int, float)):
            global_var = 1.0  # Default variance
            self.display("Global variance is NaN or not numeric, using default value of 1.0")

        self.display(f"Global target mean ({target_col}): {global_mean}")
        self.display(f"Global target variance: {global_var}")

        for col in categorical_cols:
            self.display(f"Encoding column as evidence: {col}")

            stats = (
                self.df
                .groupby(col)[target_col]
                .agg(['mean', 'var', 'count'])
                .fillna(global_var)
            )

            # Smoothed mean (μ)
            mu = (
                (stats['count'] * stats['mean'] + smoothing * global_mean) /
                (stats['count'] + smoothing)
            )

            # Smoothed variance (σ²)

            sigma2 = (
                (stats['count'] * stats['var'] + smoothing * global_var) /
                (stats['count'] + smoothing)
            )

            # Effective sample size (n)
            n_eff = stats['count']

            # Store as evidence triples
            evidence_map = {
                k: (mu[k], sigma2[k], n_eff[k])
                for k in stats.index
            }

            self.df[col] = self.df[col].map(evidence_map)

        self.display("Categorical evidence encoding complete.")
        return self.df

    
    def remove_nan(self):
        if self.df is None:
            self.display("No dataset loaded to preprocess.")
            return None
        
        self.display("Removing NaN values from dataset.")
        initial_shape = self.df.shape
        self.df.dropna(inplace=True)
        self.display(f"NaN removal complete. Shape changed from {initial_shape} to {self.df.shape}.")
        return self.df
    
    def normalize_data(self):
        if self.df is None:
            self.display("No dataset loaded to preprocess.")
            return None
        
        self.display("Normalizing numerical columns.")
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.display(f"Numerical columns identified: {numerical_cols.tolist()}")
        
        for col in numerical_cols:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            self.display(f"Normalizing column: {col} with min: {min_val}, max: {max_val}")
            if max_val - min_val != 0:
                self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
            else:
                self.df[col] = 0.0  # If all values are the same
        self.display("Normalization complete.")
        return self.df
    

        