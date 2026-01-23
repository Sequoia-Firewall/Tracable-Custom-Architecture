"""
- preprocessing node: 
    -- processes dataset
    -- tokenizes data
    --- note not LLM scale tokenization, likely just small numerical representation of categorical data unique values for now

"""
import pandas as pd
import numpy as np

class PreProcessingNode:
    def __init__(self, logging_enabled=False, logger=None):
        self.logging_enabled = logging_enabled
        self.df = None
        self.logger = logger
    
    def display(self, message, Loud):
        if self.logging_enabled:
            if self.logger is None:
                raise ValueError("Logger is not set for PreProcessingNode.")
            message = (f"[PreProcessingNode] {message}")
            self.logger.log(message, Loud= Loud)
    
    def process(self, file_path, demo, Loud):
        self.display(f"Starting preprocessing on dataset: {file_path}", Loud= Loud)
        self.load_csv(file_path, Loud= Loud)
        if self.df is None:
            self.display("No dataset loaded; aborting preprocessing.", Loud= Loud)
            return None
        
        if demo == True:
            self.df = self.df.sample(n=3000, random_state=42).reset_index(drop=True)
            self.display("Demo mode active: dataset sampled to 500 rows.", Loud= Loud)
        
        self.normalize_data(Loud= Loud)
        self.tokenize_categorical(Loud= Loud)
        self.remove_nan(Loud= Loud)
        self.display("Preprocessing complete.", Loud= Loud)
        self.display(f"Processed DataFrame info: {self.df.info()}", Loud= Loud)
        return self.df

    def load_csv(self, file_path, Loud):
        import pandas as pd
        self.display(f"Loading CSV file from: {file_path}", Loud= Loud)
        self.df = pd.read_csv(file_path)
        self.display(f"CSV loaded with shape: {self.df.shape}", Loud= Loud)
        return self.df
    
    def tokenize_categorical(self, Loud, target_col="exam_score", smoothing=10):
        if self.df is None:
            self.display("No dataset loaded to preprocess.", Loud= Loud)
            return None


        self.display("Starting target mean encoding with evidence triples.", Loud= Loud)

        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        self.display(f"Categorical columns identified: {categorical_cols.tolist()}", Loud= Loud)

        # Ensure target column is numeric
        if not pd.api.types.is_numeric_dtype(self.df[target_col]):
            self.display(f"Target column '{target_col}' is not numeric. Converting to numeric.", Loud= Loud)
            self.df[target_col] = pd.to_numeric(self.df[target_col], errors='coerce')
        
        global_mean = self.df[target_col].mean()
        global_var = self.df[target_col].var()
        
        # Handle case where variance is NaN or not numeric
        if pd.isna(global_var) or not isinstance(global_var, (int, float)):
            global_var = 1.0  # Default variance
            self.display("Global variance is NaN or not numeric, using default value of 1.0", Loud= Loud)
    
        self.display(f"Global target mean ({target_col}): {global_mean}", Loud= Loud)
        self.display(f"Global target variance: {global_var}", Loud= Loud)

        for col in categorical_cols:
            self.display(f"Encoding column as evidence: {col}", Loud= Loud)

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

        self.display("Categorical evidence encoding complete.", Loud= Loud)
        return self.df

    def remove_nan(self, Loud):
        if self.df is None:
            self.display("No dataset loaded to preprocess.", Loud= Loud)
            return None
        
        self.display("Removing NaN values from dataset.", Loud= Loud)
        initial_shape = self.df.shape
        self.df.dropna(inplace=True)
        self.display(f"NaN removal complete. Shape changed from {initial_shape} to {self.df.shape}.", Loud= Loud)
        return self.df
    
    def normalize_data(self, Loud):
        if self.df is None:
            self.display("No dataset loaded to preprocess.", Loud= Loud)
            return None
        
        self.display("Normalizing numerical columns.", Loud= Loud)
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.display(f"Numerical columns identified: {numerical_cols.tolist()}", Loud= Loud)
        
        for col in numerical_cols:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            self.display(f"Normalizing column: {col} with min: {min_val}, max: {max_val}", Loud= Loud)
            if max_val - min_val != 0:
                self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
            else:
                self.df[col] = 0.0  # If all values are the same
        self.display("Normalization complete.", Loud= Loud)
        return self.df
    
    def vectorize_training_data(self, training_data, Loud):
        vectors = []
        if training_data is None or training_data.empty:
            return vectors
        for _, row in training_data.iterrows():
            vector = []
            for value in row:
                if isinstance(value, tuple) and len(value) == 3:
                    mu, sigma2, n_eff = value
                    vector.extend([mu, sigma2, n_eff])
                else:
                    vector.append(value)
            vectors.append(vector)
        return vectors
        