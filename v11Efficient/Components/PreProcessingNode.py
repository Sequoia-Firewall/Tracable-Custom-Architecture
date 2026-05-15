import pandas as pd

class PreProcesingNode:
    def __init__(self, Logger, logger_classification):
        self.data = []
        self.Logger = Logger
        self.classification = logger_classification  
        self.removable_columns = ['student_id']  # Example of a column that might be removed during preprocessing

        self._cat_vocab = {}          
        self._all_columns = set() 

    def display(self, message, classification = None):
        message = f"[PreProcessingNode]: {message}"
        if classification is None:
            classification = self.classification
        if self.Logger is None:
            raise ValueError("Logger not assigned")
        self.Logger.log(message, classification, True)

    def process_data(self, input_data):
        data = input_data.copy()
        data = self.tokenize_categorical(data)
        self.normalize_numerical(data)
        self.data.append(data) # For processing datasets or learning
        return data

    def process_dataset(self, dataset):
        if isinstance(dataset, pd.DataFrame):
            pass
        elif isinstance(dataset, str):
            dataset = pd.read_csv(dataset)
        else:
            raise TypeError("dataset must be either a pandas DataFrame or a filepath string")
        data = self.drop_removable_columns(dataset)
        data = self.handle_missing_values(data)
        with self.Logger.make_progress(transient=True) as progress:
            task = progress.add_task("Pre-processing dataset", total=len(data))
            for _, row in data.iterrows():
                self.process_data(row)
                progress.update(task, advance=1)
        self.display("Pre-processing complete.", 4)
        return pd.DataFrame(self.data).fillna(0)

    def drop_removable_columns(self, data):
        for col in self.removable_columns:
            if col in data:
                del data[col]
        self.display(f"Dropped removable columns: {self.removable_columns}", 1)
        return data
    
    def vectorize_input(self, data):
        vectorized = []
        for col in sorted(self._all_columns):
            vectorized.append(data.get(col, 0))
        #DEBUGself.display("Vectorized input data.", 1)
        return vectorized

    def tokenize_categorical(self, data):
        if not hasattr(self, "_cat_vocab"):
            self._cat_vocab = {}

        categorical_columns = [
            key for key, value in data.items() if isinstance(value, str)
        ]

        encoded = {}

        for col in categorical_columns:
            value = data[col]
            vocab = self._cat_vocab.setdefault(col, {})

            # Learn categories dynamically
            if value not in vocab:
                vocab[value] = len(vocab)

            for k in vocab:
                encoded[f"{col}_{k}"] = 0
            encoded[f"{col}_{value}"] = 1

        # Pass through non-categorical
        for k, v in data.items():
            if k not in categorical_columns:
                encoded[k] = v
        self._all_columns.update(encoded.keys())
        for col in self._all_columns:
            encoded.setdefault(col, 0)
        #DEBUGself.display(f"Tokenized categorical columns: {categorical_columns}", 1)
        return encoded
         

    def normalize_numerical(self, data):

        for key, value in data.items():
            # If the value is a sequence/Series of numbers, apply min-max scaling.
            if isinstance(value, (list, tuple, pd.Series)):
                if len(value) == 0:
                    continue
                vmin = min(value)
                vmax = max(value)
                denom = (vmax - vmin)
                if denom == 0:
                    # All values equal; map to 0.0 to avoid division by zero.
                    data[key] = [0.0 for _ in value]
                else:
                    data[key] = [(v - vmin) / denom for v in value]
            elif isinstance(value, (int, float)):
                # Scalar values can't be min-max scaled without external bounds.
                data[key] = float(value)

        #DEBUGself.display("Normalized numerical data.", 1,)

    def handle_missing_values(self, data):
        keys_to_delete = [key for key, value in data.items() if value is None]
        for key in keys_to_delete:
            # remove key with missing value
            del data[key]
        self.display(f"Removed keys with missing values: {keys_to_delete}", 2)
        return data
    
    def get_random_sample(self, n =1):
        import random
        #for testing and demonstration purposes
        random.seed(42)
        if not self.data:
            raise ValueError("No data available. Please process data first.")
        if n > 1:
            return random.sample(self.data, n)
        else:
            return random.choice(self.data)
    

    
    
if __name__ == "__main__":
    # Example usage
    import Logger
    import time
    current_time = int(time.time())
    logger = Logger.Logger(f"preprocessing_{current_time}.log", log_level=4)
    preprocessor = PreProcesingNode(Logger = logger, logger_classification=4)
    
    preprocessor.removable_columns = ['student_id']

    file_input = open(input("Enter the path to the CSV file: "), 'r')

    processed_df = preprocessor.process_dataset(file_input)
    print(processed_df)
    logger.log("Data processing completed successfully.", 4, True)
    #save processed data
    processed_df.to_csv(f"processed_data_{current_time}.csv", index=False)

