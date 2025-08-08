import random
import math
from collections import deque

class Judge:
    def __init__(self, node_id, node_position, max_random=0.0, min_random=0.0, constant=1.0, demo=False):
        self.node_id = node_id
        self.node_position = node_position
        self.demo = demo
        self.define_node_weights(max_random, min_random, constant)
        self.init_scoring_weights()
        self.node_type = "Judge"
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0.0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }

    def init_scoring_weights(self):
        self.hidden_dim = 16 if self.demo else 512
        # One scalar output per token → score = W·x + b
        self.W_keep = [random.uniform(-0.1, 0.1) for _ in range(self.hidden_dim)]
        self.b_keep = 0.0

    def integrate_positional_encoding(self, token_embeddings, positional_encodings):
        return [
            [
                token_embeddings[t][i] + positional_encodings[t][i]
                for i in range(len(token_embeddings[0]))
            ]
            for t in range(len(token_embeddings))
        ]

    def apply_attention_mask(self, token_embeddings, attention_mask):
        return [
            [
                token_embeddings[t][i] * attention_mask[t]
                for i in range(len(token_embeddings[0]))
            ]
            for t in range(len(token_embeddings))
        ]

    def compute_keep_scores(self, token_embeddings, Controller_output):
        # Use Controller output as a bias scale factor for all tokens
        weight_factor = sum(Controller_output) / len(Controller_output)  # scalar context

        keep_scores = []
        for token in token_embeddings:
            score = sum(token[i] * self.W_keep[i] for i in range(self.hidden_dim)) + self.b_keep
            score *= weight_factor  # dynamic scaling from Controller
            # Optional noise
            score += random.uniform(self.weights['Min_random'], self.weights['Max_random'])
            score *= self.weights['constant']
            # Sigmoid for [0, 1] range
            keep_scores.append(1 / (1 + math.exp(-score)))
        return keep_scores

    def gate_tokens(self, token_embeddings, keep_scores):
        return [
            [val * keep_scores[t] for val in token_embeddings[t]]
            for t in range(len(token_embeddings))
        ]

    def process(self, input_data):
        # Handle different input types
        token_embeddings = []
        
        if isinstance(input_data, str):
            # Convert string to basic token embeddings (simplified)
            token_embeddings = [[ord(c)/256 for _ in range(self.hidden_dim)] for c in input_data]
            
        elif isinstance(input_data, list):
            if all(isinstance(item, list) for item in input_data):
                # If it's already a list of token embeddings
                token_embeddings = input_data
            elif len(input_data) > 0:
                # Convert 1D list to token embeddings
                if len(input_data) >= self.hidden_dim:
                    chunks = [input_data[i:i+self.hidden_dim] for i in range(0, len(input_data), self.hidden_dim)]
                    token_embeddings = chunks
                else:
                    # Pad if needed
                    padded = input_data + [0] * (self.hidden_dim - len(input_data))
                    token_embeddings = [padded]
        else:
            # Fallback: create default embeddings
            token_embeddings = [[0.1] * self.hidden_dim]
        
        # Create simple positional encodings based on position in sequence
        positional_encodings = [
            [0.1 * (i / self.hidden_dim + t / len(token_embeddings)) for i in range(self.hidden_dim)]
            for t in range(len(token_embeddings))
        ]
        
        # Default attention mask (all 1s)
        attention_mask = [1.0] * len(token_embeddings)
        
        # Default controller output
        controller_output = [1.0] * 4  # Assuming some default size
        
        # Use the evaluate method for processing
        gated_tokens, keep_scores = self.evaluate(token_embeddings, positional_encodings, attention_mask, controller_output)
        
        return gated_tokens

    def evaluate(self, token_embeddings, positional_encodings, attention_mask, Controller_output):
        integrated = self.integrate_positional_encoding(token_embeddings, positional_encodings)
        masked = self.apply_attention_mask(integrated, attention_mask)
        keep_scores = self.compute_keep_scores(masked, Controller_output)
        gated = self.gate_tokens(masked, keep_scores)
        return gated, keep_scores


class Controller:
    def __init__(self, node_id, node_position, num_branches, max_random=0.0, min_random=0.0, constant=1.0, demo=False):
        self.node_id = node_id
        self.node_position = node_position
        self.num_branches = num_branches
        self.demo = demo
        self.define_node_weights(max_random, min_random, constant)
        self.init_projection()
        self.node_type = "Controller"
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0.0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }

    def init_projection(self):
        # Initialize a manual projection matrix for routing: [hidden_dim x num_branches]
        self.hidden_dim = 16 if self.demo else 512  # You can make this dynamic if needed
        self.W = [[random.uniform(-1, 1) for _ in range(self.num_branches)] for _ in range(self.hidden_dim)]
        self.b = [0.0] * self.num_branches

    def mean_pooling(self, token_embeddings):
        T = len(token_embeddings)
        H = len(token_embeddings[0])
        return [sum(vec[i] for vec in token_embeddings)/T for i in range(H)]

    def linear_project(self, x):  # x: List[float] of size [hidden_dim]
        return [
            sum(x[i] * self.W[i][j] for i in range(self.hidden_dim)) + self.b[j]
            for j in range(self.num_branches)
        ]

    def softmax(self, z):
        max_z = max(z)
        exp_z = [math.exp(val - max_z) for val in z]
        sum_exp = sum(exp_z)
        return [val / sum_exp for val in exp_z]

    def run_equation(self, token_embeddings):
        summary = self.mean_pooling(token_embeddings)
        projection = self.linear_project(summary)

        # Optional noise injection
        noisy_proj = [
            val + random.uniform(self.weights['Min_random'], self.weights['Max_random'])
            for val in projection
        ]

        scaled_proj = [val * self.weights['constant'] for val in noisy_proj]
        return self.softmax(scaled_proj)  # Returns [P1, P2, ..., PN] for N branches
    
    def process(self, input_data):
        """
        Process input data through the Controller.
        Converts various input types to token embeddings and runs the controller equation.
        """
        if isinstance(input_data, str):
            # Convert string to simple token embeddings
            # For demo purposes, create embeddings based on string hash
            hash_val = hash(input_data) % 1000
            token_embeddings = []
            for i, char in enumerate(input_data[:10]):  # Process up to 10 characters
                embedding = [0.0] * self.hidden_dim
                char_val = ord(char) / 128.0  # Normalize ASCII
                for j in range(min(self.hidden_dim, 8)):
                    embedding[j] = char_val * (1 + j * 0.1) + (hash_val % (j + 1)) * 0.01
                token_embeddings.append(embedding)
            
            if not token_embeddings:  # Empty string case
                token_embeddings = [[0.0] * self.hidden_dim]
                
        elif isinstance(input_data, list) and len(input_data) > 0:
            # Assume it's already token embeddings or convert
            if isinstance(input_data[0], list):
                token_embeddings = input_data
            else:
                # Convert 1D list to token embeddings
                token_embeddings = [input_data[:self.hidden_dim]]
                
        else:
            # Fallback: create default embeddings
            token_embeddings = [[0.1] * self.hidden_dim]
        
        return self.run_equation(token_embeddings)


class Splitter:
    def __init__(self, node_id, node_position, num_branches, max_random=0.0, min_random=0.0, constant=1.0, demo=False):
        self.node_id = node_id
        self.node_position = node_position
        self.num_branches = num_branches
        self.demo = demo
        self.define_node_weights(max_random, min_random, constant)
        self.init_projection()
        self.reset_load_tracker()
        self.node_type = "Splitter"
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=1.0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }

    def init_projection(self):
        self.hidden_dim = 16 if self.demo else 512
        self.W = [[random.uniform(-0.1, 0.1) for _ in range(self.num_branches)] for _ in range(self.hidden_dim)]
        self.b = [0.0 for _ in range(self.num_branches)]

    def reset_load_tracker(self):
        self.load_tracker = [0 for _ in range(self.num_branches)]  # token count per branch

    def softmax(self, z):
        max_z = max(z)
        exp_z = [math.exp(val - max_z) for val in z]
        total = sum(exp_z)
        return [x / total for x in exp_z]

    def route_token(self, token_embedding):
        # Safety check for empty token
        if not token_embedding:
            token_embedding = [0.1] * self.hidden_dim
            
        # Ensure token_embedding is of correct length
        if len(token_embedding) < self.hidden_dim:
            # Pad the token embedding if it's too short
            token_embedding = token_embedding + [0.0] * (self.hidden_dim - len(token_embedding))
        elif len(token_embedding) > self.hidden_dim:
            # Truncate if too long
            token_embedding = token_embedding[:self.hidden_dim]
            
        # Calculate base scores with safety checks
        base_scores = []
        for j in range(self.num_branches):
            score = 0.0
            for i in range(min(len(token_embedding), self.hidden_dim, len(self.W))):
                if i < len(self.W) and j < len(self.W[i]):
                    score += token_embedding[i] * self.W[i][j]
            if j < len(self.b):
                score += self.b[j]
            base_scores.append(score)
        
        # Apply noise and constant scaling
        base_scores = [
            val * self.weights['constant'] + random.uniform(self.weights['Min_random'], self.weights['Max_random'])
            for val in base_scores
        ]

        # Penalize overloaded branches with safety check
        total_load = sum(self.load_tracker) + 1e-6  # prevent div by zero
        load_penalty = [self.load_tracker[j] / total_load for j in range(min(self.num_branches, len(self.load_tracker)))]
        
        # Ensure penalty list matches base_scores length
        while len(load_penalty) < len(base_scores):
            load_penalty.append(0.0)
            
        load_adjusted_scores = [base_scores[j] - load_penalty[j] for j in range(len(base_scores))]

        return self.softmax(load_adjusted_scores)

    def split(self, token_embeddings):
        self.reset_load_tracker()
        branch_tokens = [[] for _ in range(self.num_branches)]
        
        # Safety check for empty input
        if not token_embeddings:
            # Return empty lists for all branches
            return branch_tokens
            
        for token in token_embeddings:
            # Ensure token is not empty
            if not token:
                continue
                
            try:
                probs = self.route_token(token)
                # Safe branch selection with fallback
                if probs and len(probs) > 0:
                    max_prob = max(probs)
                    # Find the first occurrence of max probability
                    selected_branch = None
                    for i, prob in enumerate(probs):
                        if prob == max_prob:
                            selected_branch = i
                            break
                    
                    if selected_branch is None or selected_branch >= self.num_branches:
                        selected_branch = 0  # Fallback to first branch
                else:
                    selected_branch = 0  # Default fallback
                    
                # Ensure selected_branch is valid
                if 0 <= selected_branch < self.num_branches:
                    branch_tokens[selected_branch].append(token)
                    if selected_branch < len(self.load_tracker):
                        self.load_tracker[selected_branch] += 1
                else:
                    # Ultimate fallback - use first branch
                    branch_tokens[0].append(token)
                    self.load_tracker[0] += 1
                    
            except Exception as e:
                # Handle any errors in routing
                if self.demo:
                    print(f"Routing error in Splitter {self.node_id}: {str(e)}")
                # Fallback to random routing
                selected_branch = random.randint(0, self.num_branches - 1)
                branch_tokens[selected_branch].append(token)
                if selected_branch < len(self.load_tracker):
                    self.load_tracker[selected_branch] += 1

        return branch_tokens
        
    def process(self, input_data):
        # Handle different input types
        token_embeddings = []
        
        if isinstance(input_data, str):
            # Convert string to basic token embeddings (simplified)
            token_embeddings = [[ord(c)/256 for _ in range(self.hidden_dim)] for c in input_data]
            
        elif isinstance(input_data, list):
            if all(isinstance(item, list) for item in input_data):
                # If it's already a list of token embeddings
                token_embeddings = input_data
            elif len(input_data) > 0:
                # Convert 1D list to token embeddings
                if len(input_data) >= self.hidden_dim:
                    chunks = [input_data[i:i+self.hidden_dim] for i in range(0, len(input_data), self.hidden_dim)]
                    token_embeddings = chunks
                else:
                    # Pad if needed
                    padded = input_data + [0] * (self.hidden_dim - len(input_data))
                    token_embeddings = [padded]
        else:
            # Fallback: create default embeddings
            token_embeddings = [[0.1] * self.hidden_dim]
        
        # Always ensure we have at least one token embedding
        if not token_embeddings:
            token_embeddings = [[0.1] * self.hidden_dim]
            
        return self.split(token_embeddings)
    
class Computational:
    def __init__(self, node_id, node_position, output_dim=4, max_random=0.0, min_random=0.0, constant=1.0, memory_size=32, demo=False):
        self.node_id = node_id
        self.node_position = node_position
        self.output_dim = output_dim
        self.demo = demo
        self.memory_size = memory_size
        self.define_node_weights(max_random, min_random, constant)
        self.init_transform()
        self.init_memory()
        self.node_type = "Computational"
    
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=1.0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }

    def init_transform(self):
        self.hidden_dim = 16 if self.demo else 512
        self.W = [[random.uniform(-0.1, 0.1) for _ in range(self.output_dim)] for _ in range(self.hidden_dim)]
        self.b = [0.0 for _ in range(self.output_dim)]

    def init_memory(self):
        self.output_memory = deque(maxlen=self.memory_size)
        self.feedback_log = []

    def softmax(self, z):
        max_z = max(z)
        exp_z = [math.exp(val - max_z) for val in z]
        total = sum(exp_z)
        return [x / total for x in exp_z]

    def process(self, input_data):
        try:
            # Handle different input types
            token_embeddings = []
            
            if isinstance(input_data, str):
                # Convert string to basic token embeddings (simplified)
                token_embeddings = [[ord(c)/256 for _ in range(self.hidden_dim)] for c in input_data]
                
            elif isinstance(input_data, list):
                if len(input_data) == 0:
                    # Empty list case
                    token_embeddings = [[0.1] * self.hidden_dim]
                elif all(isinstance(item, list) for item in input_data):
                    # If it's already a list of token embeddings
                    # Ensure each token embedding has the correct dimension
                    token_embeddings = []
                    for item in input_data:
                        if len(item) < self.hidden_dim:
                            # Pad if too short
                            padded = item + [0.0] * (self.hidden_dim - len(item))
                            token_embeddings.append(padded)
                        elif len(item) > self.hidden_dim:
                            # Truncate if too long
                            token_embeddings.append(item[:self.hidden_dim])
                        else:
                            token_embeddings.append(item)
                elif isinstance(input_data, list) and len(input_data) > 0:
                    # Check if it's a list of numbers
                    if all(isinstance(x, (int, float)) for x in input_data):
                        # Convert 1D list to token embeddings
                        if len(input_data) >= self.hidden_dim:
                            # Split into chunks of hidden_dim size
                            chunks = []
                            for i in range(0, len(input_data), self.hidden_dim):
                                chunk = input_data[i:i+self.hidden_dim]
                                if len(chunk) < self.hidden_dim:
                                    # Pad the last chunk if needed
                                    chunk = chunk + [0.0] * (self.hidden_dim - len(chunk))
                                chunks.append(chunk)
                            token_embeddings = chunks
                        else:
                            # Pad if needed
                            padded = input_data + [0.0] * (self.hidden_dim - len(input_data))
                            token_embeddings = [padded]
                    else:
                        # Mixed or non-numeric list, create default embeddings
                        token_embeddings = [[0.1] * self.hidden_dim]
            else:
                # Fallback: create default embeddings for non-list, non-string inputs
                token_embeddings = [[0.1] * self.hidden_dim]
            
            # Final safety check: ensure we have at least one valid token embedding
            if not token_embeddings:
                token_embeddings = [[0.1] * self.hidden_dim]
            
            # Ensure all token embeddings are proper lists of numbers
            valid_embeddings = []
            for embedding in token_embeddings:
                if isinstance(embedding, list) and len(embedding) == self.hidden_dim:
                    if all(isinstance(x, (int, float)) for x in embedding):
                        valid_embeddings.append(embedding)
                    else:
                        # Convert mixed types to floats
                        valid_embeddings.append([float(x) if isinstance(x, (int, float)) else 0.0 for x in embedding])
                else:
                    # Create a default embedding if invalid
                    valid_embeddings.append([0.1] * self.hidden_dim)
            
            if not valid_embeddings:
                valid_embeddings = [[0.1] * self.hidden_dim]
                
            # Process the token embeddings
            pooled = [sum(t[i] for t in valid_embeddings)/len(valid_embeddings) for i in range(self.hidden_dim)]
            raw = [sum(pooled[i] * self.W[i][j] for i in range(self.hidden_dim)) + self.b[j]
                for j in range(self.output_dim)]
            raw = [val * self.weights['constant'] + random.uniform(self.weights['Min_random'], self.weights['Max_random'])
                for val in raw]
            probs = self.softmax(raw)
            self.output_memory.append(probs)  # Track historical output
            return probs
            
        except Exception as e:
            print(f"Error in Computational {self.node_id} process: {str(e)}")
            # Return a safe fallback - uniform distribution
            return [1.0/self.output_dim] * self.output_dim

    def forward(self, token_embeddings, next_nodes=[]):
        output_probs = self.process(token_embeddings)

        for node in next_nodes:
            if hasattr(node, "aggregate"):  # Review node
                result = node.aggregate([output_probs])
            elif hasattr(node, "process"):  # Another Comp node
                result = node.process([output_probs])
            else:
                result = None
            print(f"[{self.node_id}] → {node.node_id}: Result {result}")
        return output_probs

    def receive_feedback(self, reward):
        """External backpressure from reviewer or downstream node"""
        # Adjust constant weight (learning rate 0.05 for demo)
        self.weights['constant'] += 0.05 * reward
        self.weights['constant'] = max(0.1, min(self.weights['constant'], 5.0))  # Clamp
        self.feedback_log.append(reward)
        print(f"[{self.node_id}] Feedback received: {reward:.2f} → Constant weight now {self.weights['constant']:.3f}")

    def replay_memory(self):
        """Optional for training, introspection, or review"""
        return list(self.output_memory)
    def run(self, token_embeddings):
        """
        Main entry point for processing input embeddings.
        This method can be called directly by the brainNexus or other nodes.
        """
        return self.process(token_embeddings)

class Repeater:
    def __init__(self, node_id, node_position, max_random=0.0, min_random=0.0, constant=0.0, demo=False):
        self.node_id = node_id
        self.node_position = node_position
        self.demo = demo
        self.node_type = "Repeater"
        self.define_node_weights(max_random=.001, min_random=.00001, constant=0)
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0.0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
        
    def run_equation(self, token_embeddings):
        # Example: Repeat the input embeddings
        noise_factor = random.uniform(self.weights['Min_random'], self.weights['Max_random'])
        if isinstance(token_embeddings, list):
            if all(isinstance(item, list) for item in token_embeddings):
                # Handle 2D list (list of token embeddings)
                return [
                    [val * (1 + noise_factor) for val in token]
                    for token in token_embeddings
                ]
            else:
                # Handle 1D list (single token embedding)
                return [val * (1 + noise_factor) for val in token_embeddings]
        else:
            # Fallback for non-list inputs
            return token_embeddings
            
    def process(self, input_data):
        """Process input data through the Repeater."""
        try:
            # Initialize output with a default
            token_embeddings = []
            
            # Handle different input types
            if isinstance(input_data, str):
                # For a string, create simple token embeddings
                hidden_dim = 16 if self.demo else 512
                token_embeddings = [[ord(c)/256 for _ in range(hidden_dim)] for c in input_data]
                
            elif isinstance(input_data, list):
                if len(input_data) == 0:
                    # Empty list case
                    hidden_dim = 16 if self.demo else 512
                    token_embeddings = [[0.1] * hidden_dim]
                elif all(isinstance(item, list) for item in input_data):
                    # List of embeddings - use as is
                    token_embeddings = input_data
                else:
                    # Single embedding - wrap in a list
                    token_embeddings = [input_data]
            else:
                # For non-string, non-list inputs, create a default embedding
                hidden_dim = 16 if self.demo else 512
                token_embeddings = [[0.1] * hidden_dim]
                
            # Run the repeater equation on the token embeddings
            return self.run_equation(token_embeddings)
            
        except Exception as e:
            print(f"Error in Repeater {self.node_id} process: {str(e)}")
            # Return input unchanged as fallback
            return input_data
class Retainer:
    def __init__(self, node_id, node_position, expected_nodes, demo=False):
        self.node_id = node_id
        self.node_position = node_position
        # Ensure expected_nodes is always a list
        if isinstance(expected_nodes, (int, float)):
            self.expected_nodes = [int(expected_nodes)]
        elif isinstance(expected_nodes, list):
            self.expected_nodes = expected_nodes
        else:
            self.expected_nodes = []
        self.demo = demo
        self.collected_outputs = {}
        self.review_node = None
        self.node_type = "Retainer"

    def connect_review(self, review_node):
        self.review_node = review_node
        if self.demo:
            print(f"[Retainer {self.node_id}] Connected to review node {review_node.node_id}")
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=0.0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
    def receive(self, sender_id, output_probs):
        """Called by each computational node once it has finished processing."""
        # Ensure the output_probs is in the correct format
        if output_probs is None:
            output_probs = []
        
        # Handle non-list outputs by converting them
        if not isinstance(output_probs, list):
            output_probs = [output_probs]
            
        self.collected_outputs[sender_id] = output_probs
        print(f"[Retainer {self.node_id}] Received from {sender_id}: {output_probs}")

        if self.ready_to_forward():
            self.forward_to_review()

    def ready_to_forward(self):
        # Ensure expected_nodes is always iterable
        if not hasattr(self, 'expected_nodes') or self.expected_nodes is None:
            self.expected_nodes = []
        elif isinstance(self.expected_nodes, (int, float)):
            self.expected_nodes = [int(self.expected_nodes)]
        elif not isinstance(self.expected_nodes, (list, tuple)):
            self.expected_nodes = []
            
        return all(node_id in self.collected_outputs for node_id in self.expected_nodes)

    def forward_to_review(self):
        # Ensure expected_nodes is always iterable
        if not hasattr(self, 'expected_nodes') or self.expected_nodes is None:
            self.expected_nodes = []
        elif isinstance(self.expected_nodes, (int, float)):
            self.expected_nodes = [int(self.expected_nodes)]
        elif not isinstance(self.expected_nodes, (list, tuple)):
            self.expected_nodes = []
            
        if not self.review_node:
            print(f"[Retainer {self.node_id}] No review node connected.")
            # Still return the collected outputs for potential downstream processing
            ordered_outputs = []
            for node_id in self.expected_nodes:
                if node_id in self.collected_outputs:
                    output = self.collected_outputs[node_id]
                    # Ensure each output is a list of probabilities
                    if not isinstance(output, list):
                        output = [float(output)]  # Convert single value to list
                    elif not all(isinstance(x, (int, float)) for x in output):
                        # If it's a list but not of numbers, create a default output
                        output = [0.25, 0.25, 0.25, 0.25]  # Default uniform distribution
                    ordered_outputs.append(output)
            # Clear for next round even when no review node is connected
            self.collected_outputs.clear()
            return ordered_outputs

        # Ensure we're passing the output in the expected format for the reviewer
        ordered_outputs = []
        for node_id in self.expected_nodes:
            if node_id in self.collected_outputs:
                output = self.collected_outputs[node_id]
                # Ensure each output is a list of probabilities
                if not isinstance(output, list):
                    output = [float(output)]  # Convert single value to list
                elif not all(isinstance(x, (int, float)) for x in output):
                    # If it's a list but not of numbers, create a default output
                    output = [0.25, 0.25, 0.25, 0.25]  # Default uniform distribution
                ordered_outputs.append(output)
                
        result = self.review_node.aggregate(ordered_outputs)

        print(f"[Retainer {self.node_id}] → Review {self.review_node.node_id}: Final result {result}")

        # Clear for next round
        self.collected_outputs.clear()
        return result
        
    def process(self, input_data):
        # For Retainer, process should receive data and forward it to the review node
        try:
            # Ensure expected_nodes is iterable
            if not hasattr(self, 'expected_nodes') or self.expected_nodes is None:
                self.expected_nodes = []
            elif isinstance(self.expected_nodes, (int, float)):
                self.expected_nodes = [int(self.expected_nodes)]
            elif not isinstance(self.expected_nodes, (list, tuple)):
                self.expected_nodes = []
            
            # Handle different input types
            if isinstance(input_data, dict):
                # If it's already a dictionary of node_id -> output, merge with existing
                if any(node_id in input_data for node_id in self.expected_nodes):
                    for k, v in input_data.items():
                        if k in self.expected_nodes:
                            self.collected_outputs[k] = v if isinstance(v, list) else [float(v)]
                # Extract from a structured dict with 'sender_id' and 'output'
                elif 'sender_id' in input_data and 'output' in input_data:
                    sender_id = input_data['sender_id']
                    output = input_data['output']
                    # Convert output to list if it's not already
                    if not isinstance(output, list):
                        output = [float(output)]
                    self.collected_outputs[sender_id] = output
            
            # Handle tuple input (sender_id, output)
            elif isinstance(input_data, tuple) and len(input_data) == 2:
                sender_id, output = input_data
                # Convert output to list if it's not already
                if not isinstance(output, list):
                    output = [float(output)]
                self.collected_outputs[sender_id] = output
            
            # Handle list input - this should be treated as a single result from one node
            elif isinstance(input_data, list):
                # If we don't have any collected outputs yet, treat this as the first input
                if not self.collected_outputs and self.expected_nodes:
                    # Assign to the first expected node if we don't know the sender
                    first_node = self.expected_nodes[0]
                    self.collected_outputs[first_node] = input_data
                elif len(self.collected_outputs) < len(self.expected_nodes):
                    # Find the next expected node that doesn't have data yet
                    for node_id in self.expected_nodes:
                        if node_id not in self.collected_outputs:
                            self.collected_outputs[node_id] = input_data
                            break
            
            # Handle simple scalar input
            elif isinstance(input_data, (int, float)):
                # If we don't have any collected outputs yet, treat this as the first input
                if not self.collected_outputs and self.expected_nodes:
                    first_node = self.expected_nodes[0]
                    self.collected_outputs[first_node] = [float(input_data)]
                elif len(self.collected_outputs) < len(self.expected_nodes):
                    # Find the next expected node that doesn't have data yet
                    for node_id in self.expected_nodes:
                        if node_id not in self.collected_outputs:
                            self.collected_outputs[node_id] = [float(input_data)]
                            break
            
            # If we have all expected outputs, forward to the review node
            result = None
            if self.ready_to_forward():
                result = self.forward_to_review()
                
            # Return the result if we have one, otherwise return current collection state
            if result is not None:
                return result
            else:
                return dict(self.collected_outputs)  # Return a copy to avoid mutation issues
            
        except Exception as e:
            print(f"Error in Retainer {self.node_id} process: {str(e)}")
            # Ensure expected_nodes is iterable before using it
            if not hasattr(self, 'expected_nodes') or self.expected_nodes is None:
                self.expected_nodes = []
            elif isinstance(self.expected_nodes, (int, float)):
                self.expected_nodes = [int(self.expected_nodes)]
            elif not isinstance(self.expected_nodes, (list, tuple)):
                self.expected_nodes = []
            # Return a safe fallback
            return {node_id: [0.25, 0.25, 0.25, 0.25] for node_id in self.expected_nodes}
class Handler:
    def __init__(self, node_id, node_position, num_reviewers, confidence_threshold=0.7, max_random=0.0, min_random=0.0, constant=1.0, demo=False, num_classes=10, output_config=None):
        self.node_id = node_id
        self.node_position = node_position
        self.num_reviewers = num_reviewers
        self.confidence_threshold = confidence_threshold  # Minimum confidence for final decision
        self.demo = demo
        self.num_classes = num_classes  # Configurable number of output classes
        self.output_config = output_config or {}  # Store output configuration
        self.define_node_weights(max_random, min_random, constant)
        self.init_reviewer_weights()
        self.init_decision_metrics()
        self.node_type = "Handler"
        
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=1.0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }
    
    def init_reviewer_weights(self):
        # Dynamic weights for each reviewer based on historical performance
        self.reviewer_weights = [1.0 for _ in range(self.num_reviewers)]
        self.reviewer_accuracy_history = [[] for _ in range(self.num_reviewers)]
    
    def init_decision_metrics(self):
        self.decision_history = []
        self.confidence_scores = []
        self.consensus_metrics = []
    
    def update_reviewer_weight(self, reviewer_index, performance_score):
        """Update reviewer weight based on performance feedback"""
        self.reviewer_accuracy_history[reviewer_index].append(performance_score)
        
        # Keep only recent history (last 10 decisions)
        if len(self.reviewer_accuracy_history[reviewer_index]) > 10:
            self.reviewer_accuracy_history[reviewer_index].pop(0)
        
        # Update weight based on recent average performance
        recent_avg = sum(self.reviewer_accuracy_history[reviewer_index]) / len(self.reviewer_accuracy_history[reviewer_index])
        self.reviewer_weights[reviewer_index] = max(0.1, min(recent_avg * 2.0, 3.0))  # Clamp between 0.1 and 3.0
    
    def softmax(self, z):
        max_z = max(z)
        exp_z = [math.exp(val - max_z) for val in z]
        total = sum(exp_z)
        return [x / total for x in exp_z]
    
    def entropy(self, probs):
        return -sum(p * math.log(p + 1e-9) for p in probs)
    
    def calculate_consensus(self, reviewer_outputs):
        """Calculate how much reviewers agree with each other"""
        if len(reviewer_outputs) < 2:
            return 1.0  # Perfect consensus with single reviewer
        
        consensus_score = 0.0
        comparisons = 0
        
        for i in range(len(reviewer_outputs)):
            for j in range(i + 1, len(reviewer_outputs)):
                # Safely check for rejected reviews - handle numpy arrays
                output_i_rejected = (isinstance(reviewer_outputs[i], list) and reviewer_outputs[i] == ["REVIEW_REJECTED"]) or \
                                  (hasattr(reviewer_outputs[i], 'tolist') and reviewer_outputs[i].tolist() == ["REVIEW_REJECTED"])
                output_j_rejected = (isinstance(reviewer_outputs[j], list) and reviewer_outputs[j] == ["REVIEW_REJECTED"]) or \
                                  (hasattr(reviewer_outputs[j], 'tolist') and reviewer_outputs[j].tolist() == ["REVIEW_REJECTED"])
                
                if output_i_rejected or output_j_rejected:
                    continue
                
                # Calculate similarity between two probability distributions
                similarity = sum(abs(reviewer_outputs[i][k] - reviewer_outputs[j][k]) 
                               for k in range(len(reviewer_outputs[i])))
                consensus_score += 1.0 - (similarity / 2.0)  # Convert distance to similarity
                comparisons += 1
        
        return consensus_score / comparisons if comparisons > 0 else 0.0
    
    def aggregate_decisions(self, reviewer_outputs):
        """
        Aggregate outputs from multiple reviewers into final decision
        reviewer_outputs: List of outputs from Reviewer nodes
        """
        if not reviewer_outputs:
            return {"decision": [], "confidence": 0.0, "consensus": 0.0, "status": "NO_INPUT"}
        
        # Filter out rejected reviews
        valid_outputs = []
        for output in reviewer_outputs:
            # Check if output is a rejection (handle numpy arrays safely)
            if isinstance(output, list) and output == ["REVIEW_REJECTED"]:
                continue  # Skip rejected reviews
            else:
                valid_outputs.append(output)
        rejected_count = len(reviewer_outputs) - len(valid_outputs)
        
        if not valid_outputs:
            return {
                "decision": ["ALL_REVIEWS_REJECTED"], 
                "confidence": 0.0, 
                "consensus": 0.0, 
                "status": "ALL_REJECTED",
                "rejected_count": rejected_count
            }
        
        # Calculate consensus before aggregation
        consensus = self.calculate_consensus(valid_outputs)
        
        # Weighted aggregation of valid outputs
        dim = len(valid_outputs[0]) if valid_outputs else self.num_classes
        aggregate_vector = [0.0] * dim
        total_weight = 0.0
        
        for i, probs in enumerate(valid_outputs):
            # Use reviewer weight (accounting for rejected reviews)
            reviewer_idx = i if rejected_count == 0 else i  # Simplified indexing
            weight = self.reviewer_weights[min(reviewer_idx, len(self.reviewer_weights) - 1)]
            
            for j in range(dim):
                noise = random.uniform(self.weights['Min_random'], self.weights['Max_random'])
                aggregate_vector[j] += (probs[j] + noise) * weight * self.weights['constant']
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            aggregate_vector = [val / total_weight for val in aggregate_vector]
        
        final_probs = self.softmax(aggregate_vector)
        
        # Calculate confidence (inverse of entropy, normalized)
        max_entropy = math.log(len(final_probs))
        confidence = 1.0 - (self.entropy(final_probs) / max_entropy) if max_entropy > 0 else 0.0
        
        # Determine status based on confidence and consensus
        if confidence < self.confidence_threshold:
            status = "LOW_CONFIDENCE"
        elif consensus < 0.5:
            status = "LOW_CONSENSUS"
        else:
            status = "ACCEPTED"
        
        # Store metrics for analysis
        decision_info = {
            "decision": final_probs,
            "confidence": confidence,
            "consensus": consensus,
            "status": status,
            "rejected_count": rejected_count,
            "valid_reviewers": len(valid_outputs),
            "entropy": self.entropy(final_probs)
        }
        
        # Update internal tracking
        self.decision_history.append(final_probs)
        self.confidence_scores.append(confidence)
        self.consensus_metrics.append(consensus)
        
        # Keep history limited
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
            self.confidence_scores.pop(0)
            self.consensus_metrics.pop(0)
        
        return decision_info
    
    def provide_feedback(self, reviewer_outputs, ground_truth_reward):
        """
        Provide feedback to individual reviewers based on final outcome
        This can be called after external validation of the Handler's decision
        """
        # Use safe comparison for numpy arrays vs lists
        def is_review_rejected(output):
            if isinstance(output, list):
                return output == ["REVIEW_REJECTED"]
            elif hasattr(output, 'tolist'):  # numpy array
                return output.tolist() == ["REVIEW_REJECTED"]
            return False
        
        valid_outputs = [output for output in reviewer_outputs if not is_review_rejected(output)]
        
        for i, output in enumerate(valid_outputs):
            # Calculate how close this reviewer was to the final aggregated decision
            if len(self.decision_history) > 0:
                final_decision = self.decision_history[-1]
                similarity = 1.0 - sum(abs(output[j] - final_decision[j]) for j in range(len(output))) / 2.0
                performance_score = similarity * ground_truth_reward
                
                # Update reviewer weight based on performance
                reviewer_idx = min(i, len(self.reviewer_weights) - 1)
                self.update_reviewer_weight(reviewer_idx, performance_score)
        
        print(f"[Handler {self.node_id}] Updated reviewer weights: {[f'{w:.2f}' for w in self.reviewer_weights]}")
    
    def get_performance_summary(self):
        """Return summary of Handler's performance metrics"""
        if not self.confidence_scores:
            return {"message": "No decisions processed yet"}
        
        return {
            "avg_confidence": sum(self.confidence_scores) / len(self.confidence_scores),
            "avg_consensus": sum(self.consensus_metrics) / len(self.consensus_metrics),
            "total_decisions": len(self.decision_history),
            "reviewer_weights": dict(enumerate(self.reviewer_weights))
        }
    
    def process(self, input_data):
        """
        Process input data through the Handler.
        Expected input format: dict with 'reviewer_results', 'controller_weights', 'final_probabilities'
        """
        if isinstance(input_data, dict) and 'reviewer_results' in input_data:
            # Extract reviewer outputs from the structured input
            reviewer_results = input_data['reviewer_results']
            reviewer_outputs = []
            
            for reviewer_id, reviewer_data in reviewer_results.items():
                if 'probabilities' in reviewer_data:
                    reviewer_outputs.append(reviewer_data['probabilities'])
            
            # Use the existing aggregate_decisions method
            return self.aggregate_decisions(reviewer_outputs)
        
        elif isinstance(input_data, list):
            # Direct list of reviewer outputs
            return self.aggregate_decisions(input_data)
        
        else:
            # Fallback for simple processing
            if self.demo:
                print(f"[Handler {self.node_id}] Processing simple input: {type(input_data)}")
            
            # Return a basic decision structure with configurable number of classes
            default_probs = [1.0] + [0.0] * (self.num_classes - 1)  # Default to first class
            return {
                "decision": default_probs,
                "confidence": 0.5,
                "consensus": 0.5,
                "status": "BASIC_PROCESSING"
            }


class Reviewer:
    def __init__(self, node_id, node_position, num_comps, entropy_threshold=1.5, max_random=0.0, min_random=0.0, constant=1.0, demo=False):
        self.node_id = node_id
        self.node_position = node_position
        self.demo = demo
        self.num_comps = num_comps
        self.entropy_threshold = entropy_threshold  # Entropy gate
        self.define_node_weights(max_random, min_random, constant)
        self.init_trust_scores()
        self.node_type = "Reviewer"
    def define_node_weights(self, max_random=0.0, min_random=0.0, constant=1.0):
        self.weights = {
            'Max_random': max_random,
            'Min_random': min_random,
            'constant': constant
        }

    def init_trust_scores(self):
        # Start with neutral trust (1.0) for each comp node
        self.trust_scores = [1.0 for _ in range(self.num_comps)]

    def update_trust(self, comp_index, reward=1.0):
        # Reward/punish trust based on outcome (simulated here as a value)
        self.trust_scores[comp_index] += reward
        self.trust_scores[comp_index] = max(0.1, min(self.trust_scores[comp_index], 10.0))  # Clamp trust

    def softmax(self, z):
        max_z = max(z)
        exp_z = [math.exp(val - max_z) for val in z]
        total = sum(exp_z)
        return [x / total for x in exp_z]

    def entropy(self, probs):
        return -sum(p * math.log(p + 1e-9) for p in probs)

    def aggregate(self, comp_outputs):
        """
        comp_outputs: List of List[float] — softmaxed outputs from comp nodes
        """
        try:
            # Safety check for empty inputs
            if not comp_outputs:
                print(f"[Review {self.node_id}] Empty comp_outputs received")
                return ["REVIEW_REJECTED"]
                
            # Ensure all outputs are valid
            valid_outputs = []
            for output in comp_outputs:
                if isinstance(output, list) and len(output) > 0 and all(isinstance(x, (int, float)) for x in output):
                    valid_outputs.append(output)
                    
            if not valid_outputs:
                print(f"[Review {self.node_id}] No valid outputs found in comp_outputs")
                return ["REVIEW_REJECTED"]
                
            # Get the dimension from the first valid output
            dim = len(valid_outputs[0])
            aggregate_vector = [0.0] * dim

            # Weighted aggregation by trust and noise
            for i, probs in enumerate(valid_outputs):
                # Use the trust score for this comp node if available, otherwise default to 1.0
                trust = self.trust_scores[i] if i < len(self.trust_scores) else 1.0
                
                # Ensure probs has the correct length
                if len(probs) != dim:
                    # Resize probs to match dim
                    if len(probs) < dim:
                        probs = probs + [0.0] * (dim - len(probs))
                    else:
                        probs = probs[:dim]
                        
                for j in range(dim):
                    noise = random.uniform(self.weights['Min_random'], self.weights['Max_random'])
                    aggregate_vector[j] += (probs[j] + noise) * trust * self.weights['constant']

            output_probs = self.softmax(aggregate_vector)

            # Entropy-based rejection or fallback
            if self.entropy(output_probs) > self.entropy_threshold:
                print(f"[Review {self.node_id}] ⚠️ High entropy detected, fallback or flag triggered.")
                return ["REVIEW_REJECTED"]

            return output_probs
            
        except Exception as e:
            print(f"Error in Reviewer {self.node_id} aggregate: {str(e)}")
            return ["REVIEW_REJECTED"]
        
    def process(self, input_data):
        # Handle different input types
        comp_outputs = []
        
        if isinstance(input_data, str):
            # Not ideal for a Reviewer, but handle it by creating a dummy output
            comp_outputs = [
                [0.25, 0.25, 0.25, 0.25]  # Uniform distribution (high entropy)
            ]
        elif isinstance(input_data, list):
            if all(isinstance(item, list) for item in input_data):
                # If it's already a list of probability distributions
                comp_outputs = input_data
            elif len(input_data) > 0:
                # If it's a single distribution, wrap it
                if all(isinstance(item, (int, float)) for item in input_data):
                    comp_outputs = [input_data]
                else:
                    # Try to extract probability distributions
                    valid_outputs = []
                    for item in input_data:
                        if isinstance(item, list) and all(isinstance(x, (int, float)) for x in item):
                            valid_outputs.append(item)
                    if valid_outputs:
                        comp_outputs = valid_outputs
                    else:
                        # Fallback: create a uniform distribution
                        comp_outputs = [[1.0/self.num_comps] * self.num_comps]
        else:
            # Fallback: create a uniform distribution
            comp_outputs = [[1.0/self.num_comps] * self.num_comps]
        
        # If we don't have enough outputs, pad with uniform distributions
        while len(comp_outputs) < self.num_comps:
            comp_outputs.append([1.0/self.num_comps] * self.num_comps)
            
        # Truncate if we have too many
        if len(comp_outputs) > self.num_comps:
            comp_outputs = comp_outputs[:self.num_comps]
            
        # Use the aggregate method for processing
        return self.aggregate(comp_outputs)