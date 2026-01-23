"""
 - processing:
 -- will take in the input and feature relevance
 -- using bayes theorem I will calculate a prediction
 -- will then forward the prediction to other processing nodes which add/subtract from prior predictions in a decreasing magnitude
 --- will use calculated distance from origin as a scalar 
 -- to determine where signals are passed:
 --- nearest 3% of nodes are found.
 --- distance from origin is used. based on amount i will find 100/node-count.
 --- this value will determine the chance of the node furthest from the origin. then that chance will be divided by 2 for the rest of the nodes, eg. 25, 12.5, 6.25, etc.
 --- then I will multiply by a 10^n scalar to ensure all are integers
 --- a random function will pick which node to be passed to. 
 --- nodes closer to the origin than the current are not eligible.
 -- will also add a variance score to the signal accumulating over time
 -- integrated ML to form an NN
"""
import random

class ProcessingNode:
    def __init__(self, position, max_x, logging_enabled=False):
        self.logging_enabled = logging_enabled
        self.position = position
        self.weights = {'Feature': [], 'weight': []}  # feature_name → weight
        self.signal = None
        self.connected_nodes = []  # neighboring ProcessingNodes
        self.distance_to_origin = sum([coord ** 2 for coord in position]) ** 0.5
        self.max_x = max_x
        self.queued = False
    
    def __repr__(self) -> str:
        return f"ProcessingNode(pos={self.position})"

    def display(self, message):
        if self.logging_enabled:
            print(f"[ProcessingNode] {message}")

    def update_weights(self, feature_weights):
        self.display(f"Updating weights: {feature_weights}")
        self.weights = feature_weights
    
    def receive_signal(self, signal):
        self.display(f"Received signal: {signal}")
        if signal.life <= 0:
            self.display("Signal life expired; ignoring signal.")
            signal.alive = False
            return False
        self.signal = signal
        self.signal.position = self.position
        self.signal.visited_nodes.add(self)
        signal.life -= 1
        if self.weights['Feature'] == []:
            self.display("No weights set; Creating weights.")
            self.create_weights(signal.feature_relevance)
        
        return True

    def create_weights(self, feature_relevance):
        self.display(f"Creating weights based on feature relevance: {feature_relevance}")
        for feature, relevance in feature_relevance.items():
            self.weights['Feature'].append(feature)
            self.weights['weight'].append(relevance)
    
    def find_nearest_neighbors(self, all_nodes, percentage=0.03):
        self.display("Finding nearest neighbors.")
        distances = []
        for node in all_nodes:
            if node is not self:
                if node.distance_to_origin >= self.distance_to_origin:
                    dist = sum([(a - b) ** 2 for a, b in zip(self.position, node.position)]) ** 0.5
                    distances.append((dist, node))
        distances.sort(key=lambda x: x[0])
        num_neighbors = max(1, int(len(distances) * percentage))
        neighbors = [node for _, node in distances[:num_neighbors]]
        self.display(f"Nearest neighbors found: {neighbors}")
        self.connected_nodes = neighbors
        return neighbors
    
    def choose_next_node(self):
        self.queued = False
        if not self.connected_nodes:
            self.display("No connected nodes to choose from.")
            return None
        if self.signal is None:
            self.display("No signal present to determine next node.")
            return None

        self.display("Choosing next node based on distance weighting.")
        weights = []
        eligible_nodes = [node for node in self.connected_nodes if node.distance_to_origin > self.distance_to_origin and node not in self.signal.visited_nodes and node.signal is None]
        if not eligible_nodes:
            self.display("No baseline eligible nodes found to forward the signal. Checking to queue.")
            new_eligible_nodes = [node for node in self.connected_nodes if node.distance_to_origin > self.distance_to_origin and node not in self.signal.visited_nodes]
            if new_eligible_nodes:
                self.queued = True
                self.display(f"Queuing signal at node: {self}. Eligible nodes found: {new_eligible_nodes}")
                return None
            return None
        base = 100 / len(eligible_nodes)
        for i in range(len(eligible_nodes)):
            weight = base / (2 ** i)
            weights.append(weight)
        total_weight = sum(weights)
        rand_val = random.uniform(0, total_weight)
        cumulative = 0
        for node, weight in zip(eligible_nodes, weights):
            cumulative += weight
            if rand_val <= cumulative:
                self.display(f"Chosen next node: {node}")
                return node
    
    def process(self):
        if self.signal is None:
            return None
        

        μ = self.signal.prediction
        o2 = max(self.signal.accumulated_variance, 1e-4)

        # --- Local evidence ---
        delta_mu = 0.0
        delta_var = 0.0

        for feature, weight in zip(self.weights['Feature'], self.weights['weight']):
            x = self.signal.input_data.get(feature, 0.0)

            # --- Evidence-aware handling ---
            if isinstance(x, tuple):
                μ_f, σ2_f, n = x
                precision_f = n / (σ2_f + 1e-8)

                delta_mu += weight * μ_f * precision_f
                delta_var += (weight ** 2) / precision_f
            else:
                # numeric feature
                delta_mu += weight * x
                delta_var += weight ** 2

        if delta_var == 0:
            return self.signal

        # --- Geometric attenuation ---
        decay = 1.0 / (1.0 + self.distance_to_origin)
        delta_var /= decay

        # --- Bayesian update (Gaussian) ---
        precision_prior = 1.0 / o2
        precision_local = 1.0 / delta_var

        μ_new = (
            μ * precision_prior + delta_mu * precision_local
        ) / (precision_prior + precision_local)

        o2_new = 1.0 / (precision_prior + precision_local)

        self.signal.prediction = μ_new
        self.signal.accumulated_variance = o2_new

        return self.signal
    
    def reset(self):
        self.display("Resetting ProcessingNode state.")
        self.signal = None


    
