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
    def __init__(self, position, max_x, logging_enabled=False, logger=None):
        self.logging_enabled = logging_enabled
        self.position = position
        self.weights = {'Feature': [], 'weight': []}  # feature_name → weight
        self.signal = None
        self.connected_nodes = []  # neighboring ProcessingNodes
        self.distance_to_origin = sum([coord ** 2 for coord in position]) ** 0.5
        self.max_x = max_x
        self.queued = False
        self.logger = logger
        self.geo_lr = .05
        self.geo_max_step = .5
    
    def __repr__(self) -> str:
        return f"ProcessingNode(pos={self.position})"

    def display(self, message, Loud):
        if self.logging_enabled:
            if self.logger is None:
                raise ValueError("Logger is not set for ProcessingNode.")
            message = (f"[ProcessingNode] {message}")
            self.logger.log(message, Loud)

    def update_weights(self, feature_weights, Loud):
        self.display(f"Updating weights: {feature_weights}", Loud= Loud)
        self.weights = feature_weights

    def update_geometry(self, residual, Loud):
        if self.last_contribution is None:
            self.display("No last contribution recorded; skipping geometry update.", Loud= Loud)
            return
        # Signed influence
        influence = residual * self.last_contribution

        if influence == 0:
            return

        # Direction: push helpful nodes inward, harmful outward
        sign = -1.0 if influence > 0 else 1.0

        # Unit vector from origin
        r = self.distance_to_origin + 1e-8
        radial_dir = [c / r for c in self.position]

        # Tangential component (rotates neighborhood)
        tangential = [-radial_dir[1], radial_dir[0]] if len(radial_dir) == 2 else radial_dir

        # Combined move (THIS breaks nearest neighbors intentionally)
        delta = [
            self.geo_lr * sign * (radial_dir[i] + 0.5 * tangential[i])
            for i in range(len(self.position))
        ]

        # Clamp movement
        norm = sum(d * d for d in delta) ** 0.5
        if norm > self.geo_max_step:
            delta = [d * self.geo_max_step / norm for d in delta]

        # Apply
        self.position = tuple(self.position[i] + delta[i] for i in range(len(self.position)))
        self.distance_to_origin = sum(c * c for c in self.position) ** 0.5
    
    def receive_signal(self, signal, Loud):
        self.display(f"Received signal: {signal}", Loud= Loud)
        if signal.life <= 0:
            self.display("Signal life expired; ignoring signal.", Loud= Loud)
            signal.alive = False
            return False
        self.signal = signal
        self.signal.position = self.position
        self.signal.visited_nodes.add(self)
        signal.life -= 1
        if self.weights['Feature'] == []:
            self.display("No weights set; Creating weights.", Loud= Loud)
            self.create_weights(signal.feature_relevance, Loud = Loud)
        
        return True

    def create_weights(self, feature_relevance, Loud):
        self.display(f"Creating weights based on feature relevance: {feature_relevance}", Loud= Loud)
        for feature, relevance in feature_relevance.items():
            self.weights['Feature'].append(feature)
            self.weights['weight'].append(relevance)
    
    def find_nearest_neighbors(self, all_nodes, Loud, percentage=0.03):
        self.display("Finding nearest neighbors.", Loud= Loud)
        distances = []
        for node in all_nodes:
            if node is not self:
                if abs(node.distance_to_origin - self.distance_to_origin) < self.max_x * 0.25:
                    dist = sum([(a - b) ** 2 for a, b in zip(self.position, node.position)]) ** 0.5
                    distances.append((dist, node))
        distances.sort(key=lambda x: x[0])
        num_neighbors = max(1, int(len(distances) * percentage))
        neighbors = [node for _, node in distances[:num_neighbors]]
        self.display(f"Nearest neighbors found: {neighbors}", Loud= Loud)
        self.connected_nodes = neighbors
        return neighbors
    
    def choose_next_node(self, Loud):
        self.queued = False
        if not self.connected_nodes:
            self.display("No connected nodes to choose from.", Loud= Loud)
            return None
        if self.signal is None:
            self.display("No signal present to determine next node.", Loud= Loud)
            return None

        self.display("Choosing next node based on distance weighting.", Loud= Loud)
        weights = []
        eligible_nodes = [node for node in self.connected_nodes if node.distance_to_origin > self.distance_to_origin and node not in self.signal.visited_nodes and node.signal is None]
        if not eligible_nodes:
            self.display("No baseline eligible nodes found to forward the signal. Checking to queue.", Loud= Loud)
            new_eligible_nodes = [node for node in self.connected_nodes if node.distance_to_origin > self.distance_to_origin and node not in self.signal.visited_nodes]
            if new_eligible_nodes:
                self.queued = True
                self.display(f"Queuing signal at node: {self}. Eligible nodes found: {new_eligible_nodes}", Loud= Loud)
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
                self.display(f"Chosen next node: {node}", Loud= Loud)
                return node
    
    def process(self, Loud):
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
        self.last_contribution = abs(μ_new - μ)
        return self.signal
    
    def reset(self, Loud):
        self.display("Resetting ProcessingNode state.", Loud= Loud)
        self.signal = None


    def train(self, y, Loud, lr=0.01):
        if self.signal is None:
            return
        self.signal.life += 1  # Temporarily prevent decay during training
        μ = self.signal.prediction
        σ2 = max(self.signal.accumulated_variance, 1e-4)
        error = y - μ

        for i, feature in enumerate(self.weights['Feature']):
            x = self.signal.input_data.get(feature, 0.0)

            if isinstance(x, tuple):
                μ_f, _, _ = x
                x_val = μ_f
            else:
                x_val = x

            # Bayesian gradient update
            self.weights['weight'][i] += lr * error * x_val / σ2
        self.display(f"Weights updated to: {self.weights}", Loud= Loud)
