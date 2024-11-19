import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from typing import List, Dict, Tuple, Optional


class DataGenerator:
    def __init__(
        self,
        max_layers: int = 3,
        max_nodes_per_layer: int = 5,
        max_instance_params: int = 10,
        max_operations: int = 15,
        mod_value: int = 100,
        force: bool = False,
        seed: Optional[int] = None,
    ):
        self.max_layers = max_layers
        self.max_nodes_per_layer = max_nodes_per_layer
        self.max_instance_params = max_instance_params
        self.max_operations = max_operations
        self.mod_value = mod_value
        self.force = force
        if seed is not None:
            random.seed(seed)
        self.structure_graph = None
        self.dependency_graph = None
        self.query_parameter = None
        self.topo_order = None
        self.problem_description = ""
        self.question_description = ""
        self.solution_description = ""
        # Initialize hierarchical categories
        self.hierarchical_categories = self._initialize_categories()

    def _initialize_categories(self) -> List[List[str]]:
        """
        Initializes hierarchical categories with predefined categorizations and items.

        :return: A list of categorizations, each being a list of categories.
        """
        categorizations = [
            ["District", "Supermarket", "Product", "Ingredient"],
            ["Zoo", "Enclosure", "Animal", "Bone"],
            ["School", "Classroom", "Backpack", "Stationery"],
            ["Ecosystems", "Creatures", "Organs", "Cells"],
        ]

        # Prepare category items with subcategories and items
        self.category_items = {
            "District": {
                "Commercial Districts": [
                    "Shopping District",
                    "Business District",
                    "Financial District",
                    "Industrial District",
                    "Warehouse District",
                    "Market District",
                    "Restaurant District",
                    "Entertainment District",
                    "Arts District",
                    "Fashion District",
                    "Silicon Valley",
                    "Wall Street",
                    "Tech Park",
                    "Automotive District",
                    "Jewelry District",
                    "Medical District",
                    "Legal District",
                    "Media District",
                    "Research Park",
                    "Manufacturing District",
                ],
                # Additional subcategories...
            },
            "Supermarket": {
                "Supermarkets": [f"Supermarket {i+1}" for i in range(100)],
                # Additional subcategories...
            },
            "Product": {
                "Snack Foods": [
                    "Potato Chips",
                    "Pretzels",
                    "Popcorn",
                    "Candy Bars",
                    "Gummy Candy",
                    "Cookies",
                    "Crackers",
                    "Granola Bars",
                    "Fruit Snacks",
                    "Cheese Puffs",
                    "Nuts",
                    "Trail Mix",
                    "Beef Jerky",
                    "Rice Cakes",
                    "Yogurt Covered Raisins",
                    "Chocolate Covered Pretzels",
                    "Tortilla Chips",
                    "Salsa",
                    "Hummus",
                    "Dried Fruit",
                ],
                # Additional subcategories...
            },
            "Ingredient": {
                "Spices": [f"Spice {i+1}" for i in range(100)],
                # Additional subcategories...
            },
            # Other categories with their subcategories and items...
            # ...
        }

        return categorizations

    def generate_data(self):
        # Generate structure graph G_s
        self.structure_graph, layers = self._generate_structure_graph()
        # Generate dependency graph G_d
        self.dependency_graph = self._generate_dependency_graph(self.structure_graph)
        # Generate problem, question, and solution descriptions
        self._generate_descriptions()

    def _generate_structure_graph(
        self,
        min_items_per_layer: int,
        max_items_per_layer: int,
        num_layers: Optional[int],
    ) -> Tuple[nx.Graph, List[List[str]]]:
        """
        Generates the structure graph as per Algorithm 1.

        :return: A tuple containing the structure graph and the layers.
        """
        # BEGIN INIT
        # If num_layers is not specified, randomly choose num_layers ∈ {2,3,4}
        if num_layers is None:
            num_layers = random.randint(2, 4)

        # Compute min and max number of edges
        min_edges = (num_layers - 1) * min_items_per_layer
        max_edges = (num_layers - 1) * max_items_per_layer**2

        # Ensure num_edges is within valid range
        num_edges = random.randint(min_edges, max_edges)

        # Initialize layer_sizes = [min_items_per_layer] * num_layers
        layer_sizes = [min_items_per_layer] * num_layers

        probability = random.uniform(0, 1)
        # END INIT

        # BEGIN WHILE LOOP
        while layer_sizes != [max_items_per_layer] * num_layers:
            # Compute current min and max possible edges
            current_min_edges = sum(layer_sizes[1:])  # e^- = l_2 + l_3 + ... + l_d
            current_max_edges = sum(
                layer_sizes[i] * layer_sizes[i + 1] for i in range(num_layers - 1)
            )  # e^+ = l_1*l_2 + l_2*l_3 + ...

            if current_max_edges < num_edges:
                # Increase layer size
                indices = [
                    i for i in range(num_layers) if layer_sizes[i] < max_items_per_layer
                ]
                i = random.choice(indices)
                layer_sizes[i] += 1
            elif current_min_edges == num_edges:
                break
            elif random.uniform(0, 1) < probability:
                indices = [
                    i for i in range(num_layers) if layer_sizes[i] < max_items_per_layer
                ]
                if indices:
                    i = random.choice(indices)
                    layer_sizes[i] += 1
                else:
                    break
            else:
                break

        # Construct the structure graph
        layers = []
        for i in range(num_layers):
            layer_nodes = [f"Layer{i + 1}_Node{j + 1}" for j in range(layer_sizes[i])]
            layers.append(layer_nodes)

        structure_graph = nx.Graph()

        # Randomly select categories for each layer
        categories = self._select_categories(num_layers)

        for i, layer_nodes in enumerate(layers):
            layer_number = i + 1  # Layers are 1-indexed
            category = categories[i]
            for node in layer_nodes:
                structure_graph.add_node(node, layer=layer_number, category=category)

        # Create edges between adjacent layers
        for i in range(1, num_layers):
            for node in layers[i]:
                parent_node = random.choice(layers[i - 1])
                structure_graph.add_edge(node, parent_node)

        # Add additional edges until reaching num_edges
        total_edges = structure_graph.number_of_edges()
        while total_edges < num_edges:
            i = random.randint(
                0, num_layers - 2
            )  # layer indices from 0 to num_layers - 2
            node_a = random.choice(layers[i])
            node_b = random.choice(layers[i + 1])
            if not structure_graph.has_edge(node_a, node_b):
                structure_graph.add_edge(node_a, node_b)
                total_edges += 1

        # Assign English names to the nodes
        structure_graph = self._assign_names(structure_graph, layers)

        return structure_graph, layers

    def _select_categories(self, num_layers: int) -> List[str]:
        """
        Selects categories for each layer from hierarchical categories.

        :param num_layers: Number of layers.
        :return: A list of categories for each layer.
        """
        # Randomly select a categorization
        categorization = random.choice(self.hierarchical_categories)

        # Randomly select consecutive categories matching the number of layers
        max_start_index = len(categorization) - num_layers
        start_index = random.randint(0, max_start_index)
        selected_categories = categorization[start_index : start_index + num_layers]

        return selected_categories

    def _assign_names(
        self, structure_graph: nx.Graph, layers: List[List[str]]
    ) -> nx.Graph:
        """
        Assigns English names to the nodes in the structure graph.

        :param structure_graph: The structure graph with nodes.
        :param layers: A list of layers with node names.
        :return: The updated structure graph with English names assigned.
        """
        # Map to keep track of used names in each layer to ensure uniqueness
        used_names_per_layer: Dict[int, Set[str]] = {}

        # Assign English names to each node in the structure graph
        for i, layer_nodes in enumerate(layers):
            layer_number = i + 1  # Layers are 1-indexed
            category = structure_graph.nodes[layer_nodes[0]]["category"]
            used_names = set()
            used_names_per_layer[layer_number] = used_names

            # Get items for the category
            items = self.category_items.get(category, [])
            if not items:
                items = [f"{category} {j + 1}" for j in range(100)]
            else:
                # Flatten subcategories if necessary
                if isinstance(items, dict):
                    items = [item for sublist in items.values() for item in sublist]

            # Shuffle items to randomize selection
            random.shuffle(items)
            item_index = 0

            for node in layer_nodes:
                # Find an unused name
                while item_index < len(items):
                    item_name = items[item_index]
                    item_index += 1
                    if item_name not in used_names:
                        used_names.add(item_name)
                        break
                else:
                    # If we run out of unique names, generate a new one
                    item_name = f"{category} {len(used_names) + 1}"
                    used_names.add(item_name)

                # Assign the English name to the node
                structure_graph.nodes[node]["english_name"] = item_name

        return structure_graph

    def _generate_abstract_parameter_graph(
        self, structure_graph: nx.Graph, layers: List[List[str]]
    ) -> nx.Graph:
        """
        Generates the abstract parameter graph G_a based on the structure graph layers.

        :param structure_graph: The undirected structure graph G_s.
        :param layers: A list of layers with node names from the structure graph.
        :return: The abstract parameter graph G_a.
        """
        G_a = nx.DiGraph()

        num_layers = len(layers)

        # For each higher layer node, create abstract parameters for connected lower layers
        for i in range(num_layers - 1):  # For layers 0 to num_layers - 2
            higher_layer_nodes = layers[i]
            connected_nodes = {}
            for j in range(i + 1, num_layers):
                print(i, j)
                lower_layer_nodes = layers[j]
                lower_layer_category = structure_graph.nodes[lower_layer_nodes[0]][
                    "category"
                ]
                for node in higher_layer_nodes:
                    # Check if the node is connected to any node in the lower layer
                    if not connected_nodes.get(node, None):
                        connected_nodes[node] = structure_graph.neighbors(node)
                    else:
                        connected_extends = []
                        for n in connected_nodes[node]:
                            ce = [
                                neighbor
                                for neighbor in structure_graph.neighbors(n)
                                if neighbor in lower_layer_nodes
                            ]
                            connected_extends.extend(ce)
                        connected_nodes[node] = connected_extends
                    if connected_nodes[node]:
                        # Create an abstract parameter for this node and lower layer
                        param_name = f"Abstract_{node}_Layer{j}"
                        english_name = f"{structure_graph.nodes[node]['english_name']}'s {lower_layer_category}"
                        G_a.add_node(
                            param_name,
                            difficulty_level=j - i,
                            category=lower_layer_category,
                            english_name=english_name,
                            layers=[i + 1, 1 + j],
                            node=node,
                        )
                        # The abstract parameter depends on its instance parameters (from the lower layer)
                        for child_node in connected_nodes:
                            G_a.add_edge(child_node, param_name)

        return G_a

    # Visualization

    def visualize_structure_graph(self, structure_graph):
        """
        Visualizes the structure graph G_s.
        """
        pos = nx.multipartite_layout(structure_graph, subset_key="layer")
        plt.figure(figsize=(12, 8))
        node_colors = []
        for node in structure_graph.nodes:
            node_colors.append("lightblue")
            # if self._is_abstract(node):
            #     node_colors.append("lightblue")
            # else:
            #     node_colors.append("lightgreen")
        nx.draw_networkx_nodes(
            structure_graph, pos, node_color=node_colors, node_size=800
        )
        nx.draw_networkx_labels(
            structure_graph,
            pos,
            labels={
                node: structure_graph.nodes[node]["english_name"]
                for node in structure_graph.nodes
            },
            font_family="Times New Roman",
            font_size=10,
        )
        nx.draw_networkx_edges(structure_graph, pos)
        plt.title("Structure Graph $G_s$", fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.show()
