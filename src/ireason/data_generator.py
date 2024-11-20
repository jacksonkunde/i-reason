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
        print(f"num layers: {num_layers}")

        # Go from lower layers to higher layers, building abstract params as we go
        for i in range(num_layers - 2, -1, -1):
            print(f"top layer: {i}")
            higher_layer_nodes = layers[i]
            for node in higher_layer_nodes:
                print(f"curr node: {node}")
                # We only want to look at neighbors below us
                # Layers are 1 indexed
                neighbors = [
                    n
                    for n in structure_graph.neighbors(node)
                    if structure_graph.nodes[n]["layer"] == i + 2
                ]
                print(f"neighbors: {neighbors}")
                print(
                    f"their layers: {[structure_graph.nodes[n]["layer"] for n in structure_graph.neighbors(node)]}"
                )
                if neighbors:  # We need to create an abstract
                    # Get the category
                    lower_layer_nodes = layers[i + 1]
                    lower_layer_category = structure_graph.nodes[lower_layer_nodes[0]][
                        "category"
                    ]
                    english_name = f"{structure_graph.nodes[node]['english_name']}'s {lower_layer_category}"
                    param_name = f"Abstract_{node}_Layer{i + 1}_{english_name}"
                    print(f"adding 1 node: {english_name}")
                    print(param_name)
                    G_a.add_node(
                        param_name,
                        difficulty_level=1,
                        category=lower_layer_category,
                        english_name=english_name,
                        layers=[i + 1, i + 2],
                        node=node,
                    )
                    # Update original node from structure graph to show that it has a corresponding abstract_param
                    structure_graph.nodes[node]["abstract_nodes"] = (
                        structure_graph.nodes[node].get("abstract_nodes", [])
                        + [param_name]
                    )
                    print(f"neighbors: {neighbors}")
                    for nei in neighbors:
                        node_attributes = structure_graph.nodes[nei]
                        G_a.add_node(nei, **node_attributes)
                        G_a.add_edge(nei, param_name)
                        # Check if this node has an associated abstract param
                        associated_abstract_nodes = node_attributes.get(
                            "abstract_nodes", []
                        )
                        # We want to add new abstract params for our current node for any abstract params we depend on
                        for aa_node in associated_abstract_nodes:
                            associated_attributes = G_a.nodes[aa_node]
                            print(
                                f"associated abstract: {associated_attributes.get("english_name")}"
                            )
                            associated_difficulty = associated_attributes.get(
                                "difficulty_level"
                            )
                            print(f"ad: {associated_difficulty}")
                            associated_lower_layer_category = associated_attributes.get(
                                "category"
                            )
                            associated_english_name = f"{structure_graph.nodes[node]['english_name']}'s {associated_lower_layer_category}"
                            associated_param_name = f"Abstract_{node}_Layer{i + associated_difficulty}_{associated_english_name}"
                            print(f"adding node: {associated_english_name}")
                            print(param_name)
                            G_a.add_node(
                                associated_param_name,
                                difficulty_level=1 + associated_difficulty,
                                category=associated_lower_layer_category,
                                english_name=associated_english_name,
                                layers=[i + 1, i + 1 + associated_difficulty],
                                node=node,
                            )
                            G_a.add_edge(aa_node, associated_param_name)
        return G_a

    def _generate_G_d_nece1(
        self, structure_graph: nx.Graph, abstract_graph: nx.DiGraph, n: int, m: int
    ):
        """
        Generate a subgraph G_d_nece1 of the abstract dependency graph G_d such that
        the total number of operations required to compute its parameters is <= n.
        This is from the algorithm DrawNecessary1(Gs, n, m) in the paper the physics of llms

        Args:
            structure_graph (nx.Graph): The underlying structure graph.
            abstract_graph (nx.DiGraph): The dependency graph G_d with difficulty levels and dependencies.
            n (int): Maximum allowed number of operations for the subgraph.
            m (int): Not used in this function (can be extended later if needed).

        Returns:
            G_d_nece1 (nx.DiGraph): The subgraph with operations bounded by n.
        """
        G_d_nece1 = nx.DiGraph()

        updated = True
        while updated:
            updated = False
            # Get the maximum difficulty level of nodes in the abstract graph
            max_difficulty = max(
                [
                    abstract_graph.nodes[n].get("difficulty_level", 0)
                    for n in abstract_graph
                ]
            )
            # Loop backward through difficulty levels (highest to lowest)
            for curr_difficulty in range(max_difficulty, 0, -1):
                for node in abstract_graph:
                    # Check that we don't already have it and it is the correct difficulty level
                    if (
                        abstract_graph.nodes[node].get("difficulty_level")
                        == curr_difficulty
                        and node not in G_d_nece1.nodes
                    ):
                        attributes = abstract_graph.nodes[node]

                        # Create a temporary graph G'
                        G_prime = G_d_nece1.copy()

                        # Add the current node with its attributes to G_prime
                        G_prime.add_node(node, **attributes)

                        # Add all its dependencies (edges) to G_d_nece1
                        dependencies = self._get_node_dependencies(abstract_graph, node)
                        for d in dependencies:
                            # Add the dependencies to the graph and add edges
                            d_attributes = abstract_graph.nodes[d]
                            G_prime.add_node(d, **d_attributes)
                            G_prime.add_edge(d, node)

                        # Calculate the total number of operations op(G')
                        op_G_prime = self._calculate_op(G_prime)

                        # Check if op(G') <= n
                        if op_G_prime <= n:
                            G_d_nece1 = G_prime
                            updated = True
                            break

        return G_d_nece1

    def _calculate_op(self, graph: nx.DiGraph):
        # op_G(a) = max(1, t - 1)
        return sum(max(1, graph.in_degree(a) - 1) for a in graph.nodes)

    def _get_node_dependencies(self, graph: nx.DiGraph, node) -> List[str]:
        dependencies = []
        for predecessor in graph.predecessors(node):
            dependencies.append(predecessor)
        return dependencies

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
