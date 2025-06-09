# Δοκιμές και Επαλήθευση Αλγορίθμων
# Όνομα Ομάδας: GraphPathFinders
# Ημερομηνία δημιουργίας: 09/06/2025
# Οδηγίες μεταγλώττισης: python test_algorithms.py

from main import *
import unittest

class TestShortestPathAlgorithms(unittest.TestCase):
    
    def setUp(self):
        """Δημιουργία δοκιμαστικού γράφου"""
        # Μικρός γράφος για δοκιμές
        self.test_graph = Graph(5)
        
        # Προσθήκη ακμών
        edges = [
            (0, 1, 4),
            (0, 2, 2),
            (1, 2, 1),
            (1, 3, 5),
            (2, 3, 8),
            (2, 4, 10),
            (3, 4, 2)
        ]
        
        for u, v, w in edges:
            self.test_graph.add_edge(u, v, w)
    
    def test_dijkstra_consistency(self):
        """Έλεγχος συνέπειας Dijkstra αλγορίθμων"""
        for start in range(5):
            dist_matrix = dijkstra_matrix(self.test_graph, start)
            dist_list = dijkstra_list(self.test_graph, start)
            
            self.assertEqual(dist_matrix, dist_list, 
                           f"Διαφορά στον κόμβο εκκίνησης {start}")
    
    def test_bellman_ford_consistency(self):
        """Έλεγχος συνέπειας Bellman-Ford αλγορίθμων"""
        for start in range(5):
            dist_matrix = bellman_ford_matrix(self.test_graph, start)
            dist_list = bellman_ford_list(self.test_graph, start)
            dist_edges = bellman_ford_edges(self.test_graph, start)
            
            self.assertEqual(dist_matrix, dist_list, 
                           f"Διαφορά Matrix-List στον κόμβο {start}")
            self.assertEqual(dist_matrix, dist_edges, 
                           f"Διαφορά Matrix-Edges στον κόμβο {start}")
    
    def test_algorithm_consistency(self):
        """Έλεγχος συνέπειας μεταξύ Dijkstra και Bellman-Ford"""
        for start in range(5):
            dijkstra_result = dijkstra_matrix(self.test_graph, start)
            bellman_result = bellman_ford_matrix(self.test_graph, start)
            
            self.assertEqual(dijkstra_result, bellman_result,
                           f"Διαφορά Dijkstra-BellmanFord στον κόμβο {start}")

def run_small_example():
    """Εκτέλεση μικρού παραδείγματος"""
    print("Δοκιμή με μικρό γράφο")
    print("-" * 30)
    
    # Δημιουργία μικρού γράφου
    graph = Graph(4)
    edges = [(0, 1, 1), (0, 2, 4), (1, 2, 2), (1, 3, 5), (2, 3, 1)]
    
    for u, v, w in edges:
        graph.add_edge(u, v, w)
    
    print("Γράφος:")
    print("Ακμές: (0-1:1), (0-2:4), (1-2:2), (1-3:5), (2-3:1)")
    print()
    
    # Δοκιμή αλγορίθμων από κόμβο 0
    start = 0
    print(f"Ελάχιστες διαδρομές από κόμβο {start}:")
    
    dijkstra_m = dijkstra_matrix(graph, start)
    dijkstra_l = dijkstra_list(graph, start)
    bellman_m = bellman_ford_matrix(graph, start)
    bellman_l = bellman_ford_list(graph, start)
    bellman_e = bellman_ford_edges(graph, start)
    
    print(f"Dijkstra (Matrix):     {dijkstra_m}")
    print(f"Dijkstra (List):       {dijkstra_l}")
    print(f"Bellman-Ford (Matrix): {bellman_m}")
    print(f"Bellman-Ford (List):   {bellman_l}")
    print(f"Bellman-Ford (Edges):  {bellman_e}")
    
    # Έλεγχος ορθότητας
    expected = [0, 1, 3, 4]  # Αναμενόμενες αποστάσεις
    if dijkstra_m == expected:
        print("\n✓ Όλοι οι αλγόριθμοι δίνουν το σωστό αποτέλεσμα!")
    else:
        print(f"\n✗ Σφάλμα! Αναμενόμενο: {expected}")

def benchmark_small_graph():
    """Σύγκριση απόδοσης σε μικρό γράφο"""
    print("\nΣύγκριση απόδοσης (μικρός γράφος)")
    print("-" * 40)
    
    # Δημιουργία γράφου 20 κόμβων
    graph, _ = generate_random_graph(20, 50, 20)
    
    algorithms = {
        'Dijkstra Matrix': dijkstra_matrix,
        'Dijkstra List': dijkstra_list,
        'Bellman-Ford Matrix': bellman_ford_matrix,
        'Bellman-Ford List': bellman_ford_list,
        'Bellman-Ford Edges': bellman_ford_edges
    }
    
    print("Χρόνοι εκτέλεσης για 20 κόμβους:")
    for name, func in algorithms.items():
        start_time = time.time()
        for i in range(20):  # Όλοι οι κόμβοι
            func(graph, i)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{name:20}: {execution_time:.6f} δευτερόλεπτα")

def visualize_graph_generation():
    """Οπτικοποίηση δημιουργίας γράφου"""
    print("\nΔημιουργία και οπτικοποίηση γράφου")
    print("-" * 40)
    
    # Δημιουργία μικρού γράφου για οπτικοποίηση
    N, L, R = 10, 50, 25
    graph, points = generate_random_graph(N, L, R)
    
    print(f"Γράφος με {N} κόμβους, περιοχή {L}x{L}, ακτίνα {R}")
    print(f"Συνολικές ακμές: {len(graph.edge_list)}")
    
    # Εμφάνιση συντεταγμένων κόμβων
    print("\nΣυντεταγμένες κόμβων:")
    for i, (x, y) in enumerate(points):
        print(f"Κόμβος {i}: ({x:.2f}, {y:.2f})")
    
    # Εμφάνιση ακμών
    print(f"\nΑκμές (συνολικά {len(graph.edge_list)}):")
    for i, (u, v, w) in enumerate(graph.edge_list[:10]):  # Εμφάνιση πρώτων 10
        print(f"  {u} <-> {v} (βάρος: {w})")
    
    if len(graph.edge_list) > 10:
        print(f"  ... και {len(graph.edge_list) - 10} ακόμη")

def main():
    """Κύρια συνάρτηση δοκιμών"""
    print("Δοκιμές Αλγορίθμων Ελάχιστης Διαδρομής")
    print("=" * 45)
    
    # Εκτέλεση unit tests
    print("Εκτέλεση unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Εκτέλεση παραδειγμάτων
    run_small_example()
    benchmark_small_graph()
    visualize_graph_generation()

if __name__ == "__main__":
    main()