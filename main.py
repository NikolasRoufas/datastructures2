# Δομές Δεδομένων - Δεύτερη Εργασία
# Όνομα Ομάδας: GraphPathFinders
# Μέλη: [Συμπληρώστε τα στοιχεία σας]
# Ημερομηνία δημιουργίας: 09/06/2025
# Οδηγίες μεταγλώττισης: python main.py (Python 3.8+)

import random
import time
import math
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import os

class Graph:
    """Κλάση για αναπαράσταση γράφου"""
    
    def __init__(self, num_vertices):
        self.V = num_vertices
        # Πίνακας γειτνίασης
        self.adj_matrix = [[float('inf')] * num_vertices for _ in range(num_vertices)]
        # Λίστα γειτνίασης
        self.adj_list = defaultdict(list)
        # Λίστα ακμών
        self.edge_list = []
        
        # Διαγώνιος στο 0 (απόσταση κόμβου από τον εαυτό του)
        for i in range(num_vertices):
            self.adj_matrix[i][i] = 0
    
    def add_edge(self, u, v, weight):
        """Προσθήκη ακμής στον γράφο"""
        # Πίνακας γειτνίασης
        self.adj_matrix[u][v] = weight
        self.adj_matrix[v][u] = weight  # Μη κατευθυνόμενος γράφος
        
        # Λίστα γειτνίασης
        self.adj_list[u].append((v, weight))
        self.adj_list[v].append((u, weight))
        
        # Λίστα ακμών
        self.edge_list.append((u, v, weight))

def generate_random_graph(N, L, R):
    """
    Δημιουργία τυχαίου γράφου
    N: αριθμός κόμβων
    L: μέγεθος τετραγώνου
    R: ακτίνα σύνδεσης
    """
    # Δημιουργία τυχαίων σημείων
    points = []
    for i in range(N):
        x = random.uniform(0, L)
        y = random.uniform(0, L)
        points.append((x, y))
    
    # Δημιουργία γράφου
    graph = Graph(N)
    
    # Προσθήκη ακμών βάσει ακτίνας
    for i in range(N):
        for j in range(i + 1, N):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Υπολογισμός ευκλείδειας απόστασης
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Αν η απόσταση είναι εντός της ακτίνας
            if distance <= R:
                # Τυχαίο βάρος [1, 100]
                weight = random.randint(1, 100)
                graph.add_edge(i, j, weight)
    
    return graph, points

# DIJKSTRA ΑΛΓΟΡΙΘΜΟΣ

def dijkstra_matrix(graph, start):
    """Dijkstra με πίνακα γειτνίασης"""
    V = graph.V
    dist = [float('inf')] * V
    visited = [False] * V
    dist[start] = 0
    
    for _ in range(V):
        # Βρες τον κόμβο με τη μικρότερη απόσταση
        min_dist = float('inf')
        u = -1
        for v in range(V):
            if not visited[v] and dist[v] < min_dist:
                min_dist = dist[v]
                u = v
        
        if u == -1:
            break
            
        visited[u] = True
        
        # Ενημέρωση αποστάσεων γειτόνων
        for v in range(V):
            if (not visited[v] and 
                graph.adj_matrix[u][v] != float('inf') and
                dist[u] + graph.adj_matrix[u][v] < dist[v]):
                dist[v] = dist[u] + graph.adj_matrix[u][v]
    
    return dist

def dijkstra_list(graph, start):
    """Dijkstra με λίστα γειτνίασης και min-heap"""
    V = graph.V
    dist = [float('inf')] * V
    dist[start] = 0
    
    # Min-heap: (απόσταση, κόμβος)
    heap = [(0, start)]
    
    while heap:
        current_dist, u = heapq.heappop(heap)
        
        # Αν έχουμε ήδη βρει καλύτερη διαδρομή
        if current_dist > dist[u]:
            continue
            
        # Εξέταση γειτόνων
        for v, weight in graph.adj_list[u]:
            new_dist = dist[u] + weight
            
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    
    return dist

# BELLMAN-FORD ΑΛΓΟΡΙΘΜΟΣ

def bellman_ford_matrix(graph, start):
    """Bellman-Ford με πίνακα γειτνίασης"""
    V = graph.V
    dist = [float('inf')] * V
    dist[start] = 0
    
    # Χαλάρωση ακμών V-1 φορές
    for _ in range(V - 1):
        for u in range(V):
            if dist[u] != float('inf'):
                for v in range(V):
                    if (graph.adj_matrix[u][v] != float('inf') and
                        dist[u] + graph.adj_matrix[u][v] < dist[v]):
                        dist[v] = dist[u] + graph.adj_matrix[u][v]
    
    return dist

def bellman_ford_list(graph, start):
    """Bellman-Ford με λίστα γειτνίασης"""
    V = graph.V
    dist = [float('inf')] * V
    dist[start] = 0
    
    # Χαλάρωση ακμών V-1 φορές
    for _ in range(V - 1):
        for u in range(V):
            if dist[u] != float('inf'):
                for v, weight in graph.adj_list[u]:
                    if dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight
    
    return dist

def bellman_ford_edges(graph, start):
    """Bellman-Ford με λίστα ακμών (προαιρετικό)"""
    V = graph.V
    dist = [float('inf')] * V
    dist[start] = 0
    
    # Χαλάρωση ακμών V-1 φορές
    for _ in range(V - 1):
        for u, v, weight in graph.edge_list:
            # Χαλάρωση και στις δύο κατευθύνσεις
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
            if dist[v] != float('inf') and dist[v] + weight < dist[u]:
                dist[u] = dist[v] + weight
    
    return dist

def measure_algorithm_time(algorithm_func, graph, num_vertices):
    """Μέτρηση χρόνου εκτέλεσης αλγορίθμου για όλους τους κόμβους"""
    total_time = 0
    
    for start_vertex in range(num_vertices):
        start_time = time.time()
        algorithm_func(graph, start_vertex)
        end_time = time.time()
        total_time += (end_time - start_time)
    
    return total_time

def run_experiments():
    """Εκτέλεση πειραμάτων και συλλογή δεδομένων"""
    
    # Παράμετροι πειραμάτων
    test_cases = [
        (10, 50, 25),
        (20, 50, 20),
        (30, 100, 30),
        (40, 100, 25),
        (50, 100, 20)
    ]
    
    algorithms = {
        'Dijkstra Matrix': dijkstra_matrix,
        'Dijkstra List': dijkstra_list,
        'Bellman-Ford Matrix': bellman_ford_matrix,
        'Bellman-Ford List': bellman_ford_list,
        'Bellman-Ford Edges': bellman_ford_edges
    }
    
    results = {alg_name: [] for alg_name in algorithms.keys()}
    node_counts = []
    
    print("Ξεκίνημα πειραμάτων...")
    
    for N, L, R in test_cases:
        print(f"\nΠείραμα με {N} κόμβους, περιοχή {L}x{L}, ακτίνα {R}")
        node_counts.append(N)
        
        # Δημιουργία γράφου
        graph, points = generate_random_graph(N, L, R)
        
        # Μέτρηση κάθε αλγορίθμου
        for alg_name, alg_func in algorithms.items():
            print(f"  Εκτέλεση {alg_name}...")
            execution_time = measure_algorithm_time(alg_func, graph, N)
            results[alg_name].append(execution_time)
            print(f"    Χρόνος: {execution_time:.4f} δευτερόλεπτα")
    
    return results, node_counts

def save_results(results, node_counts):
    """Αποθήκευση αποτελεσμάτων σε αρχείο"""
    
    # Δημιουργία φακέλου για αποτελέσματα
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Αποθήκευση σε JSON
    data = {
        'node_counts': node_counts,
        'results': results
    }
    
    with open('results/execution_times.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Αποθήκευση σε κείμενο
    with open('results/execution_times.txt', 'w', encoding='utf-8') as f:
        f.write("Αποτελέσματα Μετρήσεων Χρόνου Εκτέλεσης\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Αριθμός Κόμβων: " + str(node_counts) + "\n\n")
        
        for alg_name, times in results.items():
            f.write(f"{alg_name}:\n")
            for i, time_val in enumerate(times):
                f.write(f"  {node_counts[i]} κόμβοι: {time_val:.6f} δευτερόλεπτα\n")
            f.write("\n")

def create_performance_chart(results, node_counts):
    """Δημιουργία διαγράμματος απόδοσης"""
    
    plt.figure(figsize=(12, 8))
    
    # Στυλ γραμμών και χρωμάτων
    styles = ['-o', '-s', '-^', '-D', '-v']
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (alg_name, times) in enumerate(results.items()):
        plt.plot(node_counts, times, styles[i], 
                color=colors[i], label=alg_name, 
                linewidth=2, markersize=8)
    
    plt.xlabel('Αριθμός Κόμβων', fontsize=12)
    plt.ylabel('Χρόνος Εκτέλεσης (δευτερόλεπτα)', fontsize=12)
    plt.title('Σύγκριση Απόδοσης Αλγορίθμων Ελάχιστης Διαδρομής', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Λογαριθμική κλίμακα για καλύτερη οπτικοποίηση
    
    # Αποθήκευση διαγράμματος
    plt.tight_layout()
    plt.savefig('results/performance_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Κύρια συνάρτηση εκτέλεσης"""
    
    print("Δομές Δεδομένων - Αλγόριθμοι Ελάχιστης Διαδρομής")
    print("=" * 55)
    
    # Ρύθμιση τυχαίου σπόρου για επαναληψιμότητα
    random.seed(42)
    
    # Εκτέλεση πειραμάτων
    results, node_counts = run_experiments()
    
    # Αποθήκευση αποτελεσμάτων
    print("\nΑποθήκευση αποτελεσμάτων...")
    save_results(results, node_counts)
    
    # Δημιουργία διαγράμματος
    print("Δημιουργία διαγράμματος απόδοσης...")
    create_performance_chart(results, node_counts)
    
    print("\nΗ εργασία ολοκληρώθηκε επιτυχώς!")
    print("Τα αποτελέσματα αποθηκεύτηκαν στον φάκελο 'results/'")

if __name__ == "__main__":
    main()