# Αλγόριθμοι Ελάχιστης Διαδρομής - Δομές Δεδομένων

**Ιόνιο Πανεπιστήμιο - Τμήμα Πληροφορικής**  
**Μάθημα:** Δομές Δεδομένων (Β΄ Εξάμηνο)  
**Ακαδημαϊκό Έτος:** 2024-2025  
**Διδάσκοντες:** Α. Καναβός, Α. Σωτηροπούλου

## 📝 Περιγραφή Εργασίας

Αυτή η εργασία υλοποιεί και συγκρίνει διάφορους αλγόριθμους εύρεσης ελάχιστης διαδρομής σε γράφους. Περιλαμβάνει τους αλγόριθμους **Dijkstra** και **Bellman-Ford** με τρεις διαφορετικές αναπαραστάσεις γράφων.

## ⚙️ Απαιτήσεις Συστήματος

### Λογισμικό
- **Python 3.8 ή νεότερη έκδοση**
- Λήψη από: https://www.python.org/downloads/

### Εξαρτήσεις Python
```bash
pip install matplotlib numpy
```

**Εναλλακτικά:**
```bash
pip install -r requirements.txt
```

## 🗂️ Δομή Έργου

```
📦 Εργασία_Αλγόριθμοι_Ελάχιστης_Διαδρομής/
│
├── 📄 README.md                    # Αυτό το αρχείο
├── 📄 requirements.txt             # Λίστα εξαρτήσεων Python
│
├── 🐍 main.py                      # Κύρια υλοποίηση
│   ├── class Graph                 # Κλάση γράφου (3 αναπαραστάσεις)
│   ├── dijkstra_matrix()          # Dijkstra + πίνακας γειτνίασης
│   ├── dijkstra_list()            # Dijkstra + λίστα γειτνίασης
│   ├── bellman_ford_matrix()      # Bellman-Ford + πίνακας γειτνίασης
│   ├── bellman_ford_list()        # Bellman-Ford + λίστα γειτνίασης
│   ├── bellman_ford_edges()       # Bellman-Ford + λίστα ακμών
│   ├── generate_random_graph()     # Δημιουργία τυχαίων γράφων
│   ├── run_experiments()          # Εκτέλεση πειραμάτων
│   └── create_performance_chart()  # Δημιουργία διαγραμμάτων
│
├── 🧪 test_algorithms.py           # Unit tests & επαλήθευση
├── ⚡ quick_demo.py                # Γρήγορη επίδειξη για δοκιμή
│
└── 📁 results/ (δημιουργείται αυτόματα)
    ├── 📊 execution_times.txt      # Αποτελέσματα σε κείμενο
    ├── 💾 execution_times.json     # Δεδομένα σε JSON μορφή
    └── 📈 performance_chart.png    # Διάγραμμα συγκριτικής απόδοσης
```

## 🚀 Οδηγίες Εκτέλεσης

### 1️⃣ Γρήγορη Δοκιμή (Συνιστάται για πρώτη φορά)
```bash
python quick_demo.py
```
**Διάρκεια:** ~2-5 λεπτά  
**Περιγραφή:** Δοκιμή με μικρούς γράφους (8-16 κόμβοι) για άμεσα αποτελέσματα

### 2️⃣ Πλήρης Εκτέλεση (Όπως ζητά η εργασία)
```bash
python main.py
```
**Διάρκεια:** ~30-60 λεπτά  
**Περιγραφή:** Πλήρη πειράματα με όλα τα μεγέθη γράφων (10-50 κόμβοι)

### 3️⃣ Δοκιμές Ορθότητας
```bash
python test_algorithms.py
```
**Διάρκεια:** ~1 λεπτό  
**Περιγραφή:** Επαλήθευση ότι όλοι οι αλγόριθμοι δίνουν σωστά αποτελέσματα

## ⏱️ Εκτιμώμενοι Χρόνοι Εκτέλεσης

| Μέγεθος Γράφου | Εκτιμώμενος Χρόνος | Περιγραφή |
|----------------|-------------------|-----------|
| 10 κόμβοι      | < 5 δευτερόλεπτα  | Άμεσο     |
| 20 κόμβοι      | < 30 δευτερόλεπτα | Γρήγορο   |
| 30 κόμβοι      | 2-8 λεπτά        | Μεσαίο    |
| 40 κόμβοι      | 8-20 λεπτά       | Αργό      |
| 50 κόμβοι      | 20-45 λεπτά      | Πολύ αργό |

⚠️ **Σημείωση:** Οι χρόνοι εξαρτώνται από την ισχύ του υπολογιστή σας

## 🔬 Υλοποιημένοι Αλγόριθμοι

### Dijkstra
1. **dijkstra_matrix()** - Με πίνακα γειτνίασης
2. **dijkstra_list()** - Με λίστα γειτνίασης + min-heap

### Bellman-Ford  
3. **bellman_ford_matrix()** - Με πίνακα γειτνίασης
4. **bellman_ford_list()** - Με λίστα γειτνίασης
5. **bellman_ford_edges()** - Με λίστα ακμών (προαιρετικό)

## 📊 Παράμετροι Πειραμάτων

| Αριθμός Κόμβων | Περιοχή Γράφου | Ακτίνα Σύνδεσης | Πυκνότητα |
|----------------|-----------------|-----------------|-----------|
| 10             | 50×50           | 25              | Υψηλή     |
| 20             | 50×50           | 20              | Μεσαία    |
| 30             | 100×100         | 30              | Μεσαία    |
| 40             | 100×100         | 25              | Χαμηλή    |
| 50             | 100×100         | 20              | Χαμηλή    |

## 📈 Αναμενόμενα Αποτελέσματα

### Ταξινόμηση κατά Απόδοση (ταχύτερο → αργότερο)
1. 🥇 **Dijkstra με λίστα γειτνίασης** - Καλύτερη συνολική απόδοση
2. 🥈 **Dijkstra με πίνακα γειτνίασης** - Καλή για πυκνούς γράφους  
3. 🥉 **Bellman-Ford με λίστα γειτνίασης** - Καλύτερος Bellman-Ford
4. 🏃 **Bellman-Ford με πίνακα γειτνίασης** - Αργός για μεγάλους γράφους
5. 🐌 **Bellman-Ford με λίστα ακμών** - Πιο αργός συνολικά

### Διαφορές Απόδοσης
- **Dijkstra vs Bellman-Ford:** ~10-70x ταχύτερος (ανάλογα με μέγεθος)
- **Λίστα vs Πίνακας:** ~20-40% βελτίωση για αραιούς γράφους

## 🔧 Αντιμετώπιση Προβλημάτων

### ❌ Σφάλμα: "ModuleNotFoundError: No module named 'matplotlib'"
**Λύση:**
```bash
pip install matplotlib numpy
```

### ⏳ Πολύ Αργή Εκτέλεση 
**Αιτίες & Λύσεις:**
- Δοκιμάστε πρώτα το `quick_demo.py` για γρήγορα αποτελέσματα
- Κλείστε άλλα προγράμματα που καταναλώνουν πολλούς πόρους
- Για δοκιμή μόνο με μικρούς γράφους, επεξεργαστείτε το `main.py`:
  ```python
  test_cases = [
      (10, 50, 25),  # Μόνο 10 κόμβοι για γρήγορη δοκιμή
      # (20, 50, 20), # Σχολιάστε τις υπόλοιπες
  ]
  ```

### 💾 Σφάλμα Μνήμης
**Λύσεις:**
- Κλείστε άλλες εφαρμογές
- Δοκιμάστε μικρότερους γράφους
- Επανεκκινήστε τον υπολογιστή

### 📊 Δεν Εμφανίζεται το Διάγραμμα
**Λύσεις:**
- **Windows:** Βεβαιωθείτε ότι έχετε GUI support
- **Linux:** `sudo apt-get install python3-tk`
- **macOS:** Το matplotlib συνήθως λειτουργεί out-of-the-box
- **Εναλλακτικά:** Το διάγραμμα αποθηκεύεται ως `performance_chart.png`

### 🐍 Λάθος Έκδοση Python
**Έλεγχος έκδοσης:**
```bash
python --version
# ή
python3 --version
```
**Αν έχετε Python < 3.8:** Κατεβάστε νεότερη έκδοση από python.org

## ⚙️ Προσαρμογή Παραμέτρων

### Αλλαγή Μεγεθών Γράφων
Στο αρχείο `main.py`, γραμμή ~150:
```python
test_cases = [
    (N, L, R),  # N=κόμβοι, L=περιοχή, R=ακτίνα
    (8, 30, 15),   # Προσθήκη μικρότερου γράφου
    (15, 60, 25),  # Προσθήκη μεσαίου γράφου
]
```

### Αλλαγή Αριθμού Επαναλήψεων
Για περισσότερη ακρίβεια στις μετρήσεις:
```python
# Στη συνάρτηση measure_algorithm_time()
for run in range(3):  # Αλλαγή από 1 σε 3 επαναλήψεις
    for start_vertex in range(num_vertices):
        # ...
```

### Προσθήκη Νέου Αλγορίθμου
1. Υλοποιήστε τη συνάρτηση στο `main.py`
2. Προσθέστε στο dictionary `algorithms`:
   ```python
   algorithms = {
       'Νέος Αλγόριθμος': νεα_συναρτηση,
       # ... υπόλοιποι αλγόριθμοι
   }
   ```

## 📋 Απαιτήσεις Παράδοσης

### ✅ Τι Περιλαμβάνεται στο .zip
```

├── README.md                 # Αυτό το αρχείο
├── main.py                   # Κύρια υλοποίηση  
├── test_algorithms.py        # Δοκιμές
├── quick_demo.py            # Γρήγορη επίδειξη
├── requirements.txt         # Εξαρτήσεις Python
└── Αναφορα_Εργασιας.pdf    # Αναφορά σε PDF
```

### ❌ Τι ΔΕΝ Περιλαμβάνεται
- Φάκελος `results/` (δημιουργείται αυτόματα)
- Αρχεία `.pyc` ή `__pycache__/`
- Εκτελέσιμα αρχεία
- Βίντεο ή άλλα μεγάλα αρχεία

### 🎯 Βήματα Παράδοσης
1. **Εκτελέστε** `python main.py` για να δημιουργηθούν αποτελέσματα
2. **Συμπληρώστε** τα στοιχεία της ομάδας σας στα αρχεία
3. **Δημιουργήστε** PDF με την ανάλυση αποτελεσμάτων  
4. **Συμπεριλάβετε** όλα τα αρχεία σε .zip
5. **Παραδώστε** μέχρι 6 Ιουλίου 2025

## 🎓 Πληροφορίες Μαθήματος

**Μάθημα:** Δομές Δεδομένων  
**Εξάμηνο:** Β΄ Εξάμηνο  
**Ίδρυμα:** Ιόνιο Πανεπιστήμιο  
**Τμήμα:** Πληροφορικής  
**Διδάσκοντες:** Α. Καναβός, Α. Σωτηροπούλου  

## 📜 Άδεια Χρήσης

Αυτός ο κώδικας δημιουργήθηκε για εκπαιδευτικούς σκοπούς στο πλαίσιο του μαθήματος "Δομές Δεδομένων" του Ιονίου Πανεπιστημίου. Μπορείτε να τον χρησιμοποιήσετε και να τον τροποποιήσετε για μάθηση και έρευνα.

---

## 🚀 Ξεκινώντας Γρήγορα

**Για πλήρη εκτέλεση (30-60 λεπτά):**
```bash
python main.py
```
