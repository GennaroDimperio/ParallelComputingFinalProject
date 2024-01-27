import time
import numpy as np
import pycuda.gpuarray as gpuarray
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from pycuda.compiler import SourceModule


# Funzione k-means sequenziale
def kmeans_sequential(X, n_clusters, max_iter=200):
    # Inizializzazione dei centroidi
    random_state_seq = np.random.RandomState(seed=42)
    centers = X[random_state_seq.permutation(X.shape[0])[:n_clusters]]

    # Iterazione fino alla convergenza o al numero massimo di iterazioni
    for _ in range(max_iter):
        labels, _ = pairwise_distances_argmin_min(X, centers)

        # Aggiornamento dei centroidi in base ai punti assegnati
        new_centers = np.array([X[labels == i].mean(0) for i in range(len(centers))])

        # Verifica della convergenza confrontando i centroidi vecchi con i nuovi
        if np.all(centers == new_centers):
            break

        centers = new_centers

    # Calcolo dell'inerzia dopo la convergenza
    inertia = sum(np.min(pairwise_distances(X, centers), axis=1))
    return centers, labels, inertia

# Definizione del kernel CUDA
cuda_kernel = """
  // Assegnazione di cluster ai punti
  __global__ void kmeans_kernel(float* X, float* centers, int* labels, int n_points, int n_clusters, int n_features) {
    // Calcolo dell'ID del thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Verifica che il thread sia all'interno del range dei punti dati
    if (tid < n_points) {
        int min_label = -1;
        float min_distance = INFINITY;

        // Calcolo della distanza tra il punto X[tid] e ciascun centro
        for (int cluster = 0; cluster < n_clusters; ++cluster) {
            float distance = 0.0f;
            
            // Calcolo della somma dei quadrati delle differenze per ciascuna caratteristica
            for (int feature = 0; feature < n_features; ++feature) {
                float diff = X[tid * n_features + feature] - centers[cluster * n_features + feature];
                distance += diff * diff;
            }

            // Aggiornamento dell'etichetta se viene trovato un centro più vicino
            if (distance < min_distance) {
                min_distance = distance;
                min_label = cluster;
            }
        }

        // Assegnazione del punto al cluster corrispondente
        labels[tid] = min_label;
    }
}

  // Aggiornamento dei centroidi
  __global__ void update_centers_kernel(float* X, int* labels, int n_points, int n_clusters, int n_features, float* new_centers) {
    // Calcolo dell'ID del blocco
    int cluster = blockIdx.x * blockDim.x + threadIdx.x;

    // Verifica che il blocco sia all'interno del range dei cluster
    if (cluster < n_clusters) {
        for (int feature = 0; feature < n_features; ++feature) {
            float sum = 0.0f;
            int count = 0;

            // Calcolo della somma delle caratteristiche per i punti assegnati al cluster
            for (int point = 0; point < n_points; ++point) {
                if (labels[point] == cluster) {
                    sum += X[point * n_features + feature];
                    count++;
                }
            }

            // Calcolo del nuovo centro come la media
            if (count > 0) {
                new_centers[cluster * n_features + feature] = sum / count;
            }
        }
    }
}
"""

def kmeans_parallel(X, n_clusters, max_iter=200):
    # Compila i kernel CUDA
    mod = SourceModule(cuda_kernel)
    kmeans_kernel_func = mod.get_function("kmeans_kernel")
    update_centers_kernel_func = mod.get_function("update_centers_kernel")

    n_points, n_features = X.shape

    # Indice X casuale per inizializzare i centroidi
    random_state_par = np.random.RandomState(seed=42)
    random_indices = random_state_par.choice(n_points, n_clusters, replace=False)
    centers = X[random_indices]

    # Trasferimento dei dati e delle strutture GPU
    X_gpu = gpuarray.to_gpu(X.astype(np.float32))
    centers_gpu = gpuarray.to_gpu(centers.astype(np.float32))
    labels_gpu = gpuarray.empty(n_points, dtype=np.int32)

    # Calcolo delle dimensioni del blocco e della griglia
    block = (256, 1, 1)
    grid = ((n_points + block[0] - 1) // block[0], 1)

    for _ in range(max_iter):
        # Esecuzione del kernel CUDA
        kmeans_kernel_func(X_gpu, centers_gpu, labels_gpu, np.int32(n_points), np.int32(n_clusters), np.int32(n_features), block=block, grid=grid)

        # Trasferimento dei risultati dalla GPU alla CPU
        labels = labels_gpu.get()

        # Allocazione della memoria per i nuovi centroidi sul dispositivo
        new_centers_gpu = gpuarray.zeros((n_clusters, n_features), dtype=np.float32)

        # Esecuzione del kernel CUDA per aggiornare i centroidi
        update_centers_kernel_func(X_gpu, labels_gpu, np.int32(n_points), np.int32(n_clusters), np.int32(n_features), new_centers_gpu, block=block, grid=grid)

        # Trasferimento dei risultati dalla GPU alla CPU
        new_centers = new_centers_gpu.get()

        # Verifica della convergenza confrontando i centroidi vecchi e nuovi
        if np.all(centers == new_centers):
            break

        centers = new_centers

    # Liberazione della memoria della GPU
    X_gpu.gpudata.free()
    centers_gpu.gpudata.free()
    labels_gpu.gpudata.free()
    new_centers_gpu.gpudata.free()

    # Calcolo dell'inerzia dopo la convergenza
    inertia = sum(np.min(pairwise_distances(X, centers), axis=1))
    return centers, labels, inertia

# Generazione di dati
X, _ = make_blobs(n_samples=100000, centers=5, random_state=2)

# Esecuzione dell'algoritmo sequenziale K-means
start_time = time.time()
centers_seq, labels_seq, inertia_seq = kmeans_sequential(X, n_clusters=5)
end_time = time.time()
seq_time = end_time - start_time

# Esecuzione dell'algoritmo parallelo K-means
start_time = time.time()
centers_par, labels_par, inertia_par = kmeans_parallel(X, n_clusters=5)
end_time = time.time()
par_time = end_time - start_time

# Confronto delle prestazioni
print("Sequential Time:", seq_time)
print("Parallel Time:", par_time)

speedup = seq_time / par_time
print("Speedup:", speedup)

# Plot dei risultati
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=labels_seq)
plt.scatter(centers_seq[:, 0], centers_seq[:, 1], marker='x', color='red', s=200, label='Centers')
plt.title("Sequential K-means")
plt.legend()

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=labels_par)
plt.scatter(centers_par[:, 0], centers_par[:, 1], marker='x', color='red', s=200, label='Centers')
plt.title("Parallel K-means")
plt.legend()

plt.show()