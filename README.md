# Relazione sul Progetto: Implementazione Parallela di K-means tramite CUDA
# Introduzione
Il progetto si propone di implementare una versione parallela dell'algoritmo K-means utilizzando la tecnologia CUDA. L'obiettivo è confrontare le prestazioni dell'algoritmo rispetto alla sua controparte sequenziale, sfruttando la potenza di calcolo parallelo delle GPU. L'ambiente di sviluppo principale è stato "Google Colab" che permette di sfruttare le risorse di accelerazione grafica messe a disposizione.

# Descrizione dell'Algoritmo K-means
L'algoritmo K-means è un metodo di clustering che mira a suddividere un insieme di dati in gruppi omogenei, noti come cluster.

# Implementazione Sequenziale
La funzione sequenziale dell'algoritmo K-means è stata implementata utilizzando la libreria scikit-learn in Python.

# Implementazione Parallela con CUDA
L'implementazione parallela è stata realizzata utilizzando la libreria CUDA per sfruttare la potenza di calcolo delle GPU. Il codice CUDA è composto da due kernel principali:

# Kernel di Assegnazione dei Cluster 
(kmeans_kernel): calcola la distanza tra ciascun punto e tutti i centroidi.
Assegna il punto al cluster più vicino.

# Kernel di Aggiornamento dei Centroidi 
(update_centers_kernel): calcola la media delle caratteristiche per i punti assegnati a ciascun cluster.
Aggiorna il centroide di ciascun cluster con la media calcolata.

In entrambi i kernel, sono state adottate pratiche per evitare race conditions durante l'aggiornamento dei risultati.

# Risultati e Confronto
L'implementazione parallela di K-means mediante CUDA ha dimostrato un notevole miglioramento delle prestazioni rispetto alla controparte sequenziale. L'analisi dei risultati evidenzia uno speedup considerevole, con il tempo di esecuzione del codice parallelo costantemente prossimo a 0.08 secondi (per 100.000 punti).
Tuttavia è importante notare che il tempo di esecuzione della funzione sequenziale ottenuto con lo stesso numero di punti è un po' variabile, oscillando tra il secondo e i decimi di secondo.

