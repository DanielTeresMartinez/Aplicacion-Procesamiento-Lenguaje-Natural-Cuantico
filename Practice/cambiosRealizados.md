# Pasos que debo de seguir para implementar SGD en `loss_f`

> Esta es la versión clásica de SGD la de MiniBatch se deja para un futuro.

Seleccionar 1 índice aleatorio de qc_data
Filtrar qc_data → 1 circuito
Filtrar label_vectors → solo esa muestra
Recalcular target_distances para esa muestra — aquí ten cuidado porque con 1 sola palabra pdist da un vector vacío (necesitas al menos 2 palabras para calcular distancia entre pares). Tendrás que pensar cómo manejar ese caso en calculate_custom_loss.
Llamar a forward_pass y calculate_custom_loss con los datos filtrados