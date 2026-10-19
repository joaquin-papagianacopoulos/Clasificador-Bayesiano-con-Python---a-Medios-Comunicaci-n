Clasificador Naive Bayes de noticias falsas (fake) o reales (real).

Hecho por Joaquin Papagianacopoulos

## Qué hace
- Convierte los textos en conteo de palabras (CountVectorizer).
- Entrena un MultinomialNB con 7 noticias y prueba con 3 (split 70/30).
- Muestra accuracy, matriz de confusión y reporte de clasificación.
- Clasifica las 2 noticias nuevas de la consigna.

## Cómo correrlo
```
pip install scikit-learn
python punto_a_fake_news.py
```

Con tan pocos datos (10 noticias) las métricas son poco confiables.
