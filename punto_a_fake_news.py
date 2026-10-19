from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

textos = [
    "El presidente anunció una nueva reforma educativa",
    "Descubren que la vacuna convierte a las personas en robots",
    "La NASA confirma el hallazgo de agua en Marte",
    "Científicos afirman que la Tierra es plana",
    "El ministerio de salud lanza campaña contra el dengue",
    "Celebridades usan crema milagrosa para rejuvenecer 30 años",
    "Se inaugura el nuevo hospital en la ciudad",
    "Estudio revela que comer chocolate cura el cáncer",
    "Gobierno aprueba ley de protección ambiental",
    "Investigadores aseguran que los teléfonos espían nuestros sueños",
]
etiquetas = ["real", "fake", "real", "fake", "real", "fake", "real", "fake", "real", "fake"]

X_train, X_test, y_train, y_test = train_test_split(
    textos, etiquetas, test_size=0.3, random_state=42, stratify=etiquetas
)

vec = CountVectorizer()
modelo = MultinomialNB()
modelo.fit(vec.fit_transform(X_train), y_train)

y_pred = modelo.predict(vec.transform(X_test))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Matriz de confusión (filas=real, columnas=predicho, orden fake/real):")
print(confusion_matrix(y_test, y_pred, labels=["fake", "real"]))
print(classification_report(y_test, y_pred, zero_division=0))

nuevas_noticias = [
    "Nuevo estudio demuestra que el café mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos",
]
for n, p in zip(nuevas_noticias, modelo.predict(vec.transform(nuevas_noticias))):
    print(n, "->", p)
