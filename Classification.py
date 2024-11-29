from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Charger un modèle pré-entraîné pour la classification des sentiments en français
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"  # Exemple pour le sentiment multilingue
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Créer un pipeline de classification
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Textes à classer
textes = [
    "J'adore ce produit, il est vraiment incroyable !",
    "Je suis très déçu par la qualité, c'est inadmissible.",
    "Le service client était correct mais pas exceptionnel.",
    "Une expérience fantastique, je recommande fortement !",
]

# Classification
resultats = classifier(textes)

# Mapper les labels pour une sortie plus intuitive
label_mapping = {
    "LABEL_0": "Très négatif",
    "LABEL_1": "Négatif",
    "LABEL_2": "Neutre",
    "LABEL_3": "Positif",
    "LABEL_4": "Très positif"
}

# Afficher les résultats
for texte, resultat in zip(textes, resultats):
    label = label_mapping.get(resultat['label'], resultat['label'])
    print(f"Texte : {texte}")
    print(f"Classe : {label} (Score : {resultat['score']:.2f})\n")
