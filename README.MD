# README - Backend Speech-to-Text & AI Assistant

## Description
Ce backend implémente un pipeline local de reconnaissance vocale (Speech-to-Text), génération de texte assistée par IA et synthèse vocale (Text-to-Speech). Il permet de convertir un fichier audio en texte, d'interpréter le contenu avec un modèle de génération de texte et de fournir une réponse sous forme d'audio généré par une voix de synthèse.

## Fonctionnalités principales
1. **Reconnaissance vocale (STT - Speech-to-Text)**
   - Utilisation du modèle **Whisper Medium** pour une transcription rapide et légère.
   - Fonctionne en local sur CPU ou GPU.

2. **Génération de réponse par IA**
   - Utilisation du modèle **bigscience/bloomz-1b1** pour générer des réponses intelligentes en français.
   - Optimisation pour fonctionner avec des ressources limitées.

3. **Synthèse vocale (TTS - Text-to-Speech)**
   - Génération de la réponse audio avec **Edge TTS**.
   - Mise en cache des réponses pour accélérer les traitements futurs.

## Workflow
1. Un fichier audio est envoyé via une requête API.
2. Le fichier est transcrit en texte avec **Whisper**.
3. La transcription est analysée et une réponse est générée par **bigscience/bloomz-1b1**.
4. La réponse est transformée en fichier audio avec **Edge TTS**.
5. L'audio généré est retourné à l'utilisateur sous forme de fichier WAV.

## Technologies utilisées
- **Django REST Framework** pour la gestion des requêtes API.
- **Whisper (OpenAI)** pour la reconnaissance vocale.
- **bigscience/bloomz-1b1 (Hugging Face)** pour la génération de texte en français.
- **Edge TTS** pour la synthèse vocale.
- **PyTorch** pour l'exécution des modèles IA.

## Lancement
1. Installer les dépendances avec `pip install -r requirements.txt`.
2. Définir les variables d'environnement (`.env`) pour le token Hugging Face.
3. Lancer le serveur Django avec `python manage.py runserver`.
4. Envoyer des requêtes API pour tester le pipeline.

## Remarque
Ce backend est conçu pour un usage **local** et n'envoie aucune donnée vers des services externes, garantissant ainsi confidentialité et rapidité d'exécution.

