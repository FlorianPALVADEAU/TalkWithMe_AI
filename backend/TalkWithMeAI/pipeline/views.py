import torch
import whisper
import edge_tts
import os
import traceback
import subprocess
import asyncio
import time
import gc
import threading
import random
from transformers import pipeline
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import FileResponse
from dotenv import load_dotenv

load_dotenv()

# Vérifier disponibilité du GPU
if torch.cuda.is_available():
    current_device = "cuda"
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    current_device = "cpu"
    print("Utilisation du CPU")

# Créez un répertoire temporaire pour les fichiers
TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Cache pour les réponses TTS
tts_cache = {}

# Variables globales pour les modèles
stt_model = None
text_generator = None
models_loaded = False
model_loading_lock = threading.Lock()

def load_models():
    global stt_model, text_generator, models_loaded, model_loading_lock
    
    if models_loaded:
        return True
    
    with model_loading_lock:
        if models_loaded:
            return True
            
        try:
            # 1. Charger le modèle Whisper tiny - ultra léger
            print("Chargement du modèle Whisper tiny...")
            start_time = time.time()
            stt_model = whisper.load_model("tiny", device=current_device)
            print(f"Modèle STT chargé en {time.time() - start_time:.2f} secondes")
            
            # 2. Charger un modèle de génération de texte très léger
            print("Chargement du modèle de génération de texte léger...")
            start_time = time.time()
            
            # Utiliser un modèle DistilGPT2 très léger
            text_generator = pipeline(
                "text-generation",
                model="distilgpt2",
                device=0 if current_device == "cuda" else -1
            )
            
            print(f"Modèle de texte chargé en {time.time() - start_time:.2f} secondes")
            
            models_loaded = True
            return True
        
        except Exception as e:
            print(f"Erreur lors du chargement des modèles: {str(e)}")
            print(traceback.format_exc())
            return False

# Fonction pour trouver la meilleure réponse prédéfinie ou générer une nouvelle
def get_best_response(transcription):
    transcription_lower = transcription.lower()
  
    
    # Si le message contient un nom, générer une réponse personnalisée
    if "je m'appelle" in transcription_lower or "mon nom est" in transcription_lower:
        parts = transcription_lower.replace("je m'appelle", "").replace("mon nom est", "").strip().split()
        if parts:
            name = parts[0].capitalize()
            responses = [
                f"Ravi de vous rencontrer, {name}! Je suis Louis Le Foufou.",
                f"Enchanté, {name}! Comment puis-je vous aider aujourd'hui?",
                f"Bonjour {name}! C'est un plaisir de faire votre connaissance."
            ]
            return random.choice(responses)
    
    # Sinon, générer une réponse avec le modèle
    try:
        # Ajouter un préfixe pour guider la génération en français
        prompt = f"Réponse courte et amicale à: {transcription}\nLouis: "
        
        # Générer avec le modèle léger
        response = text_generator(
            prompt,
            max_length=30,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            no_repeat_ngram_size=2
        )[0]['generated_text']
        
        # Nettoyer la réponse
        response = response.replace(prompt, "").strip()
        
        # Assurer que la réponse est en français en ajoutant un préfixe français si nécessaire
        if not any(word in response.lower() for word in ["bonjour", "salut", "je", "suis", "louis", "merci", "bien"]):
            response = "Bonjour! Je suis Louis Le Foufou. " + response
            
        # Limiter la taille
        if len(response) > 100:
            response = response[:100] + "..."
            
        return response
        
    except Exception as e:
        print(f"Erreur lors de la génération de texte: {str(e)}")
        # Réponse de secours en cas d'erreur
        return "Bonjour! Je suis Louis Le Foufou. Comment puis-je vous aider aujourd'hui?"

# Endpoint API optimisé
@api_view(['POST'])
def process_audio(request):
    global models_loaded
    
    # Charger les modèles si nécessaire
    if not models_loaded:
        success = load_models()
        if not success:
            return Response({"error": "Erreur lors du chargement des modèles"}, status=500)
    
    # Récupérer le fichier audio
    audio_file = request.FILES.get('audio')
    if not audio_file:
        return Response({"error": "No audio file provided"}, status=400)

    # Générer un nom de fichier unique
    file_id = int(time.time() * 1000)
    file_path = os.path.join(TEMP_DIR, f"input_{file_id}.wav")
    
    # Sauvegarder le fichier audio
    with open(file_path, "wb") as f:
        for chunk in audio_file.chunks():
            f.write(chunk)

    if not os.path.exists(file_path):
        return Response({"error": "Erreur lors de l'enregistrement du fichier"}, status=500)

    try:
        # 1. Transcription audio → texte
        start_time = time.time()
        transcription_result = stt_model.transcribe(
            file_path,
            fp16=(current_device == "cuda"),
            language="fr"
        )
        transcription = transcription_result["text"].strip()
        transcription_time = time.time() - start_time
        print(f"Transcription effectuée en {transcription_time:.2f} secondes: '{transcription}'")

        # 2. Génération de la réponse
        start_time = time.time()
        ai_response = get_best_response(transcription)
        generation_time = time.time() - start_time
        print(f"Réponse générée en {generation_time:.2f} secondes: '{ai_response}'")

        # 3. Génération de l'audio (TTS)
        start_time = time.time()
        output_audio_path = os.path.join(TEMP_DIR, f"output_{file_id}.mp3")
        
        # Vérifier le cache TTS
        cache_key = ai_response.lower().strip()
        if cache_key in tts_cache and os.path.exists(tts_cache[cache_key]):
            # Copier depuis le cache
            with open(tts_cache[cache_key], "rb") as src, open(output_audio_path, "wb") as dst:
                dst.write(src.read())
            print(f"TTS récupéré du cache ({time.time() - start_time:.2f} secondes)")
        else:
            # Générer nouveau TTS
            asyncio.run(edge_tts.Communicate(ai_response, "fr-FR-HenriNeural").save(output_audio_path))
            
            # Gérer le cache
            if len(tts_cache) > 100:
                old_key = next(iter(tts_cache))
                old_path = tts_cache.pop(old_key)
                if os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except:
                        pass
            
            # Créer une copie permanente pour le cache
            cache_path = os.path.join(TEMP_DIR, f"cache_{len(tts_cache)}.mp3")
            with open(output_audio_path, "rb") as src, open(cache_path, "wb") as dst:
                dst.write(src.read())
            tts_cache[cache_key] = cache_path
            
        tts_time = time.time() - start_time
        print(f"TTS généré en {tts_time:.2f} secondes")
        
        # Temps total
        total_time = transcription_time + generation_time + tts_time
        print(f"Temps total de traitement: {total_time:.2f} secondes")

        # Libérer la mémoire
        if current_device == "cuda":
            torch.cuda.empty_cache()

        if not os.path.exists(output_audio_path):
            return Response({"error": "Fichier TTS non généré"}, status=500)

        # Nettoyer le fichier d'entrée
        try:
            os.remove(file_path)
        except:
            pass
        
        # Renvoyer le fichier audio
        response = FileResponse(open(output_audio_path, "rb"), content_type="audio/mpeg")
        response['Content-Disposition'] = 'attachment; filename="response.mp3"'
        
        # Supprimer le fichier après envoi
        def delete_later():
            try:
                time.sleep(5)
                if os.path.exists(output_audio_path):
                    os.remove(output_audio_path)
            except:
                pass
            
        threading.Thread(target=delete_later).start()
        
        return response

    except Exception as e:
        print(f"Erreur: {str(e)}")
        print(traceback.format_exc())
        
        # Libérer la mémoire GPU
        if current_device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            
        return Response({
            "error": str(e),
            "details": traceback.format_exc()
        }, status=500)

# Initialiser les modèles au démarrage
load_models()