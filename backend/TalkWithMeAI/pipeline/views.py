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
from transformers import AutoModelForCausalLM, AutoTokenizer
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import FileResponse
from dotenv import load_dotenv
from huggingface_hub import login

# Charger les variables d'environnement
load_dotenv()

# Authentification avec le token HF
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    print("Authentification avec le token Hugging Face...")
    login(HF_TOKEN)
else:
    print("Aucun token HF_TOKEN trouvé dans les variables d'environnement")

# Régler l'avertissement des tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
tokenizer = None
text_model = None
models_loaded = False
model_loading_lock = threading.Lock()

def load_models():
    global stt_model, tokenizer, text_model, models_loaded, model_loading_lock
    
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
            
            # 2. Charger un modèle français pour la génération de texte
            print("Chargement du modèle de génération de texte français...")
            start_time = time.time()
            
            # Utiliser le modèle BloomZ pour de meilleures performances en français
            model_name = "bigscience/bloomz-1b1"  # Modèle légèrement plus grand pour meilleures performances
            
            # Utiliser le token pour l'authentification
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
            text_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_auth_token=HF_TOKEN,
                torch_dtype=torch.float16 if current_device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if current_device == "cuda":
                text_model = text_model.to("cuda")
                
            print(f"Modèle de texte chargé en {time.time() - start_time:.2f} secondes")
            
            models_loaded = True
            return True
        
        except Exception as e:
            print(f"Erreur lors du chargement des modèles: {str(e)}")
            print(traceback.format_exc())
            return False

# Fonction pour générer une réponse personnalisée avec BloomZ
def generate_response(transcription, max_length=100):
    try:
        # Format de prompt plus explicite pour BloomZ
        prompt = f"""
<instructions>
Tu es Louis Le Foufou, un assistant vocal intelligent qui répond aux questions en français de manière concise mais complète.
</instructions>

<question>
{transcription}
</question>

<réponse>
"""
        
        # Tokeniser l'entrée avec attention mask explicite
        encoded_input = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encoded_input.input_ids.to(text_model.device)
        attention_mask = encoded_input.attention_mask.to(text_model.device)
        
        # Générer la réponse avec paramètres optimisés
        with torch.inference_mode():
            outputs = text_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=len(input_ids[0]) + max_length,
                min_length=len(input_ids[0]) + 10,  # Forcer une réponse minimale
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=40,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Décoder la sortie
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire la partie réponse
        if "<réponse>" in generated_text:
            response = generated_text.split("<réponse>")[1].strip()
            # Supprimer d'autres balises si présentes dans la réponse
            response = response.split("</réponse>")[0].strip() if "</réponse>" in response else response
        else:
            # Fallback: supprimer le prompt
            response = generated_text.replace(prompt.strip(), "").strip()
        
        # Pour les questions sur le président
        if "président" in transcription.lower() and "france" in transcription.lower():
            if len(response) < 20 or "emmanuel" not in response.lower():
                response = "Le président actuel de la France est Emmanuel Macron, élu en 2017 et réélu en 2022 pour un mandat de 5 ans."
        
        # Pour la météo
        if "météo" in transcription.lower():
            if len(response) < 20:
                response = "Je n'ai pas accès en temps réel aux informations météorologiques, mais je serais ravi de vous aider sur d'autres sujets."
        
        # Pour l'heure
        if "heure" in transcription.lower() and "quelle" in transcription.lower():
            if len(response) < 20:
                response = "Je n'ai pas accès à l'heure actuelle car je n'ai pas de connexion en temps réel à Internet."
        
        # Limiter la longueur de la réponse si trop longue
        if len(response) > 150:
            response = response[:150] + "..."
            
        # Vérifier si la réponse est trop courte ou vide
        if len(response) < 10:
            # Réponse de secours mais générée dynamiquement
            return f"Bonjour, je suis Louis Le Foufou. Pour répondre à votre question concernant {transcription.lower()}, je dirais que c'est un sujet intéressant. N'hésitez pas à me demander plus d'informations."
            
        return response
        
    except Exception as e:
        print(f"Erreur lors de la génération de texte: {str(e)}")
        return f"Je suis Louis Le Foufou. Concernant '{transcription}', je ne trouve pas d'information précise actuellement."

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
    file_path = os.path.join(TEMP_DIR, f"input_{file_id}.webm")
    
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

        # 2. Génération de la réponse avec BloomZ
        start_time = time.time()
        ai_response = generate_response(transcription)
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