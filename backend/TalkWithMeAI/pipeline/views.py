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


load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    print("Authentification avec le token Hugging Face...")
    login(HF_TOKEN)
else:
    print("Aucun token HF_TOKEN trouvé dans les variables d'environnement")


os.environ["TOKENIZERS_PARALLELISM"] = "false"


if torch.cuda.is_available():
    current_device = "cuda"
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    current_device = "cpu"
    print("Utilisation du CPU")


TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


tts_cache = {}


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
            
            print("Chargement du modèle Whisper large...")
            start_time = time.time()
            stt_model = whisper.load_model("medium", device=current_device)
            print(f"Modèle STT chargé en {time.time() - start_time:.2f} secondes")
            
            
            print("Chargement du modèle de génération de texte français...")
            start_time = time.time()
            
            
            model_name = "bigscience/bloomz-1b1"  
            
            
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


def generate_response(transcription, max_length=100):
    
    prompt = f"""
<instructions>
Tu es Louis Le Foufou, un assistant vocal intelligent qui répond aux questions en français de manière concise mais complète.
</instructions>

<question>
{transcription}
</question>

<réponse>
"""
    
    
    encoded_input = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = encoded_input.input_ids.to(text_model.device)
    attention_mask = encoded_input.attention_mask.to(text_model.device)
    
    
    with torch.inference_mode():
        outputs = text_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=len(input_ids[0]) + max_length,
            min_length=len(input_ids[0]) + 10,  
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    
    if "<réponse>" in generated_text:
        response = generated_text.split("<réponse>")[1].strip()
        
        response = response.split("</réponse>")[0].strip() if "</réponse>" in response else response
    else:
        
        response = generated_text.replace(prompt.strip(), "").strip()
    
    
    if len(response) < 10:
        
        alt_prompt = f"Question: {transcription}\nRéponse: "
        
        encoded_input = tokenizer(alt_prompt, return_tensors="pt", padding=True)
        input_ids = encoded_input.input_ids.to(text_model.device)
        attention_mask = encoded_input.attention_mask.to(text_model.device)
        
        with torch.inference_mode():
            outputs = text_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=len(input_ids[0]) + max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.9,  
                top_p=0.95,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
        
        alt_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Réponse:" in alt_text:
            response = alt_text.split("Réponse:")[1].strip()
        else:
            response = alt_text.replace(alt_prompt, "").strip()
    
    return response


@api_view(['POST'])
def process_audio(request):
    global models_loaded
    
    
    if not models_loaded:
        success = load_models()
        if not success:
            return Response({"error": "Erreur lors du chargement des modèles"}, status=500)
    
    
    audio_file = request.FILES.get('audio')
    if not audio_file:
        return Response({"error": "No audio file provided"}, status=400)

    
    file_id = int(time.time() * 1000)
    file_path = os.path.join(TEMP_DIR, f"input_{file_id}.webm")
    
    
    with open(file_path, "wb") as f:
        for chunk in audio_file.chunks():
            f.write(chunk)

    if not os.path.exists(file_path):
        return Response({"error": "Erreur lors de l'enregistrement du fichier"}, status=500)

    try:
        
        start_time = time.time()
        transcription_result = stt_model.transcribe(
            file_path,
            fp16=(current_device == "cuda"),
            language="fr"
        )
        transcription = transcription_result["text"].strip()
        transcription_time = time.time() - start_time
        print(f"Transcription effectuée en {transcription_time:.2f} secondes: '{transcription}'")

        
        start_time = time.time()
        ai_response = generate_response(transcription)
        generation_time = time.time() - start_time
        print(f"Réponse générée en {generation_time:.2f} secondes: '{ai_response}'")

        
        start_time = time.time()
        output_audio_path = os.path.join(TEMP_DIR, f"output_{file_id}.mp3")
        
        
        cache_key = ai_response.lower().strip()
        if cache_key in tts_cache and os.path.exists(tts_cache[cache_key]):
            
            with open(tts_cache[cache_key], "rb") as src, open(output_audio_path, "wb") as dst:
                dst.write(src.read())
            print(f"TTS récupéré du cache ({time.time() - start_time:.2f} secondes)")
        else:
            
            asyncio.run(edge_tts.Communicate(ai_response, "fr-FR-HenriNeural").save(output_audio_path))
            
            
            if len(tts_cache) > 100:
                old_key = next(iter(tts_cache))
                old_path = tts_cache.pop(old_key)
                if os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except:
                        pass
            
            
            cache_path = os.path.join(TEMP_DIR, f"cache_{len(tts_cache)}.mp3")
            with open(output_audio_path, "rb") as src, open(cache_path, "wb") as dst:
                dst.write(src.read())
            tts_cache[cache_key] = cache_path
            
        tts_time = time.time() - start_time
        print(f"TTS généré en {tts_time:.2f} secondes")
        
        
        total_time = transcription_time + generation_time + tts_time
        print(f"Temps total de traitement: {total_time:.2f} secondes")

        
        if current_device == "cuda":
            torch.cuda.empty_cache()

        if not os.path.exists(output_audio_path):
            return Response({"error": "Fichier TTS non généré"}, status=500)

        
        try:
            os.remove(file_path)
        except:
            pass
        
        
        response = FileResponse(open(output_audio_path, "rb"), content_type="audio/mpeg")
        response['Content-Disposition'] = 'attachment; filename="response.mp3"'
        
        
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
        
        
        if current_device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            
        return Response({
            "error": str(e),
            "details": traceback.format_exc()
        }, status=500)


load_models()