import torch
import whisper
import edge_tts
import asyncio
import os
import traceback
import subprocess

from transformers import AutoModelForCausalLM, AutoTokenizer
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import FileResponse
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("⚠️ Aucune clé Hugging Face trouvée. Ajoute-la dans ton .env !")
else:
    login(HF_TOKEN)

try:
    subprocess.run(["ffmpeg", "-version"], check=True)
    print("✅ FFMPEG est installé")
except FileNotFoundError:
    print("❌ FFMPEG n'est pas installé. Vérifie ton installation.")

current_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation de {current_device}")

tokenizer, lm_model, stt_model = None, None, None

def load_models():
    global tokenizer, lm_model, stt_model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", trust_remote_code=True)
    if lm_model is None:
        lm_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    if stt_model is None:
        stt_model = whisper.load_model("tiny").to(current_device)

TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

@api_view(['POST'])
def process_audio(request):
    load_models()  # Chargement des modèles si nécessaire

    audio_file = request.FILES.get('audio')
    if not audio_file:
        return Response({"error": "No audio file provided"}, status=400)

    file_path = os.path.join(TEMP_DIR, "temp_audio.wav")
    with open(file_path, "wb") as f:
        for chunk in audio_file.chunks():
            f.write(chunk)

    if not os.path.exists(file_path):
        return Response({"error": f"Erreur lors de l'enregistrement du fichier : {file_path}"}, status=500)

    try:
        # Transcription
        transcription = stt_model.transcribe(file_path)["text"]
        print(f"🔊 Transcription : {transcription}")

        # Response generation
        inputs = tokenizer(transcription, return_tensors="pt").to(current_device)
        outputs = lm_model.generate(**inputs, max_new_tokens=min(150, len(transcription.split()) * 3))
        ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Réponse AI : {ai_response}")

        # Audio generation
        output_audio_path = os.path.join(TEMP_DIR, "output.mp3")

        async def generate_tts():
            tts = edge_tts.Communicate(ai_response, "fr-FR-DeniseNeural")
            await tts.save(output_audio_path)

        asyncio.run(generate_tts())

        if not os.path.exists(output_audio_path):
            return Response({"error": "Fichier TTS non généré"}, status=500)

        os.remove(file_path)

        response = FileResponse(open(output_audio_path, "rb"), as_attachment=True)

        os.remove(output_audio_path)

        return response

    except Exception as e:
        torch.cuda.empty_cache()
        return Response({"error": str(e), "details": traceback.format_exc()}, status=500)
