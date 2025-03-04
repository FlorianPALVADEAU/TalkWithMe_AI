import torch
import whisper
import edge_tts
import asyncio
import os
import traceback
import subprocess
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
    login(HF_TOKEN)

try:
    subprocess.run(["ffmpeg", "-version"], check=True)
except FileNotFoundError:
    pass

current_device = "cuda" if torch.cuda.is_available() else "cpu"

whisper_device = "cuda" if torch.cuda.is_available() and torch.cuda.memory_allocated() < 6e9 else "cpu"

tokenizer, lm_model, stt_model = None, None, None

def load_models():
    global tokenizer, lm_model, stt_model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    if lm_model is None:
        lm_model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/zephyr-7b-beta", 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        lm_model = torch.compile(lm_model)
    if stt_model is None:
        stt_model = whisper.load_model("tiny").to(whisper_device)

TEMP_DIR = os.path.join(os.getcwd(), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

@api_view(['POST'])
def process_audio(request):
    load_models()

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
        transcription = stt_model.transcribe(file_path, fp16=False, language="fr")["text"]
        
        inputs = tokenizer(transcription, return_tensors="pt").to(current_device)
        inputs = inputs.to(lm_model.device)
        outputs = lm_model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        output_audio_path = os.path.join(TEMP_DIR, "output.mp3")

        def run_tts(text, output_path):
            asyncio.run(edge_tts.Communicate(text, "fr-FR-HenriNeural").save(output_path))

        tts_thread = threading.Thread(target=run_tts, args=(ai_response, output_audio_path))
        tts_thread.start()
        tts_thread.join()

        if not os.path.exists(output_audio_path):
            return Response({"error": "Fichier TTS non généré"}, status=500)

        os.remove(file_path)
        response = FileResponse(open(output_audio_path, "rb"), as_attachment=True)
        os.remove(output_audio_path)

        return response

    except Exception as e:
        torch.cuda.empty_cache()
        return Response({"error": str(e), "details": traceback.format_exc()}, status=500)