import os
import uuid
import shutil
import subprocess
import numpy as np
import soundfile as sf
import noisereduce as nr
import matchering as mg
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pedalboard import Pedalboard, HighpassFilter, PeakFilter, Compressor, Reverb, Limiter, Gain

app = FastAPI(title="Playback Voice Tools IA Server")

STORAGE_DIR = "/tmp/voice_sessions"
os.makedirs(STORAGE_DIR, exist_ok=True)

def get_session_path(session_id: str) -> str:
    path = os.path.join(STORAGE_DIR, session_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    return path

@app.get("/")
def home():
    return {"status": "online", "service": "Playback Voice Tools 5.0 Backend"}

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    raw_path = os.path.join(session_dir, "step_0.wav")
    temp_input = os.path.join(session_dir, f"input_{file.filename}")
    
    with open(temp_input, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Padroniza qualquer arquivo enviado para WAV 48kHz Mono
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_input, "-ar", "48000", "-ac", "1", raw_path
    ], check=True)
    
    return {"session_id": session_id, "audio_url": f"/audio/{session_id}/step_0.wav"}

@app.post("/process/step/{step_num}/{session_id}")
def process_step(step_num: int, session_id: str):
    session_dir = get_session_path(session_id)
    
    # Pega o áudio da etapa anterior (ou o original se a anterior não foi rodada)
    prev_file = os.path.join(session_dir, f"step_{step_num - 1}.wav")
    if not os.path.exists(prev_file):
        prev_file = os.path.join(session_dir, "step_0.wav")
        
    out_file = os.path.join(session_dir, f"step_{step_num}.wav")
    audio, sr = sf.read(prev_file)

    if step_num == 1:
        # Etapa 1: Limpeza de Ruído de Fundo e Ambiente (De-Noise)
        reduced = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.75)
        sf.write(out_file, reduced, sr)

    elif step_num == 2:
        # Etapa 2: Corte de Graves e Vento (High-Pass 85Hz)
        board = Pedalboard([HighpassFilter(cutoff_frequency_hz=85.0)])
        processed = board(audio, sr)
        sf.write(out_file, processed, sr)

    elif step_num == 3:
        # Etapa 3: Corpo e Equalização Vocal (Tira som de caixa + dá presença)
        board = Pedalboard([
            PeakFilter(cutoff_frequency_hz=270.0, gain_db=-3.5, q=1.5),
            PeakFilter(cutoff_frequency_hz=3800.0, gain_db=4.0, q=1.2)
        ])
        processed = board(audio, sr)
        sf.write(out_file, processed, sr)

    elif step_num == 4:
        # Etapa 4: Controle de Sibilância (De-Esser na região de 6.5kHz)
        board = Pedalboard([
            PeakFilter(cutoff_frequency_hz=6500.0, gain_db=-4.5, q=2.5)
        ])
        processed = board(audio, sr)
        sf.write(out_file, processed, sr)

    elif step_num == 5:
        # Etapa 5: Nivelamento Dinâmico (Compressor Óptico + Ganho)
        board = Pedalboard([
            Compressor(threshold_db=-18.0, ratio=3.5, attack_ms=10.0, release_ms=120.0),
            Gain(gain_db=4.0)
        ])
        processed = board(audio, sr)
        sf.write(out_file, processed, sr)

    elif step_num == 6:
        # Etapa 6: Espaço e Ambiência (Plate Reverb de Estúdio)
        board = Pedalboard([
            Reverb(room_size=0.35, damping=0.6, wet_level=0.14, dry_level=0.86)
        ])
        processed = board(audio, sr)
        sf.write(out_file, processed, sr)

    elif step_num == 7:
        # Etapa 7: Masterização Final e Volume Comercial (-1.0 dBFS)
        board = Pedalboard([
            Gain(gain_db=3.0),
            Limiter(threshold_db=-1.0, release_ms=100.0)
        ])
        processed = board(audio, sr)
        sf.write(out_file, processed, sr)
    else:
        raise HTTPException(status_code=400, detail="Etapa inválida (use 1 a 7)")

    return {"step": step_num, "audio_url": f"/audio/{session_id}/step_{step_num}.wav"}

@app.post("/mixdown/{session_id}")
async def mixdown(session_id: str, playback: UploadFile = File(...), vocal_step: int = 7):
    session_dir = get_session_path(session_id)
    
    vocal_file = os.path.join(session_dir, f"step_{vocal_step}.wav")
    if not os.path.exists(vocal_file):
        vocal_file = os.path.join(session_dir, "step_0.wav")
        
    pb_raw = os.path.join(session_dir, f"pb_{playback.filename}")
    with open(pb_raw, "wb") as f:
        shutil.copyfileobj(playback.file, f)
        
    final_mix = os.path.join(session_dir, "mixdown_final.wav")
    
    # Converte o playback (MP3/OGG/WAV) e mistura com a voz tratada
    subprocess.run([
        "ffmpeg", "-y", "-i", pb_raw, "-i", vocal_file,
        "-filter_complex", "[0:a]volume=0.85[a0];[1:a]volume=1.15[a1];[a0][a1]amix=inputs=2:duration=longest",
        "-ar", "48000", final_mix
    ], check=True)
    
    return {"audio_url": f"/audio/{session_id}/mixdown_final.wav"}

@app.get("/audio/{session_id}/{filename}")
def get_audio(session_id: str, filename: str):
    file_path = os.path.join(STORAGE_DIR, session_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)
