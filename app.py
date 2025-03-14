from flask import Flask, request, jsonify, url_for
import openai
import requests
import time
import os

app = Flask(__name__)

# ===== Claves (OpenAI y ElevenLabs) =====
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
# ===== Historial de conversación =====
conversation_history = []

# ===== Función para generar respuesta GPT (streaming) con ChatCompletion =====
def generate_gpt_response(history):
    """
    Usa la interfaz openai>=1.0.0: 'openai.ChatCompletion.create(...)'
    con stream=True para obtener la respuesta en chunks.
    """

    # Prompt del sistema (tu personalidad de IA)
    system_prompt = {
        "role": "system",
        "content": (
            "Eres AVA, la primer agente virtual de la Secretaría de Agricultura y Desarrollo Rural "
            "especializado en la agroindustria y el desarrollo rural del estado de Puebla. Tu misión "
            "es responder de manera clara, confiable y oportuna las preguntas de las y los usuarios "
            "que buscan información sobre producción agrícola, pecuaria y pesquera, así como sobre "
            "indicadores económicos, sociales y geográficos del estado de Puebla. "
            "Responde de forma clara y concisa, en un estilo cercano, pero profesional."
        )
    }

    messages = [system_prompt] + history[-10:]  # toma solo las últimas 10 interacciones

    # Llamada con streaming:
    response_stream = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True
    )

    # Recorremos los chunks de la respuesta
    full_response = ""
    for chunk in response_stream:
        if "content" in chunk.choices[0].delta:
            full_response += chunk.choices[0].delta["content"]

    return full_response  # <-- Devuelve el texto completo al final


# ===== Convertir texto a voz con ElevenLabs =====
def eleven_labs_text_to_speech(text):
    """
    Llama a la API de Eleven Labs para generar la voz.
    Guarda el MP3 en 'static/output_audio.mp3'.
    Devuelve la URL pública para acceder al MP3.
    """
    voice_id = "5foAkxpX0K5wizIaF5vu"  # Reemplaza con tu Voice ID de ElevenLabs
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

    headers = {
        "Accept": "application/json",
        "xi-api-key": elevenlabs_api_key
    }

    data = {
        "text": text,
        "model_id": "eleven_multilingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=data, stream=True)
        if resp.status_code == 200:
            # Guardar el audio
            os.makedirs("static", exist_ok=True)  # Crea 'static/' si no existe
            audio_file_path = os.path.join("static", "output_audio.mp3")
            with open(audio_file_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            # Generar URL con timestamp para evitar caché
            timestamp = int(time.time())
            audio_url = url_for("static", filename="output_audio.mp3", _external=True) + f"?t={timestamp}"
            return audio_url
        else:
            print(f"Error en ElevenLabs: {resp.status_code} => {resp.text}")
            return None
    except Exception as e:
        print(f"Error al conectar con Eleven Labs: {str(e)}")
        return None


# ===== Endpoint /gpt-tts =====
@app.route("/gpt-tts", methods=["POST"])
def gpt_tts_endpoint():
    global conversation_history

    data = request.get_json()
    user_text = data.get("message", "").strip()

    if not user_text:
        return jsonify({"error": "No se proporcionó texto en 'message'."}), 400

    # 1. Agregar mensaje del usuario al historial
    conversation_history.append({"role": "user", "content": user_text})

    # 2. Generar respuesta con GPT (streaming)
    gpt_response = generate_gpt_response(conversation_history)

    # 3. Agregar la respuesta al historial
    conversation_history.append({"role": "assistant", "content": gpt_response})

    # 4. Generar el audio con Eleven Labs
    audio_url = eleven_labs_text_to_speech(gpt_response)

    # 5. Responder con texto + URL del audio
    return jsonify({
        "response": gpt_response,
        "audio_url": audio_url
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Puerto que asigna Railway
    app.run(host="0.0.0.0", port=port)        # Escucha en todas las interfaces