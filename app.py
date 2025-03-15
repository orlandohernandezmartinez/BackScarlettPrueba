import os
import time
import requests
import openai
from dotenv import load_dotenv
from flask import Flask, request, jsonify, url_for

load_dotenv()

app = Flask(__name__)

# Configura la API key de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

conversation_history = []

@app.route("/ping", methods=["GET"])
def ping():
    """Endpoint de prueba para verificar que la app responde sin usar OpenAI/ElevenLabs."""
    print("🔎 /ping endpoint reached!")
    return jsonify({"message": "pong"}), 200

def generate_gpt_response(history):
    print("🌀 Entrando a generate_gpt_response...")
    system_prompt = {
        "role": "system",
        "content": (
            "Eres AVA, la primer agente virtual de la Secretaría de Agricultura y Desarrollo Rural "
            "especializado en la agroindustria y el desarrollo rural del estado de Puebla. Tu misión "
            "es responder de manera clara, confiable y oportuna las preguntas de las y los usuarios "
            "que buscan información sobre producción agrícola, pecuaria y pesquera, así como sobre "
            "indicadores económicos, sociales y geográficos del estado de Puebla."
        )
    }

    messages = [system_prompt] + history[-10:]
    print(f"🌀 Mensajes enviados a ChatCompletion: {messages}")

    try:
        print("🌀 Llamando a openai.ChatCompletion.create...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        print("🌀 Respuesta recibida de OpenAI.")
        full_response = response.choices[0].message.content
        print(f"🌀 Respuesta GPT: {full_response}")
        return full_response
    except Exception as e:
        print(f"❌ Error en generate_gpt_response: {e}")
        raise e  # Propagamos el error para que se capture en el endpoint

def eleven_labs_text_to_speech(text):
    print("🎤 Entrando a eleven_labs_text_to_speech...")
    voice_id = "5foAkxpX0K5wizIaF5vu"
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

    print("🎤 Realizando POST a ElevenLabs...")
    try:
        resp = requests.post(url, headers=headers, json=data, stream=True, timeout=30)
        print(f"🎤 Respuesta de ElevenLabs: status_code={resp.status_code}")

        if resp.status_code == 200:
            print("🎤 Convirtiendo respuesta a archivo .mp3...")
            os.makedirs("static", exist_ok=True)
            audio_file_path = os.path.join("static", "output_audio.mp3")
            with open(audio_file_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            timestamp = int(time.time())
            audio_url = url_for("static", filename="output_audio.mp3", _external=True) + f"?t={timestamp}"
            print(f"🎤 Audio disponible en: {audio_url}")
            return audio_url
        else:
            print(f"❌ Error ElevenLabs: {resp.status_code} => {resp.text}")
            return None
    except Exception as e:
        print(f"❌ Error en eleven_labs_text_to_speech: {str(e)}")
        return None

@app.route("/gpt-tts", methods=["POST"])
def gpt_tts_endpoint():
    print("🚀 Entrando al endpoint /gpt-tts...")
    global conversation_history
    try:
        data = request.get_json()
        print(f"✅ Mensaje recibido: {data}")

        user_text = data.get("message", "").strip()
        print(f"🔎 user_text = {user_text}")

        if not user_text:
            print("⚠️ user_text vacío. Devolviendo error 400.")
            return jsonify({"error": "No se proporcionó texto en 'message'."}), 400

        print("🗨️ Añadiendo mensaje del usuario al historial.")
        conversation_history.append({"role": "user", "content": user_text})

        print("🧠 Generando respuesta de GPT...")
        gpt_response = generate_gpt_response(conversation_history)
        print(f"🧩 Respuesta GPT generada: {gpt_response}")

        conversation_history.append({"role": "assistant", "content": gpt_response})

        print("🎤 Convirtiendo texto a voz con ElevenLabs...")
        audio_url = eleven_labs_text_to_speech(gpt_response)
        print(f"🎤 URL de audio: {audio_url}")

        print("✅ Respuesta final lista para enviar.")
        return jsonify({
            "response": gpt_response,
            "audio_url": audio_url
        })

    except Exception as e:
        print(f"❌ Error en endpoint /gpt-tts: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🔧 Iniciando Flask en el puerto {port}")
    app.run(host="0.0.0.0", port=port)