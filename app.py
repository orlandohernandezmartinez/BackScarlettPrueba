from flask import Flask, request, jsonify, url_for
import openai
import requests
import time
import os
from dotenv import load_dotenv

app = Flask(__name__)

# ===== Cargar variables de entorno =====
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
    print("✅ Entrando en generate_gpt_response()")

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

    messages = [system_prompt] + history[-10:]

    try:
        response_stream = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )

        full_response = ""
        for chunk in response_stream:
            if "content" in chunk.choices[0].delta:
                full_response += chunk.choices[0].delta["content"]

        print(f"✅ Respuesta completa generada: {full_response[:100]}...")  # Muestra los primeros 100 chars
        return full_response

    except Exception as e:
        print(f"❌ Error en generate_gpt_response(): {str(e)}")
        return "Lo siento, hubo un error al generar la respuesta."


# ===== Convertir texto a voz con ElevenLabs =====
def eleven_labs_text_to_speech(text):
    """
    Llama a la API de Eleven Labs para generar la voz.
    Guarda el MP3 en 'static/output_audio.mp3'.
    Devuelve la URL pública para acceder al MP3.
    """
    print("✅ Entrando en eleven_labs_text_to_speech()")

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
            os.makedirs("static", exist_ok=True)
            audio_file_path = os.path.join("static", "output_audio.mp3")
            with open(audio_file_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            timestamp = int(time.time())
            audio_url = url_for("static", filename="output_audio.mp3", _external=True) + f"?t={timestamp}"
            print(f"✅ Audio generado y guardado: {audio_url}")
            return audio_url
        else:
            print(f"❌ Error en ElevenLabs: {resp.status_code} => {resp.text}")
            return None
    except Exception as e:
        print(f"❌ Error al conectar con Eleven Labs: {str(e)}")
        return None


# ===== Endpoint /gpt-tts =====
@app.route("/gpt-tts", methods=["POST"])
def gpt_tts_endpoint():
    print("✅ Nueva solicitud en /gpt-tts")
    global conversation_history

    try:
        data = request.get_json(force=True)
        print(f"👉 JSON recibido: {data}")

        user_text = data.get("message", "").strip()
        if not user_text:
            print("❌ No se proporcionó texto en 'message'")
            return jsonify({"error": "No se proporcionó texto en 'message'."}), 400

        # 1. Agregar mensaje del usuario al historial
        conversation_history.append({"role": "user", "content": user_text})

        # 2. Generar respuesta con GPT
        gpt_response = generate_gpt_response(conversation_history)

        # 3. Agregar respuesta de GPT al historial
        conversation_history.append({"role": "assistant", "content": gpt_response})

        # 4. Generar audio con Eleven Labs
        audio_url = eleven_labs_text_to_speech(gpt_response)

        if not audio_url:
            print("❌ No se pudo generar el audio.")
            return jsonify({"error": "No se pudo generar el audio."}), 500

        # 5. Responder
        print("✅ Respuesta enviada correctamente.")
        return jsonify({
            "response": gpt_response,
            "audio_url": audio_url
        })

    except Exception as e:
        print(f"❌ Error en gpt_tts_endpoint(): {str(e)}")
        return jsonify({
            "error": "Error interno en el servidor.",
            "details": str(e)
        }), 500


# ===== Main =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"✅ Iniciando app en 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)