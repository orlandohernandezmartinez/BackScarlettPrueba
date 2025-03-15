from flask import Flask, request, jsonify, url_for
from openai import OpenAI
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

conversation_history = []

def generate_gpt_response(history):
    print("✅ Generando respuesta GPT...")

    system_prompt = {
        "role": "system",
        "content": (
            "Eres AVA, la primer agente virtual de la Secretaría de Agricultura y Desarrollo Rural especializado..."
        )
    }

    messages = [system_prompt] + history[-10:]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            timeout=10  # Evitar que se cuelgue
        )

        full_response = response.choices[0].message.content
        print("✅ Respuesta GPT generada")

        return full_response

    except Exception as e:
        print(f"❌ Error en GPT: {str(e)}")
        return "Lo siento, no puedo responder ahora."

def eleven_labs_text_to_speech(text):
    print("✅ Generando audio en ElevenLabs...")

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

            print(f"✅ Audio listo en: {audio_url}")
            return audio_url

        else:
            print(f"❌ Error en ElevenLabs: {resp.status_code} {resp.text}")
            return None

    except Exception as e:
        print(f"❌ Error ElevenLabs: {str(e)}")
        return None

@app.route("/gpt-tts", methods=["POST"])
def gpt_tts_endpoint():
    global conversation_history

    print("✅ Nueva solicitud recibida")
    data = request.get_json()

    user_text = data.get("message", "").strip()

    if not user_text:
        return jsonify({"error": "No se proporcionó texto en 'message'."}), 400

    conversation_history.append({"role": "user", "content": user_text})

    gpt_response = generate_gpt_response(conversation_history)

    conversation_history.append({"role": "assistant", "content": gpt_response})

    audio_url = eleven_labs_text_to_speech(gpt_response)

    return jsonify({
        "response": gpt_response,
        "audio_url": audio_url
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)