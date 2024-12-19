from gtts import gTTS
import speech_recognition as sr

def text_to_speech(text: str, output_path: str) -> None:
    """
    Converts text to speech and saves it to a file.

    Parameters:
    - text (str): The text to convert.
    - output_path (str): Path to save the audio file.
    """
    tts = gTTS(text)
    tts.save(output_path)
    print(f"Text-to-Speech conversion complete. Audio saved to {output_path}")

def speech_to_text(audio_path: str) -> str:
    """
    Transcribes speech from an audio file to text.

    Parameters:
    - audio_path (str): Path to the audio file.

    Returns:
    - str: Transcribed text.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"Transcription: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return "" 