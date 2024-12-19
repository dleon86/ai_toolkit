from tools.image_processing import resize_image, apply_filter
from tools.tts_stt import text_to_speech, speech_to_text

def process_and_announce(image_path: str, resized_path: str, filtered_path: str, text: str) -> None:
    """
    Resizes an image, applies a filter, and converts text to speech announcing the process.

    Parameters:
    - image_path (str): Path to the original image.
    - resized_path (str): Path to save the resized image.
    - filtered_path (str): Path to save the filtered image.
    - text (str): Text to convert to speech.
    """
    # Resize Image
    resize_image(image_path, resized_path, (800, 600))

    # Apply Filter
    apply_filter(resized_path, filtered_path, "BLUR")

    # Convert Text to Speech
    text_to_speech(text, "announcement.mp3")
    print("Process completed and announcement saved as announcement.mp3") 