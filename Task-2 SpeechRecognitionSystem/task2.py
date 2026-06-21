import speech_recognition as sr
# Create recognizer object
recognizer = sr.Recognizer()
try:
    # Access microphone
    with sr.Microphone() as source:
        print("Speak something...")

        # Adjust for background noise
        recognizer.adjust_for_ambient_noise(source)

        # Listen to audio
        audio = recognizer.listen(source)

        print("Recognizing...")

        # Convert speech to text
        text = recognizer.recognize_google(audio)

        print("\nYou said:")
        print(text)

except sr.UnknownValueError:
    print("Sorry, could not understand the audio.")
except sr.RequestError:
    print("Could not connect to Google Speech Recognition service.")
except Exception as e:
    print("Error:", e)
