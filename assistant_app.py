import cv2
import speech_recognition as sr
import detector
import scene_describer
import currency_detector

from text_reader import read_text_from_scene, say


def describe_scene(frame):
    say("Analyzing the scene. This may take a moment...")
    
    description = scene_describer.generate_enhanced_caption(frame)
    say(description) 

def detect_currency(frame):
    say("Detecting currency. Please hold the note steady.")
    frame_with_label, label, confidence = currency_detector.detect_currency_in_frame(frame)
    say(f"I think this is a {label} rupee note with confidence {confidence:.0%}.")
    return frame_with_label

def detect_objects(frame):

    frame_with_objects, objects = detector.detect_objects_in_frame(frame)

    if objects:
        object_list = " and ".join(objects) if len(objects) <= 2 else ", ".join(objects[:-1]) + " and " + objects[-1]
        say(f"I saw {object_list} in front of you.")
    else:
        say("I did not see any objects with enough confidence.")
    
    return frame_with_objects  

def listen_for_commands():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("[INFO] Listening for a command...")
            r.pause_threshold = 1.0
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
    except sr.WaitTimeoutError:
        print("[WARN] No speech detected (timeout).")
        return None
    except Exception as e:
        print(f"[ERROR] Microphone error: {e}")
        return None

    try:
        command = r.recognize_google(audio, language='en-in').lower()
        print(f"[INFO] You said: {command}")
        return command
    except sr.UnknownValueError:
        print("[WARN] Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"[ERROR] Could not request results; {e}")
        return None


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    say("System online. Awaiting commands.")
    

    display_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame.")
            break

        if display_frame is None:
            cv2.imshow('Live Feed', frame)
        else:
            cv2.imshow('Live Feed', display_frame)
         
            display_frame = None

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'): 
       
            command = listen_for_commands()
            
            if command:
                if "describe the scene" in command:
                    describe_scene(frame)
                
                elif "read text" in command:
                   
                    say("Okay, reading text now.")
                    display_frame = read_text_from_scene(frame)
                
                elif "detect currency" in command:
                    say("Okay, detecting currency now.")
                    display_frame = detect_currency(frame)
                
                elif "detect objects" in command:
                    say("Okay, detecting objects now.")
                    display_frame = detect_objects(frame)
                
                else:
                    say("Sorry, I did not recognize that command.")
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

