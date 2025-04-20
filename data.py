import cv2
import os

def collect_images(output_dir, person_name, num_images=50):
    person_path = os.path.join(output_dir, person_name)
    os.makedirs(person_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to access the webcam. Please check your camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    count = 0

    print(f"[INFO] Starting image collection for {person_name}...")
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame not captured. Retrying...")
            continue

        cv2.imshow("Capturing - Press 'q' to quit", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("[INFO] Quitting image collection early.")
            break

        file_path = os.path.join(person_path, f"{count + 1}.jpg")
        cv2.imwrite(file_path, frame)
        print(f"[INFO] Saved {file_path}")
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Image collection complete for {person_name}.")

if __name__ == "__main__":
    person = input("Enter person's name: ").strip().lower()
    collect_images("faces", person)