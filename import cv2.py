import cv2.py

def load_image(image_path):
    return cv2.imread(image_path)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect_faces_haarcascades(gray_image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    return faces

def draw_faces(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image
def display_image(image):
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def face_detection_pipeline(image_path):
    # Stage 1: Load Image
    image = load_image(image_path)

    # Stage 2: Convert to Grayscale
    gray_image = convert_to_grayscale(image)

    # Stage 3: Detect Faces using Haarcascades
    faces = detect_faces_haarcascades(gray_image)

 # Stage 4: Draw Faces
    result_image = draw_faces(image.copy(), faces)

    # Stage 5: Display Result
    display_image(result_image)

# Replace 'path/to/your/image.jpg' with the actual path to your image file
image_path = 'path/to/your/image.jpg'

# Run the face detection pipeline
face_detection_pipeline(image_path)




