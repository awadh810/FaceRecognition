import cv2   # في معالجة الصور والفيديو وتحليلها
import numpy as np
import face_recognition
import os

# Path to the directory containing known person images
path = 'persons'

# List of image filenames in the 'path' directory
personsList = os.listdir(path)

# Empty lists to store image data, person names, and names extracted from filenames
images = []
classNames = []
namesList = []

# Function to find and encode faces in images
def findEncodeings(images):
    encodeList = []
    namesList = []
    for cl in personsList:
        curPersonn = cv2.imread(f'{path}/{cl}')
        images.append(curPersonn)
        classNames.append(os.path.splitext(cl)[0])

        img = curPersonn  # Assign current image to img within the loop
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        namesList.append(os.path.splitext(cl)[0])
    return encodeList, namesList

# Load encoded faces of known persons
encodeListKnown, namesListKnown = findEncodeings(images)

# Set up webcam capture
cap = cv2.VideoCapture(0)  # For webcam feed

# Optional: Disable audio if needed
# cap.set(cv2.CAP_PROP_AUDIO, False)

# Variable to track number of detected faces
faceCount = 0

while True:
    ret, img = cap.read()

    # Resize the image for performance optimization
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Experiment with different resizing ratios
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces in the resized image
    faceCurentFrame = face_recognition.face_locations(imgS)
    encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)

    # Update the number of detected faces
    faceCount = len(faceCurentFrame)

    # Iterate through each detected face and its location
    for (encodeface, faceLoc) in zip(encodeCurentFrame, faceCurentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)   # تحسب مسافة الاختلاف بين الترميزات و لترميز الحالي
        matchIndex = np.argmin(faceDis)   # يجاد أقرب ترميز معروف للترميز الحالي من حيث مسافة الاختلاف.

        # Define a threshold for matching accuracy
        threshold = 0.5

        # Check if there's a good enough match
        if matches[matchIndex]:
            name = namesListKnown[matchIndex]
            print(f"Match found for : {name}" , " >> Ratio of Matching is : " , faceDis[matchIndex])
            print(faceDis)

            # Calculate bounding box coordinates and area
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Adjust scaling factor as needed

            faceArea = (x2 - x1) * (y2 - y1)
            imageArea = img.shape[0] * img.shape[1]
            faceRatio = faceArea / imageArea  # Calculate face area ratio

            # Draw bounding box, name, and face ratio on the original image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Adjust colors and thickness

            y_name = y1 - 20  # Adjust name position
            color = (0, 0, 255)  # Adjust name color
            fontScale = 1  # Adjust font size
            org = (x1 + 6, y_name)  # Adjust name position relative to bounding box
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)  # Adjust name background shape
            cv2.putText(img, f"{name}", org,
                        cv2.FONT_HERSHEY_COMPLEX, fontScale, color, 2)
            cv2.putText(img, f"Ratio: {faceRatio:.2%}", (x1 + 6, y2 - 25),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f"Total Faces: {faceCount}", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the processed image with detected faces and information
    cv2.imshow('Face Recognition', img)

     # Wait for a key press
    cv2.waitKey(1)
