'''
    This file contains the scripting code for cropping CAFE images using haarcascade
'''
import cv2
import glob
import os

# input_file_path = 'sessions/6401/10541-happy_F-EA-14.jpg'
# output_directory = 'cropped_sessions'
# output_file_path = '10541-happy_F-EA-14-cropped.jpg'
oldpwd = os.getcwd()

# Recursively find all files under Data/
for filepath in glob.iglob("sessions/**/*.*", recursive=True):
    # Ignore non images
    if not filepath.endswith((".jpg")):
        continue

    # Open image and perform cropping
    with open(filepath, "r") as f:
        # image = f.read()
        img = cv2.imread(filepath)
        # img = image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 2)

        cropped_face = None
        detected = False
        for (x, y, w, h) in faces:
            if (w < 900 or detected):
                continue
            cv2.rectangle(img, (x, y), (x+w, y+h),
                        (0, 0, 255), 2)
            face = img[y:y + h, x:x + w]
            # cv2.imshow("face",faces)
            # cv2.imwrite(output_file_path, face)
            cropped_face = face
            detected = True
        # cv2.imshow('img', img)
        # cv2.waitKey()
    
    # Get the output file and folder path
    output_filepath = filepath.replace("sessions", "cropped_sessions")
    output_dir = os.path.dirname(output_filepath)
    # Ensure the folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Write output files
    # with open(output_filepath, "w") as f:
        # f.write(cropped_face)
    os.chdir(output_dir)    
    if (len(faces) > 1):
        print("multiple faces detected for", filepath)
    if (cropped_face is None):
        print("failed to detect face in", filepath)
    else:
        cv2.imwrite(os.path.basename(output_filepath).split('/')[-1], cropped_face)
    # print(os.listdir(os.getcwd()))
    os.chdir(oldpwd)





