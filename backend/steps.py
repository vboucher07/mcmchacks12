import os


IMAGES_DIRECTORY = "./frontend/static/steps"
ARUCO_CODES = [2,
               1, 1, 1, 1,
               3, 3, 3, 3,
               2,
               3, 3, 3, 3,
               2,
               3, 3, 3, 3,
               2,
               4, 4, 4, 4]

index = 0
images = []
for image in sorted(os.listdir(IMAGES_DIRECTORY)):
    if "step" in image:
        image_path = os.path.join(IMAGES_DIRECTORY, image)
        images.append(image_path.replace("./frontend", ""))

def next(): 
    global index
    if index < len(images) - 1:
        index += 1

    return images[index], ARUCO_CODES[index]

def prev(): 
    global index
    if index > 0:
        index -= 1

    return images[index], ARUCO_CODES[index]
