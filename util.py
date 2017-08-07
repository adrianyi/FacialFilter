import numpy as np
import cv2 # OpenCV 3
#from PIL import Image

def load_image(image_path):
    '''
    Load image from file

    Load image file from file_dir, returns color_image and gray_image

    Args:
        image_path: A string of a path to an image
            (i.e. '/example_images/test_image_1.jpg')

    Returns:
        (color_image, color_gray)
        color_BGR: class numpy.ndarray with dtype 'uint8' and shape (h, w, 3)
            3 is for 3 BGR colors, each pixel is in range [0,255]
        color_gray: class numpy.ndarray with dtype 'uint8' and shape (h, w)
            each pixel is in range [0,255]
    '''
    image_BGR = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)

    return image_BGR, image_gray

def detect_cascade(image_gray, cascadeClassifier,
                    scaleFactor = 1.2,
                    minNeighbors = 6):
    '''
    Detects objects in an image

    Detects cascade objects in the image_gray, and returns (x,y,w,h)
    for all the objects detected.

    Args:
        image_gray: 2-dimensional numpy.ndarray
        cascade: string of a path to a cascade architecture
        scaleFactor (optional, default 1.2): float (>1) used for scaling factor
                in the cascade detection
        minNeighbors (optional, default 6): integer

        For good explanation of scaleFactor and minNeighbors, see
        http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php

    Returns:
        detected: list of (x,y,w,h) for all detected objects
                x,y are position coordinates
                w,h are width (in x-direction) and height (in y-direction)
    '''
    detected = cascadeClassifier.detectMultiScale(image_gray, scaleFactor, minNeighbors)
    return detected

def detect_facial_features(image_gray, faces, model):
    '''
    Args:
        image_gray: 2-dimensional np.ndarray
        faces: list of (x,y,w,h)
        model: Keras model (NOT directory)
    Returns:
        features_list: list of (f1x,f1y,f2x,f2y,...,f15x,f15y)
    '''
    faces_array = np.zeros(((len(faces),96,96,1)))
    for i, (x,y,w,h) in enumerate(faces):
        face = image_gray[y:y+h,x:x+w]
        face = cv2.resize(face/255, (96,96))
        face = np.reshape(face, (96,96,1))
        faces_array[i] = face
    features_list = model.predict(faces_array)
    return features_list

def add_sunglasses(image_BGR, faces, list_facialFeatures, sg_image='images/sunglasses.png'):
    '''
    Overlays sunglasses on faces

    Args:
        sg_image: string path to a sunglasses image
                (4-channel, where 4th dim is opacity)
    '''
    image_sg = np.copy(image_BGR)
    sunglasses = cv2.imread(sg_image, cv2.IMREAD_UNCHANGED)
    #mask = sunglasses[:,:,[3,3,3]]/255
    #mask_inv = 1-mask
    #sunglasses = sunglasses[:,:,:3]

    for face, facialFeatures in zip(faces,list_facialFeatures):
        (xmax, xmin, ymax, ymin) = extent_sunglasses(face, facialFeatures)
        sg = cv2.resize(sunglasses, (xmax-xmin,ymax-ymin))
        mask = sg[:,:,[3,3,3]]/255
        mask_inv = 1-mask
        image_sg[ymin:ymax,xmin:xmax] = np.multiply(sg[:,:,:3],mask) + \
                    np.multiply(image_sg[ymin:ymax,xmin:xmax],mask_inv)
    return image_sg

def extent_sunglasses(face, facialFeatures):
    '''
    Calculates extent (xmax, xmin, ymax, ymin) for sunglasses given
    a face and its facialFeatures
    '''
    # Right eyebrow (18, 19)
    # Left eyebrow (14, 15)
    # Outer point of right eye (6, 7)
    # Outer point of left eye (10, 11)
    # Tip of the nose (20, 21)
    x,y,w,h = face
    brow_rx, brow_lx, eye_rx, eye_lx = \
                        (1.3*facialFeatures[[18,14,6,10]] + 1) * w/2 + x
    brow_ry, brow_ly, eye_ry, eye_ly, nose_y = \
                        (facialFeatures[[19,15,7,11,21]] + 1) * h/2 + y
    xmin = np.int(min(brow_rx, eye_rx))
    xmax = np.int(max(brow_lx, eye_lx))
    ymin = np.int(min(brow_ly,brow_ry))
    ymax = np.int(nose_y)

    return (xmax, xmin, ymax, ymin)

def drawRects(image_BGR, objList, color_BGR=(0,0,255), thickness=3):
    '''
    Draws rectangles for all objects

    Given list of coordinates (x,y,w,h) in objList, draws rectangles with
    vertices (x,y), (x+w,y), (x,y+h), (x+w,y+h).

    Args:
        image_BGR: 3-dimensional numpy.ndarray (BGR is OpenCV's default)
        objList: list of (x,y,w,h) for all objects to draw
        color_BGR (optional, default (0,0,255): BGR color, tuple of 3 uint8
                                            (i.e. (0,0,255) is red)
        thickness (optional, default 3): pixel thickness for lines

    Returns:
        image_with_rects: 3-dimensional numpy.ndarray (BGR)
                with all the rectangles drawn in
    '''
    image_with_rects = np.copy(image_BGR)
    for x, y, w, h in objList:
        cv2.rectangle(image_with_rects, (x,y), (x+w,y+h), color_BGR, thickness)
    return image_with_rects

def plot_features(image_ndarray, coords, color_BGR=[0,255,0], thickness=2):
    '''
    Draws a dot for all coords

    Given list of coordinates (x,y) in coords, draws circles with
    center (x,y) and radius = (thickness+1)//2.

    Args:
        image_ndarray: numpy.ndarray (grayscale or BGR)
        coords: list of (x,y) for all facial features to draw
        color_BGR (optional, default (0,255,0): BGR color, tuple of 3 uint8
                                            (i.e. (0,0,255) is red)
        thickness (optional, default 3): pixel thickness for lines

    Returns:
        image_with_rects: 3-dimensional numpy.ndarray (BGR)
                with all the rectangles drawn in
    '''
    image = np.copy(image_ndarray)
    w,h = image.shape[:2]
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        pass
    else:
        raise TypeError("Input must be either a grayscale or BGR image")
    #undo normalization
    x_features, y_features = coords[0::2]*w/2+w/2, coords[1::2]*h/2+h/2

    for coord in zip(x_features,y_features):
        cv2.circle(image, coord, (thickness+1)//2, color_BGR, thickness)
    return image
