import numpy as np
import cv2

def getfacecoords(imagepath):
    cascade_file_src = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascade_file_src)
    image = cv2.imread(imagepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the image :
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    cropIm = []
    # crop face
    if len(faces) >= 1:
        return faces[0]

def getface(pictureLoc):
    cascade_file_src = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascade_file_src)
    # load image on gray scale :
    image = pictureLoc
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the image :
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    cropIm = []
    # crop face
    if len(faces) >= 1:
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]
        for r in range(y, h + y):
            new = []
            for c in range(x, w + x):
                new.append(image[r][c])

            cropIm.append(new)
        cropIm = np.asarray(cropIm)
        cropIm = cv2.resize(cropIm, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
        cropIm = np.asarray(cropIm)
        return cropIm


def getMatch(picloc):
    import face_recognition

    # Load a sample picture and learn how to recognize it.
    bill_image = face_recognition.load_image_file("billnye.jpeg")
    bill_face_encoding = face_recognition.face_encodings(bill_image)[0]

    jane_image = face_recognition.load_image_file("janegoodall.jpeg")
    jane_face_encoding = face_recognition.face_encodings(jane_image)[0]

    neil_image = face_recognition.load_image_file("neiltyson.jpg")
    neil_face_encoding = face_recognition.face_encodings(neil_image)[0]

    sally_image = face_recognition.load_image_file("sallyride.jpg")
    sally_face_encoding = face_recognition.face_encodings(sally_image)[0]

    carl_image = face_recognition.load_image_file("sagan.jpg")
    carl_face_encoding = face_recognition.face_encodings(carl_image)[0]

    musk_image = face_recognition.load_image_file("elonmusk.jpg")
    musk_face_encoding = face_recognition.face_encodings(musk_image)[0]

    steve_image = face_recognition.load_image_file("jobsGood.jpg")
    steve_face_encoding = face_recognition.face_encodings(steve_image)[0]

    curie_image = face_recognition.load_image_file("curieGood.jpg")
    curie_face_encoding = face_recognition.face_encodings(curie_image)[0]

    michio_image = face_recognition.load_image_file("michiokaku.jpg")
    michio_face_encoding = face_recognition.face_encodings(michio_image)[0]

    susan_image = face_recognition.load_image_file("swGood.jpg")
    susan_face_encoding = face_recognition.face_encodings(susan_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        bill_face_encoding, jane_face_encoding, neil_face_encoding, sally_face_encoding, carl_face_encoding,
        musk_face_encoding, steve_face_encoding, curie_face_encoding, michio_face_encoding, susan_face_encoding
    ]

    print(known_face_encodings)

    unknown_image = face_recognition.load_image_file(picloc)
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    results = face_recognition.face_distance(known_face_encodings, unknown_encoding)

    def getBestIndex(w):
        rating = 0
        value = 0
        for i in range(len(w)):
            if w[i] < rating:
                rating = w[i]
                value = i
        return value

    guess = getBestIndex(results)
    if guess == 0:
        return ["billnye.jpeg", "Bill Nye"]
    elif guess == 5:
        return ["elonmusk.jpg", "Elon Musk"]
    elif guess == 1:
        return ["janegoodall.jpeg", "Jane Goodall"]
    elif guess == 8:
        return ["michiokaku.jpg", "Michio Kaku"]
    elif guess == 2:
        return ["neiltyson.jpg", "Neil Degrassi Tyson"]
    elif guess == 3:
        return ["sallyride.jpg", "Sally Ride"]
    elif guess == 6:
        return ["jobsGood.jpg", "Steve Jobs"]
    elif guess == 7:
        return ["curieGood.jpg", "Marie Curie"]
    elif guess == 4:
        return ["sagan.jpg", "Carl Sagan"]
    else:
        return ["swGood.jpg", "Susan Wojcicki"]


