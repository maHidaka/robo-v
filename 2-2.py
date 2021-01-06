import os
import cv2
import numpy as np

# python 3 --
# _file_などを使用しているためipynb形式での実行はサポートしない
# ディレクトリに2バイト文字が含まれる実行環境での正常動作は保証されない


# 画像比較
def compar(comparing_img):

    TARGET_FILE = 'target.png'
    IMG_DIR = os.path.abspath(os.path.dirname(__file__)) + '/images/'
    IMG_SIZE = (300, 300)
    target_img_path = IMG_DIR + TARGET_FILE
    target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.resize(target_img, IMG_SIZE)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # detector = cv2.ORB_create()
    detector = cv2.AKAZE_create()
    (target_kp, target_des) = detector.detectAndCompute(target_img, None)

    try:
        (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)
        matches = bf.match(target_des, comparing_des)
        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)
    except cv2.error:
        ret = 100000

    return ret


capture = cv2.VideoCapture(0)
capture.set(3, 320)
capture.set(4, 240)

# 顔認識用カスケードファイルの読み込み
face_cascade = cv2.CascadeClassifier(
    os.path.abspath(os.path.dirname(__file__)) + '\haarcascade_frontalface_default.xml')

while True:
    ret, img = capture.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    level = 0

    for (x, y, w, h) in faces:
        cv2.circle(img, (int(x+w/2), int(y+h/2)),
                   int(w/2), (0, 50, 250), 2)
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (300, 300))
        level = compar(face)
        level = str(level)
        img = cv2.putText(img, level, (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX,
                          1.0, (0, 50, 250), thickness=2)
        cv2.imshow('face', face)

    height = img.shape[0]
    width = img.shape[1]
    img = cv2.resize(img, (int(width*3.5), int(height*3.5)))
    cv2.imshow('img', img)

    # key Operation
    key = cv2.waitKey(10)
    if key == 27 or key == ord('q'):  # escまたはqキーで終了
        break
capture.release()
cv2.destroyAllWindows()
print("Exit")
