import cv2
import dlib

detector = dlib.get_frontal_face_detector()

img = cv2.imread("data/images/04.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = detector(gray)

for face in faces:
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green rectangle

height, width = img.shape[:2]
new_width = 800  # desired width
new_height = int((new_width / width) * height)
resized_img = cv2.resize(img, (new_width, new_height))

cv2.namedWindow("Detected faces", cv2.WINDOW_NORMAL)

cv2.imshow("Detected faces", resized_img)

while True:
    key = cv2.waitKey(1)  # wait 1ms
    if (
        key == ord("q")
        or cv2.getWindowProperty("Detected faces", cv2.WND_PROP_VISIBLE) < 1
    ):
        break

cv2.destroyAllWindows()
