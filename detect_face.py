import cv2
import dlib

# Cargar el detector de rostros de dlib
detector = dlib.get_frontal_face_detector()

# Cargar la imagen
img = cv2.imread("data/images/04.jpg")

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detectar los rostros en la imagen
faces = detector(gray)

# Dibujar rectángulos alrededor de los rostros detectados
for face in faces:
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rectángulo verde

# Redimensionar la imagen para que se ajuste a la pantalla
height, width = img.shape[:2]
new_width = 800  # Nuevo ancho deseado
new_height = int((new_width / width) * height)
resized_img = cv2.resize(img, (new_width, new_height))

# Crear una ventana que se pueda redimensionar
cv2.namedWindow("Rostros detectados", cv2.WINDOW_NORMAL)

# Mostrar la imagen con los rostros detectados
cv2.imshow("Rostros detectados", resized_img)

# Esperar hasta que se presione una tecla para cerrar la ventana
while True:
    key = cv2.waitKey(1)  # Espera por un poco, 1ms
    if (
        key == ord("q")
        or cv2.getWindowProperty("Rostros detectados", cv2.WND_PROP_VISIBLE) < 1
    ):
        break  # Romper el bucle si la ventana se cierra o si se presiona 'q'

# Cerrar todas las ventanas y terminar el proceso
cv2.destroyAllWindows()
