import numpy as np
import cv2


def nouvelle_image(markerIds, markerCorners, image):
    min_G = float('inf')
    max_D = 0

    min_x = float('inf')
    max_x = 0

    coordonnee = {}
    x = {}
    y = {}

    coin = 1

    aruco = {}
    for coin in range(len(markerIds)):
        aruco[coin + 1] = {}

    if len(markerIds) == 4:
        coin = 1

        ids = markerIds.flatten()

        for (markerCorner, markerID) in zip(markerCorners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            aruco[coin][1] = topRight
            aruco[coin][2] = bottomRight
            aruco[coin][3] = bottomLeft
            aruco[coin][4] = topLeft

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            coordonnee[coin] = cX + cY
            x[coin] = cX
            y[coin] = cY

            cv2.putText(image, str(markerID),
                        (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            coin += 1

        for coin in range(1, 5):
            if coordonnee[coin] < min_G:
                min_G = coordonnee[coin]
                bottom_Left = aruco[coin][1]

            if coordonnee[coin] > max_D:
                max_D = coordonnee[coin]
                top_Right = aruco[coin][3]

        for coin in range(1, 5):
            if x[coin] < min_x and coordonnee[coin] != min_G and coordonnee[coin] != max_D:
                min_x = x[coin]
                top_Left = aruco[coin][2]

            if x[coin] > max_x and coordonnee[coin] != max_D and coordonnee[coin] != min_G:
                max_x = x[coin]
                bottom_Right = aruco[coin][4]

        h=360
        w = 180

        points1 = np.float32([top_Left, top_Right, bottom_Left, bottom_Right])
        points2 = np.float32([[-10, -20], [w + 10, -20], [-10, h + 10], [w + 10, h + 10]])

        vecttrans = cv2.getPerspectiveTransform(points1, points2)

        finalimage = cv2.warpPerspective(image, vecttrans, (w, h))

        couleur(finalimage)
        cv2.waitKey(0)

    else:
        print("Il n'y a pas assez d'ArUco détectés.")

#fonction qui permet de determiner la couleur du panneau à partir de l'image transformée

def couleur(image):
    # Convertir l'image en HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir les plages de couleurs
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Créer un masque pour chaque couleur
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combiner les masques
    mask = mask_red + mask_green + mask_blue

    # Appliquer le masque à l'image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Trouver la couleur dominante
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    blue_pixels = cv2.countNonZero(mask_blue)

    max_pixels = max(red_pixels, green_pixels, blue_pixels)

    if max_pixels == red_pixels:
        print("Le panneau est rouge.")
    elif max_pixels == green_pixels:
        print("Le panneau est vert.")
    elif max_pixels == blue_pixels:
        print("Le panneau est bleu.")
    else:
        print("La couleur du panneau n'a pas pu être déterminée.")


def camera():
    # Ouvrir la caméra
    robot_camera = cv2.VideoCapture(0)
    if not robot_camera.isOpened():
        print("Impossible d'ouvrir la caméra du robot.")
        exit()

    # Attendre que la caméra soit prête

    # Attendre jusqu'à ce que 4 marqueurs ArUco soient détectés
    while True:
        ret, frame = robot_camera.read()
        cv2.imshow('Robot Camera', frame)
        
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            print("Erreur lors de la capture de l'image depuis la caméra du robot.")
            break

        # On définit le dictionnaire de travail
        # Détection des marqueurs ArUco dans l'image
        # Définition du dictionnaire ArUco
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(frame, dictionary)

        # Vérifier si 4 marqueurs ArUco ont été détectés
        if markerIds is not None and len(markerIds) == 4:
            # Continuer le traitement
            # Prendre la photo
            cv2.imwrite('image_capturee.png', frame)

            image = cv2.imread('image_capturee.png')
            robot_camera.release()

            # Afficher les marqueurs détectés
            #print("Marqueur ArUco ID:", markerIds[i])
            nouvelle_image(markerIds, markerCorners, image)
            break

    # Fermer la caméra
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    camera()

