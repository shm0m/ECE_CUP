import cv2
import numpy as np


def chiffre (img) :
    #créons un dictionnaire de correspondance entre état des segement set valeur lue
    correspondances = {
        (1, 1, 1, 0, 1, 1, 1): 0, # pas retourné
        (0, 0, 1, 0, 0, 1, 0): 1, # pas retourné
        (1, 0, 1, 1, 1, 0, 1): 5, # retourné
        (1, 0, 1, 1, 0, 1, 1): 3, # pas retourné
        (0, 0, 1, 1, 1, 1, 0): 4, # retourné
        (1, 1, 0, 1, 0, 1, 1): 2, # retourné
        (1, 1, 1, 1, 1, 0, 1): 6, # retourné
        (0, 0, 1, 0, 0, 1, 1): 7, # retourné
        (1, 1, 1, 1, 1, 1, 1): 8, # pas retourné
        (1, 0, 1, 1, 1, 1, 1): 9, # retourné
        #faire de même pour chaque chiffre
#la numérotation des segments est donné par la liste "segments"
#ci dessous et représentée dans la présentation du cours
    }

    (H, L) = img.shape[:2] #récupérer la hauteur et largeur de l'image

    # définir 7 zones d'intérêts, une par segment
    segments = [
#complétez les positions des rengles définissants les zones d'intérêts
#on propose d'utiliser une grille 6x4 sauf pour le centre
#vous pouvez tout à fait choisir une autre manière de zoner
#la position dans la liste des segments est dépendante du dictionnaire ci-dessus
#complétez ci dessous, vous pouvez insérer cv2.imshow('segments', image) ci-deesous pour vous aider à postionner vos rectangles correctement
            ((int(L/4), int(H/6)), (int(L*3/4), int(H*2/6))), # haut
            ((0, int(H/6)), (int(L/4), int(H*3/6))), # haut-gauche
            ((int(3*L/4), int(H/6)), (int(L), int(H*3/6))), # haut-droite
            ((int(L/4), int(H*2/5)) , (int(L*3/4), int(H*3/5))), # centre
            ((0, int(H*3/6)) , (int(L/4), int(H*5/6))), # bas-gauche
            ((int(L*3/4), int(H*3/6)) , (int(L), int(H*5/6))), # bas-droite
            ((int(L/4), int(H*4/6)) , (int(L*3/4), int(H*5/6))) # bas
        ]

    #optionnel : dessiner des rectangles autour des zones d'intérêts
    for rect in segments:
        color=tuple(np.random.random(size=3) * 256)
        img=cv2.rectangle(img, rect[0], rect[1],color, 3)

    #passer en niveau de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #définir un seuil et binariser l'image
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    #nettoyer l'image avec un morphose (optionnel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    #créer un tableau des états vus des segments (1 le segment est noir, 0 le segment est blanc)
    on = [0] * len(segments)

    #voir l'état pour chaque segment :
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extraire une image binaire de la zone d'intérêt correspondant au segment
        segROI = thresh[yA:yB, xA:xB]
        
        #compter le nombre de pixel noir
        nbpixels = cv2.countNonZero(segROI)
        
        print("segment "+str(i)+" a "+ str(nbpixels) +" pixels non blancs")
        
        #calculer l'aire du segment
        area = (xB - xA) * (yB - yA)
        
        # modifier le tableau d'état vu si le nombre de pixels noir dépasse 30% (à affiner en fonction de vos caméras)
        if nbpixels / float(area) > 0.3:
            on[i]= 1

    print(on) #optionnel

    #optionnel : afficher une croix au centre des segments identifiés
    for i in range(len(on)):
        if on[i]==1:
            milsegement=(int((segments[i][0][0]+segments[i][1][0])/2),int((segments[i][0][1]+segments[i][1][1])/2))
            
            cv2.putText(img, str("X"), milsegement,
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    #finalement on va chercher dans le dictinnaire du début quel est le chiffre lu
     
    
    nombrelu = correspondances[tuple(on)]

    print(nombrelu)
   



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

        chiffre(finalimage)
        cv2.waitKey(0)

    else:
        print("Il n'y a pas assez d'ArUco détectés.")

def camera():
    # Ouvrir la caméra
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Impossible d'ouvrir la caméra du robot.")
        exit()

    while True:
        ret, frame = camera.read()
        cv2.imshow('Robot Camera', frame)
        
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            print("Erreur lors de la capture de l'image depuis la caméra du robot.")
            break

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters =  cv2.aruco.DetectorParameters()
        #instancier le détecteur
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

        if markerIds is not None and len(markerIds) == 4:
            cv2.imwrite('image_capturee.png', frame)
            image = cv2.imread('image_capturee.png')
            camera.release()

            nouvelle_image(markerIds, markerCorners, image)
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    camera()