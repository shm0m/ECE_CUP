import numpy as np
import cv2

# Fonction pour détecter la couleur dominante dans une image
def detect_color(image):
    # Convertir l'image en HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir les plages de couleurs
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([30, 50, 50])  # Adjusted lower bound for green
    upper_green = np.array([90, 255, 255])  # Adjusted upper bound for green

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

    
def chiffre (image) :
    h, w, _ = image.shape
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)

    coordinates_aruco = []

    # Vérifier qu'au moins un ArUco est détecté
    if len(markerCorners) > 0:
        ids = markerIds.flatten()
        # Boucle pour chaque ArUco détecté
        for (markerCorner, markerID) in zip(markerCorners, ids):
            corners = markerCorner.reshape((4, 2))
            echelle = np.linalg.norm(corners[0] - corners[1])
            coordinates_aruco.append([tuple(row) for row in corners])
    coordinates_aruco_sorted = sorted(coordinates_aruco,key=lambda corners: sum(coord[0] + coord[1] for coord in corners))

    echelle = echelle*0.5

    coordinates_aruco_sorted[0][1] = (coordinates_aruco_sorted[0][1][0] , coordinates_aruco_sorted[0][2][1])
    coordinates_aruco_sorted[1][2] = (coordinates_aruco_sorted[1][3][0], coordinates_aruco_sorted[1][2][1])
    coordinates_aruco_sorted[2][0] = (coordinates_aruco_sorted[2][1][0], coordinates_aruco_sorted[2][0][1])
    coordinates_aruco_sorted[3][3] = (coordinates_aruco_sorted[3][3][0], coordinates_aruco_sorted[3][0][1])

    final_aruco_coordinate = [coordinates_aruco_sorted[0][1],coordinates_aruco_sorted[1][2],coordinates_aruco_sorted[2][0],coordinates_aruco_sorted[3][3]]
    print(final_aruco_coordinate)
    print(coordinates_aruco_sorted)
    # Convertir les coordonnées en format adapté à la transformation
    points1 = np.float32(final_aruco_coordinate)
    points2 = np.float32([[0, 0],[w, 0],[0, h],[w, h]])
    # Calculer la matrice de transformation
    vecttrans = cv2.getPerspectiveTransform(points1, points2)

    # Appliquer la transformation à l'image originale
    finalimage = cv2.warpPerspective(image, vecttrans, (w, h))

    # Enregistrer l'image transformée
    cv2.imwrite('new_image.png', finalimage)

    cv2.imshow('Image originale', image)
    cv2.waitKey(0)
    for point in final_aruco_coordinate:
        cv2.circle(image, tuple(map(int, point)), 5, (0, 255, 0), -1)

    cv2.imshow('Image originale avec coordonnées finales', image)
    cv2.imshow('Transformed Image', finalimage)
    image = finalimage
    # Créons un dictionnaire de correspondance entre l'état des segments et la valeur lue
    correspondances = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 0, 1): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9,
    }

    (H, L) = image.shape[:2]  # Récupérer la hauteur et la largeur de l'image

    # Définir 7 zones d'intérêts, une par segment
    segments = [
        ((int(L / 4), int(H / 16)), (int(L * 3 / 4), int(H / 6))),  # haut
        ((int(L / 10), int(H / 7)), (int(L / 4), int(H * 9 / 20))),  # haut-gauche
        ((int(L * 3 / 4), int(H / 7)), (int(L * 9 / 10), int(H * 9 / 20))),  # haut-droite
        ((int(L / 5), int(H * 9 / 20)), (int(L * 3 / 4), int(H * 11 / 20))),  # centre
        ((int(L / 10), int(H * 11 / 20)), (int(L / 4), int(H * 4 / 5))),  # bas-gauche
        ((int(L * 3 / 4), int(H * 11 / 20)), (int(L * 9 / 10), int(H * 4 / 5))),  # bas-droite
        ((int(L / 4), int(H * 5 / 6)), (int(L * 3 / 4), int(H * 9 / 10)))  # bas
    ]

    # Afficher l'image avec les zones d'intérêt
    for rect in segments:
        color = tuple(np.random.random(size=3) * 256)
        image = cv2.rectangle(image, rect[0], rect[1], color, 3)

    cv2.imshow('test', image)
    cv2.waitKey(0)

    # Passer en niveau de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Définir un seuil et binariser l'image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Créer un tableau des états vus des segments (1 si le segment est noir, 0 si le segment est blanc)
    on = [0] * len(segments)

    # Voir l'état pour chaque segment :
    for i, ((xA, yA), (xB, yB)) in enumerate(segments):
        # Extraire une image binaire de la zone d'intérêt correspondant au segment
        segROI = thresh[yA:yB, xA:xB]
        # Compter le nombre de pixels noirs
        nbpixels = cv2.countNonZero(segROI)
        # Calculer l'aire du segment
        area = (xB - xA) * (yB - yA)
        # Modifier le tableau d'état vu si le nombre de pixels noirs dépasse 50% de l'aire
        if nbpixels / float(area) > 0.5:
            on[i] = 1

    # Finalement, on va chercher dans le dictionnaire du début quel est le chiffre lu
    nombrelu = correspondances[tuple(on)]

    print(nombrelu)

    cv2.imshow('test', image)
    cv2.waitKey(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

   

def fleches(img,):
    img = cv2.GaussianBlur(img, (11,11), 0)
    #On travaille sur une image en niveau de gris
    #on pourrait également binariser l'image, mais il faudrait calculer un seuil
    #la fonction goodFeaturesToTrack le fait déjà assez bien par elle même
    img_gris = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #on calcule les sommets avec une fonction bien pratique d'openCV
    #une flèche devrait avoir 7 sommets
    sommets = np.int0(cv2.goodFeaturesToTrack(img_gris,7,0.01,10))

    #par principe de déboguage on afiche en console les différents sommets trouvés :
    #cette boucle peut donc être supprimée
    for i,vals in enumerate(sommets):
        x,y = vals.ravel()
        print('sommet '+str(i)+' : '+str(x)+','+str(y))
        #les lignes suivantes permettent un rendu visuel pour débugger
        cv2.circle(img,(x,y),3,(255,255,0),-1)
        cv2.putText(img, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA )

    #on prends les sommets les plus éloignés
    xmax, ymax = (np.max(sommets, axis = 0)).ravel()
    xmin, ymin = (np.min(sommets, axis = 0)).ravel() 

    #déterminons l'axe du milieu de notre flèche, puis traçons le visuellemet (cv2.line)
    xmil=int(xmin+((xmax-xmin)/2))
    cv2.line(img,(xmil,0),(xmil,img.shape[0]),(255,0,0),2)

    #on compte le nombre de sommets à gauche puis à droite de notre milieu
    #au vu de la forme de notre flèche le nombre de sommmets est dans la partie "pointe" notre flèche
    nbSommetsDroite=np.count_nonzero(sommets[:,0,0]>xmil)
    nbSommetsGauche=np.count_nonzero(sommets[:,0,0]<xmil)

    #on finit par afficher le sens de notre flèche
    if nbSommetsDroite>nbSommetsGauche:
        print('Droite')
    else:
        print('Gauche')


    #les prochaines lignes ne servent qu'à l'affichage graphique
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

        if 9 in markerIds:
            chiffre(finalimage)
        elif 13 in markerIds:
            fleches(finalimage)
        elif 8 in markerIds:
            detect_color(finalimage)
            

# Capturer la vidéo depuis la webcam
vidcap = cv2.VideoCapture(0)

if vidcap.isOpened():
    while True:
        ret, frame = vidcap.read()
        # Définir le dictionnaire de travail
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary)

        if ret:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or len(markerCorners) >= 4:
                break  # Quit the loop if 'q' is pressed or enough markers are detected

# Save the captured image
cv2.imwrite('opencvtest4.png', frame)

# Charger l'image capturée
image = cv2.imread('opencvtest4.png')


# Afficher les ArUco détectés
print("ArUco IDs:", markerIds)
print("ArUco Corners:", markerCorners)

# Afficher les coins des ArUco détectés
for i, (markerCorner, markerID) in enumerate(zip(markerCorners, markerIds)):
    print("ArUco marker ID:", markerID)
    corners = markerCorner.reshape((4, 2))
    for j, corner in enumerate(corners):
        print(f"Corner {j + 1}: {corner}")

nouvelle_image(markerIds,markerCorners,image)
# Afficher l'image avec les marqueurs ArUco encadrés
for markerCorner in markerCorners:
    cv2.polylines(image, np.int32([markerCorner]), True, (255, 0, 0), 2)

# Afficher l'image
cv2.imshow("Detected ArUco Markers", image)
cv2.waitKey(0)
cv2.destroyAllWindows()