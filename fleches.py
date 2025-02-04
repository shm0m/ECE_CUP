import numpy as np
import cv2 

vidcap = cv2.VideoCapture(0)
if vidcap.isOpened():
    while(True):
        ret, frame = vidcap.read()
        #définir le dictionnaire de travail
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters =  cv2.aruco.DetectorParameters()
        #instancier le détecteur
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
        if ret: 
            cv2.imshow("Frame",frame)
            if cv2.waitKey(1) & 0xFF==ord('q') or len(markerCorners) >= 4:
                cv2.imwrite('opencvtest.png', frame)
                break
image = cv2.imread('opencvtest.png')
imageneutre = cv2.imread('opencvtest.png')

cv2.destroyAllWindows()

del(vidcap)

#stocker les arucos détectés : les positions de leurs coins, leur identifiants et les éventuelles erreurs

print(markerIds)
print(markerCorners)
my_listx = []
my_listy = []
my_listtopleft = []
topleft = []
my_listtopright = []
topright = []
my_listbotomleft = []
botomleft = []
my_listbotomright = []
botomright = []
i=0
# verifier qu'au moins 3 aruco est détecté
if len(markerCorners) > 3:
	ids = markerIds.flatten()
	# boucle pour chaque aruco détecté
	for (markerCorner, markerID) in zip(markerCorners, ids):
		# extraire les angles des aruco (toujours dans l'ordre
		# haut-gauche, haut-droite, bas-gauche, bas-droit)
		corners = markerCorner.reshape((4, 2))
		(topLeft, topRight, bottomRight, bottomLeft) = corners
		# convertir en entier (pour l'affichage)
		topRight = (int(topRight[0]), int(topRight[1]))
		bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
		bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
		topLeft = (int(topLeft[0]), int(topLeft[1]))
		
		my_listtopleft.append(topLeft)
		my_listtopright.append(topRight)
		my_listbotomleft.append(bottomLeft)
		my_listbotomright.append(bottomRight)
		


        # dessiner un quadrilatère autour de chaque aruco
		cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
		cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
		cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
		cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
		# calculer puis afficher un point rouge au centre
		my_listx.append(int((topLeft[0] + bottomRight[0]) / 2.0))
		my_listy.append(int((topLeft[1] + bottomRight[1]) / 2.0))
		cv2.circle(image, (my_listx[i] , my_listy[i]), 4, (0, 0, 255), -1)
		# affiher l'identifiant
		cv2.putText(image, str(markerID),
			(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 255, 0), 2)
		print("[INFO] ArUco marker ID: {}".format(markerID))
		i=i+1
		# afficher l'image
		
#test crop image


cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

compteury = [0] * 4
compteurx = [0] * 4
compteurbotomRight = [0] * 4
compteurbotomleft = [0] * 4
compteurtopleft = [0] * 4
compteurtopright= [0] * 4
compteurbotomRighty = [0] * 4
compteurbotomlefty = [0] * 4
compteurtoplefty = [0] * 4
compteurtoprighty= [0] * 4
print(my_listy)
print(my_listx)
print(my_listbotomright)
print(my_listbotomleft)
print(my_listtopleft)
print(my_listtopright)

for l in range(4):
        if my_listbotomright[l][0] > my_listbotomright[0][0]:
            compteurbotomRight[l] += 1
        if my_listbotomright[l][0] > my_listbotomright[1][0]:
            compteurbotomRight[l] += 1
        if my_listbotomright[l][0] > my_listbotomright[2][0]:
            compteurbotomRight[l] += 1
        if my_listbotomright[l][0] > my_listbotomright[3][0]:
            compteurbotomRight[l] += 1
            
        if my_listbotomleft[l][0] > my_listbotomleft[0][0]:
            compteurbotomleft[l] += 1
        if my_listbotomleft[l][0] > my_listbotomleft[1][0]:
            compteurbotomleft[l] += 1
        if my_listbotomleft[l][0] > my_listbotomleft[2][0]:
            compteurbotomleft[l] += 1
        if my_listbotomleft[l][0] > my_listbotomleft[3][0]:
            compteurbotomleft[l] += 1
            
        if my_listtopleft[l][0] > my_listtopleft[0][0]:
            compteurtopleft[l] += 1
        if my_listtopleft[l][0] > my_listtopleft[1][0]:
            compteurtopleft[l] += 1
        if my_listtopleft[l][0] > my_listtopleft[2][0]:
            compteurtopleft[l] += 1
        if my_listtopleft[l][0] > my_listtopleft[3][0]:
            compteurtopleft[l] += 1
            
        if my_listtopright[l][0] > my_listtopright[0][0]:
            compteurtopright[l] += 1
        if my_listtopright[l][0] > my_listtopright[1][0]:
            compteurtopright[l] += 1
        if my_listtopright[l][0] > my_listtopright[2][0]:
            compteurtopright[l] += 1
        if my_listtopright[l][0] > my_listtopright[3][0]:
            compteurtopright[l] += 1
            
            
        if my_listbotomright[l][1] > my_listbotomright[0][1]:
            compteurbotomRighty[l] += 1
        if my_listbotomright[l][1] > my_listbotomright[1][1]:
            compteurbotomRighty[l] += 1
        if my_listbotomright[l][1] > my_listbotomright[2][1]:
            compteurbotomRighty[l] += 1
        if my_listbotomright[l][1] > my_listbotomright[3][1]:
            compteurbotomRighty[l] += 1
            
        if my_listbotomleft[l][1] > my_listbotomleft[0][1]:
            compteurbotomlefty[l] += 1
        if my_listbotomleft[l][1] > my_listbotomleft[1][1]:
            compteurbotomlefty[l] += 1
        if my_listbotomleft[l][1] > my_listbotomleft[2][1]:
            compteurbotomlefty[l] += 1
        if my_listbotomleft[l][1] > my_listbotomleft[3][1]:
            compteurbotomlefty[l] += 1
            
        if my_listtopleft[l][1] > my_listtopleft[0][1]:
            compteurtoplefty[l] += 1
        if my_listtopleft[l][1] > my_listtopleft[1][1]:
            compteurtoplefty[l] += 1
        if my_listtopleft[l][1] > my_listtopleft[2][1]:
            compteurtoplefty[l] += 1
        if my_listtopleft[l][1] > my_listtopleft[3][1]:
            compteurtoplefty[l] += 1
            
        if my_listtopright[l][1] > my_listtopright[0][1]:
            compteurtoprighty[l] += 1
        if my_listtopright[l][1] > my_listtopright[1][1]:
            compteurtoprighty[l] += 1
        if my_listtopright[l][1] > my_listtopright[2][1]:
            compteurtoprighty[l] += 1
        if my_listtopright[l][1] > my_listtopright[3][1]:
            compteurtoprighty[l] += 1

for m in range(4):
    if compteurtopright[m]>1:
        if compteurtoprighty[m]>1:
            botomright.append(my_listtopright[m])
    if compteurtopright[m]>1:
        if compteurtoprighty[m]<=1:
            topright.append(my_listtopright[m])
    if compteurtopright[m]<=1:
        if compteurtoprighty[m]>1:
            botomleft.append(my_listtopright[m])
    if compteurtopright[m]<=1:
        if compteurtoprighty[m]<=1:
            topleft.append(my_listtopright[m])
            
    if compteurtopleft[m]>1:
        if compteurtoplefty[m]>1:
            botomright.append(my_listtopleft[m])
    if compteurtopleft[m]>1:
        if compteurtoplefty[m]<=1:
            topright.append(my_listtopleft[m])
    if compteurtopleft[m]<=1:
        if compteurtoplefty[m]>1:
            botomleft.append(my_listtopleft[m])
    if compteurtopleft[m]<=1:
        if compteurtoplefty[m]<=1:
            topleft.append(my_listtopleft[m])
            
    if compteurbotomRight[m]>1:
        if compteurbotomRighty[m]>1:
            botomright.append(my_listbotomright[m])
    if compteurbotomRight[m]>1:
        if compteurbotomRighty[m]<=1:
            topright.append(my_listbotomright[m])
    if compteurbotomRight[m]<=1:
        if compteurbotomRighty[m]>1:
            botomleft.append(my_listbotomright[m])
    if compteurbotomRight[m]<=1:
        if compteurbotomRighty[m]<=1:
            topleft.append(my_listbotomright[m])
            
    if compteurbotomleft[m]>1:
        if compteurbotomlefty[m]>1:
            botomright.append(my_listbotomleft[m])
    if compteurbotomleft[m]>1:
        if compteurbotomlefty[m]<=1:
            topright.append(my_listbotomleft[m])
    if compteurbotomleft[m]<=1:
        if compteurbotomlefty[m]>1:
            botomleft.append(my_listbotomleft[m])
    if compteurbotomleft[m]<=1:
        if compteurbotomlefty[m]<=1:
            topleft.append(my_listbotomleft[m])

for y in range(4):
        if my_listy[y] > my_listy[0]:
            compteury[y] += 1
        if my_listy[y] > my_listy[1]:
            compteury[y] += 1
        if my_listy[y] > my_listy[2]:
            compteury[y] += 1
        if my_listy[y] > my_listy[3]:
            compteury[y] += 1

for j in range(4):
        if my_listx[j] > my_listx[0]:
            compteurx[j] += 1
        if my_listx[j] > my_listx[1]:
            compteurx[j] += 1
        if my_listx[j] > my_listx[2]:
            compteurx[j] += 1
        if my_listx[j] > my_listx[3]:
            compteurx[j] += 1
            
print("compteury : ")
print(compteury)
print("compteurx : ")
print(compteurx)
sommets = []
sommet1 =(0,0)
sommet2=(0,0)
sommet3=(0,0)
sommet4=(0,0)

for k in range(4):
    if compteury[k] > 1 :
        if compteurx[k] > 1:
            sommets.append(topleft[k])
    if compteury[k] > 1 :
        if compteurx[k] <= 1:
            sommets.append(topright[k])
    if compteury[k] <= 1 :
        if compteurx[k] > 1:
            sommets.append(botomleft[k])
    if compteury[k] <= 1 :
        if compteurx[k] <= 1:
            sommets.append(botomright[k])



for k in range(4):
    if (compteurx[k] > 1 and compteury[k] >1) :
        sommet1 = sommets[k]
    if (compteurx[k] > 1 and compteury[k] <=1) :
        sommet2 = sommets[k]
    if (compteurx[k] <= 1 and compteury[k] >1) :
        sommet3 = sommets[k]
    if (compteurx[k] <= 1 and compteury[k] <=1) :
        sommet4 = sommets[k]


print("apres traitements : ")
print(my_listy)
print(my_listx)
print(my_listbotomright)
print(my_listbotomleft)
print(my_listtopleft)
print(my_listtopright)
print(topleft)
print(topright)
print(botomleft)
print(botomright)

h,w=imageneutre.shape[:2]

#specifions d'abord la positions de points dans la première image puis leurs position dans la nouvelle
points1 = np.float32([ sommet1, sommet3, sommet2,sommet4])
points2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
print(points1)
#calculons une matrice de transformation
vecttrans = cv2.getPerspectiveTransform(points1, points2)
#appliquons la transformation à l'image d'orgine,spécifions un taille
FIMG = cv2.warpPerspective(imageneutre, vecttrans, (w, h))

#displaying the original image and the transformed image as the output on the screen
cv2.imshow('Source_image', FIMG)
cv2.waitKey(0)
cv2.destroyAllWindows()


img = FIMG
image = FIMG
ret = FIMG
if (markerIds[0] == 13):

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
 