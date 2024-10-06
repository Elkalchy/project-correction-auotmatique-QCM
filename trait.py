import cv2
import numpy as np
import utlis

def extrat_code_qr_from_image(image_path):
    #------------------------------------code read code Qr---------------------------------------
    img = cv2.imread(image_path)

    if img is None:
        print("Erreur : Impossible d'ouvrir ou de lire le fichier. Vérifie le chemin d'accès.")
        return

    # Créer un détecteur de QR code
    detector = cv2.QRCodeDetector()
    # Détecter et décoder le QR code
    data, points, _ = detector.detectAndDecode(img)
    
    if data:
        #print(f"QR Code trouvé for image_references: {data}")

        # Si des points ont été trouvés, dessiner un rectangle autour du QR code
        if points is not None:
            points = points[0].astype(int)  # Convertir les points en entiers
            for i in range(4):
                # Dessiner des lignes entre les points
                cv2.line(img, tuple(points[i]), tuple(points[(i + 1) % 4]), (0, 255, 0), 3)

        # Afficher l'image avec le QR code détecté
        #cv2.imshow("Image QR", img)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        return data
    else:
        print("Aucun QR Code trouvé.")
def extract_answers_from_image(image_path, questions=5, choices=5):
#---------------------------------traitement pour detecter les reponses ---------------------

    # Lire et redimensionner l'image
    img = cv2.imread(image_path)
    heightImg = 700
    widthImg = 700
    img = cv2.resize(img, (widthImg, heightImg))

    # Prétraitement de l'image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 70)

    # Trouver tous les contours
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rectCon = utlis.rectContour(contours)

    # Si les contours rectangulaires sont trouvés
    if len(rectCon) > 0:
        biggestPoints = utlis.getCornerPoints(rectCon[0])  # Plus grand rectangle

        if biggestPoints.size != 0:
            # Transformation de perspective
            biggestPoints = utlis.reorder(biggestPoints)
            pts1 = np.float32(biggestPoints)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # Appliquer un seuil pour isoler les cases
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

            # Diviser en cases
            boxes = utlis.splitBoxes(imgThresh)
            myPixelVal = np.zeros((questions, choices))

            countR = 0
            countC = 0
            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if countC == choices:
                    countC = 0
                    countR += 1

            # Trouver les réponses correctes
            ans = []
            for x in range(0, questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                ans.append(myIndexVal[0][0])  # Ajouter l'indice de la réponse la plus remplie

            return ans
    return []

# Utilisation de la fonction
if __name__ == "__main__":
    # Image de QCM avec les bonnes réponses
    correct_answers = extract_answers_from_image()
    data=extrat_code_qr_from_image()
    #print(f"Réponses correctes extraites : {correct_answers,data}")
    
