import cv2

# Fonction pour lire un QR code à partir d'une image
def lire_qr_code(image_path):
    # Lire l'image
    img = cv2.imread(image_path)

    if img is None:
        print("Erreur : Impossible d'ouvrir ou de lire le fichier. Vérifie le chemin d'accès.")
        return

    # Créer un détecteur de QR code
    detector = cv2.QRCodeDetector()

    # Détecter et décoder le QR code
    data_control, points, _ = detector.detectAndDecode(img)

    if data_control:
        #print(f"QR Code trouvé  for image_controles: {data_control}")

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

        return data_control
    else:
        print("Aucun QR Code trouvé.")

        
if __name__ == "__main__":
    # Image de QCM avec les bonnes réponses
    data_control=lire_qr_code() 
# Exemple d'utilisation
# Assure-toi que le chemin est correct
