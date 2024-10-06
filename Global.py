import tkinter as tk
import ttkbootstrap as ttk
from tkinter import Menu, filedialog, messagebox
from tkinter import ttk as tkttk  # Import de ttk pour Treeview
import cv2
import trait
import qr_img
import utlis
import numpy as np
import threading
import pytesseract
import qrcode
from PIL import Image, ImageTk  # Pour afficher le code QR dans Tkinter

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

chemin_control = []  # Liste pour stocker les chemins des images de contrôle
chemin_referen = None
scores = []  # Liste pour stocker les scores pour chaque image
data=None
data_control=None
# Fonction pour quitter l'application
def quitter():
    root.quit()

# Fonction pour vider le tableau
def clear_table():
    for row in tree.get_children():
        tree.delete(row)

# Fonction pour ouvrir une fenêtre des notes
def afficher_fenetre_notes():
    fenetre_notes = ttk.Toplevel(root)
    fenetre_notes.title("Fenêtre des Notes")
    label = ttk.Label(fenetre_notes, text="Ceci est la fenêtre des notes.", bootstyle="info")
    label.pack(pady=10)

# Fonction pour sélectionner des images de contrôle
def img_control():
    global chemin_control
    chemin_control_all = filedialog.askopenfilenames(title="Sélectionnez des images des Controles",
                                                     filetypes=[("Fichiers image", "*.png;*.jpg;*.jpeg;*.gif")])
    if chemin_control_all:
        messagebox.showinfo("Images sélectionnées", f"Vous avez sélectionné {len(chemin_control_all)} image(s).")
        chemin_control = list(chemin_control_all)  # Stocker les chemins dans la liste
        print("Chemins des images sélectionnées :", chemin_control)

# Fonction pour sélectionner une image de référence
def img_referen():
    global chemin_referen
    chemin_referen = filedialog.askopenfilename(title="Sélectionnez une image des references", 
                                                filetypes=[("Fichiers image", "*.png;*.jpg;*.jpeg;*.gif")])
    if chemin_referen:
        messagebox.showinfo("Image sélectionnée", f"Vous avez sélectionné : {chemin_referen}")

# Fonction qui sera appelée au début de l'opération
def operation():
    processing_thread = threading.Thread(target=process_operation)
    processing_thread.start()

# Fonction pour traiter l'opération principale
def process_operation():
    global chemin_control, chemin_referen, cap, scores ,data , data_control# Inclure la liste des scores

    webCamFeed = False  # Définissez ceci sur True si vous souhaitez utiliser la webcam
    cap = cv2.VideoCapture(0) if webCamFeed else None
    #--Tailler de l'image pour le traitement ---
    heightImg = 700
    widthImg = 700
    questions = 5
    choices = 5
    
    #---------------------------------------return the valeurs correction answers img refrence ------------------
    correct_answers= trait.extract_answers_from_image(image_path=chemin_referen, questions=questions, choices=choices)
    # for clear le score --------- pour nouveau score-----
    scores.clear()
    # -----   boucle pour  dans l'image des controles
    for pathImage in chemin_control:
            #---------------------------------------return the valeurs de code qr in img controles ------------------
            data_control=qr_img.lire_qr_code(image_path=pathImage)

            #---------------------------------------return the valeurs de code qr in img refrence ------------------
            data=trait.extrat_code_qr_from_image(image_path=chemin_referen)
            #------condition if code qr img_ref==img_control ------contenue le detection de nom et traitement de l'image -----------------
            if data==data_control:
                #-------------depart de détetcter le nom --------------
                print(f"Traitement de l'image : {pathImage}")
                img = cv2.imread(pathImage)
                img = cv2.resize(img, (800, 800))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                texte_extrait = "Non détecté"
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w > 100 and h > 20 and y < 200:
                        roi = img[y:y+h, x:x+w]
                        cv2.waitKey(0)
                        texte_extrait = pytesseract.image_to_string(roi, lang='eng')
                        print("Texte détecté :", texte_extrait)

                cv2.destroyAllWindows()
                #-----traitement pour calculter le score ------------
                if not webCamFeed:
                    img = cv2.imread(pathImage)
                    img = cv2.resize(img, (widthImg, heightImg))

                imgFinal = img.copy() if img is not None else None

                try:
                    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
                    imgCanny = cv2.Canny(imgBlur, 10, 70)
                    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    rectCon = utlis.rectContour(contours)

                    if len(rectCon) > 1:
                        biggestPoints = utlis.getCornerPoints(rectCon[0])
                        gradePoints = utlis.getCornerPoints(rectCon[1])

                        if biggestPoints.size != 0 and gradePoints.size != 0:
                            biggestPoints = utlis.reorder(biggestPoints)
                            pts1 = np.float32(biggestPoints)
                            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
                            matrix = cv2.getPerspectiveTransform(pts1, pts2)
                            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

                            gradePoints = utlis.reorder(gradePoints)
                            ptsG1 = np.float32(gradePoints)
                            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
                            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
                            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

                            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
                            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

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
                                    if countR == questions:
                                        break

                            myIndex = []
                            for x in range(0, questions):
                                arr = myPixelVal[x]
                                myIndexVal = np.where(arr == np.amax(arr))
                                myIndex.append(myIndexVal[0][0])

                            grading = []
                            for x in range(0, questions):
                                grading.append(1 if correct_answers[x] == myIndex[x] else 0)

                            score = (sum(grading) / questions) * 20
                            scores.append((pathImage, score, texte_extrait))
                            print(f"Score pour {texte_extrait}: {score}")

                except Exception as e:
                    print("Erreur lors du traitement : ", e)

                update_scores_table()
            
#---------------update for tableau  --------
def update_scores_table():
    for row in tree.get_children():
        tree.delete(row)

    for img_name, score, texte in scores:
        tree.insert("", "end", values=(img_name, texte, score))

# Fonction pour convertir le texte en QR code
def generer_qr_code():
    texte = entry_qr.get()
    if texte:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(texte)
        qr.make(fit=True)

        img_qr = qr.make_image(fill='black', back_color='white')
        img_qr.show()

        # Afficher dans une nouvelle fenêtre
        fenetre_qr = ttk.Toplevel(root)
        fenetre_qr.title("Code QR")

        img_qr_tk = ImageTk.PhotoImage(img_qr)
        label_qr = ttk.Label(fenetre_qr, image=img_qr_tk)
        label_qr.image = img_qr_tk  # Stocker une référence pour éviter que l'image soit supprimée par le garbage collector
        label_qr.pack(pady=10)

# Création de la fenêtre principale
root = ttk.Window(themename="cyborg")
root.geometry("800x700")
root.title("Interface de Correction Automatisée")

# Ajouter un menu
menu_bar = Menu(root)
root.config(menu=menu_bar)

fichier_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Fichier", menu=fichier_menu)
fichier_menu.add_command(label="Ouvrir les images de contrôle", command=img_control)
fichier_menu.add_command(label="Ouvrir l'image de référence", command=img_referen)
fichier_menu.add_separator()
fichier_menu.add_command(label="Quitter", command=quitter)

# Ajouter une section pour les paramètres de correction
frame_settings = ttk.LabelFrame(root, text="Paramètres", bootstyle="primary")
frame_settings.pack(fill="x", padx=10, pady=5)


btn_img_control = ttk.Button(frame_settings, text="Choisir les images de contrôle", command=img_control, bootstyle="info")
btn_img_control.grid(row=0, column=0, padx=5, pady=5)

btn_img_referen = ttk.Button(frame_settings, text="Choisir l'image de référence", command=img_referen, bootstyle="info")
btn_img_referen.grid(row=0, column=1, padx=5, pady=5)


btn_lancer = ttk.Button(frame_settings, text="Lancer l'opération", command=operation, bootstyle="primary")
btn_lancer.grid(row=2, columnspan=2, padx=5, pady=5)

# Ajouter une section pour les scores
frame_scores = ttk.LabelFrame(root, text="Résultats", bootstyle="primary")
frame_scores.pack(fill="both", padx=10, pady=5, expand=True)

tree = tkttk.Treeview(frame_scores, columns=("Image", "Texte Détecté", "Score"), show="headings")
tree.heading("Image", text="Image")
tree.heading("Texte Détecté", text="Texte Détecté")
tree.heading("Score", text="Score")
tree.pack(fill="both", expand=True)

# Ajouter une section pour générer le QR code
frame_qr = ttk.LabelFrame(root, text="Générer un Code QR", bootstyle="primary")
frame_qr.pack(fill="x", padx=10, pady=5)

label_qr = ttk.Label(frame_qr, text="Entrez du texte :", bootstyle="primary")
label_qr.grid(row=0, column=0, padx=5, pady=5)

entry_qr = ttk.Entry(frame_qr)
entry_qr.grid(row=0, column=1, padx=5, pady=5)

btn_qr = ttk.Button(frame_qr, text="Générer QR Code", command=generer_qr_code, bootstyle="primary")
btn_qr.grid(row=1, columnspan=2, padx=5, pady=5)

root.mainloop()