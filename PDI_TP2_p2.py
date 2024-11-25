import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

def patente_umbral(imagen):
        
    img_original = cv2.imread(imagen)

    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) 

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) 

    w2 = -np.ones((5,5))/(5*5)  

    w2[2,2] = 24 / 25

    img_filtrada = cv2.filter2D(img_gray,-1,w2) # Filtro pasa altos

    img_thresh = cv2.adaptiveThreshold(img_filtrada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  
    contours_area = sorted(contours, key=cv2.contourArea, reverse=True)

    for contorno in contours_area:

        x, y, w, h = cv2.boundingRect(contorno) # Sacamos coordenadas de recuadro de patente

        aspect_ratio = w / h  # Relación de aspecto

        area = cv2.contourArea(contorno) # Cálculo del área de la patente

        rect_area = w * h # Área del recuadro rectangular

        fill_ratio = area / rect_area  # Proporción de área cubierta por el contorno
               
        if 1.78 <= aspect_ratio <= 3.1 and 0.9 > fill_ratio > 0.5 and 3800 > rect_area > 1400: # Filtrar por criterios típicos de patentes

            cv2.drawContours(img_rgb, [contorno], -1, (0, 255, 0), 2)

            #imshow(img_rgb)

            patente_recortada = img_original[y:y+h, x:x+w]

            #imshow(patente_recortada)

            return patente_recortada
        
        elif 1.78 <= aspect_ratio <= 3.1 and 2000 > rect_area > 1800:

            cv2.drawContours(img_rgb, [contorno], -1, (0, 255, 0), 2)

            #imshow(img_rgb)

            patente_recortada = img_original[y:y+h, x:x+w]

            #imshow(patente_recortada)
            
            return patente_recortada
    
        elif 1.78 <= aspect_ratio <= 3.1 and 2200 > rect_area > 2100:

            cv2.drawContours(img_rgb, [contorno], -1, (0, 255, 0), 2)

            #imshow(img_rgb)

            patente_recortada = img_original[y:y+h+10, x:x+w]

            #imshow(patente_recortada)
            
            return patente_recortada

    print("No se encontró la patente")

    return None

def checkear_tolerancia_consecutiva(arr, tolerancia):
    for i in range(1, len(arr)):
        dif = abs(arr[i] - arr[i - 1])
        max_dif_permitida = tolerancia * arr[i - 1] 
        if dif > max_dif_permitida :
            return False 
    return False if arr[0] / arr[len(arr)-1] > 1.5 else True

def segmentar_fuerza_bruta(patente):
    
    gray = cv2.cvtColor(patente, cv2.COLOR_BGR2GRAY)

    for t in range (50,200,10):

        patente_copy2 = patente.copy()

        patente_copy = cv2.cvtColor(patente_copy2, cv2.COLOR_BGR2RGB)

        _, thresh_img = cv2.threshold(gray, thresh=t, maxval=255, type=cv2.THRESH_BINARY)

        # Encontrar contornos
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        areas_detectadas = []
        caracteres = []
        for contorno in contours:

            x, y, w, h = cv2.boundingRect(contorno)
            aspect_ratio = h / w
            area_rect = w*h
            # Filtrar contornos que parezcan caracteres

            if 1.5 < aspect_ratio < 3.0:
                areas_detectadas.append(area_rect)

                caracter_recortado = gray[y:y+h, x:x+w]
                caracteres.append(({"x":x,"y":y,"w":w,"h":h}, caracter_recortado,area_rect))

        if len(caracteres) < 5:
            continue
        # Ordenar caracteres por su tamaño

        caracteres = sorted(caracteres, key=lambda c: c[2],reverse=True)
        
        for i,caracter in enumerate(caracteres[:-5]):

            arr = [x[2] for x in caracteres[i:i+6]]
            resultado = np.apply_along_axis(checkear_tolerancia_consecutiva,arr=arr,axis=0,tolerancia=0.1)
            
            if resultado:
                for d in [x[0] for x in caracteres[i:i+6]]:
                    cv2.rectangle(patente_copy, (d["x"], d["y"]), (d["x"] + d["w"], d["y"] + d["h"]), (0, 255, 0), 1)
                    
                imshow(patente_copy)
                return arr
    print("No se encontraron 6 caracteres")
    return None

def main():
    
    for i in range(1, 13):
        img_name = f"img{i:02d}.png"

        print(f"Procesando {img_name}...")

        patente = patente_umbral(img_name)
        
        if patente is not None:
            print(f"Patente detectada en {img_name}. Intentando segmentar caracteres...")

            caracteres = segmentar_fuerza_bruta(patente)
            
            if caracteres is not None:
                print(f"Segmentación exitosa en {img_name}. Áreas detectadas: {caracteres}")
            else:
                print(f"No se pudieron segmentar los caracteres en {img_name}.")
        else:
            print(f"No se detectó una patente en {img_name}.")
    
    print("Procesamiento completado.")

if __name__ == "__main__":
    main()

