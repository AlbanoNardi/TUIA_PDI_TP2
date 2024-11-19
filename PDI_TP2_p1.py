import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
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

img = cv2.imread('monedas.jpg')

img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

imshow(img_color)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imshow(img_gray)

img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

plt.figure()
ax1 = plt.subplot(121); imshow(img_gray, title='Imagen original en escala de grises', new_fig=False)
plt.subplot(122,sharex=ax1,sharey=ax1), imshow(img_blur, title='+ filtro Gaussiano para suavizar', new_fig=False)
plt.show(block=False)

circles = cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT,1,minDist=200,param1=150,param2=40,minRadius=100,maxRadius=200)
print("Cantidad de círculos detectados ",len(circles[0]))

img_color_ = img_color.copy()
circles = np.uint16(np.around(circles))  # Se usa para reordenar coordenadas de los círculos detectados al entero más cercano
for i in circles[0,:]:
    cv2.circle(img_color_,(i[0],i[1]),i[2],(0,255,0),3)
    cv2.circle(img_color_,(i[0],i[1]),2,(0,0,255),3)
imshow(img_color_)

img_pintada = cv2.cvtColor(img_color_, cv2.COLOR_BGR2GRAY)
for i in circles[0,:]:
    cv2.circle(img_pintada,(i[0],i[1]),i[2] + 10 ,(0,0,0),thickness=cv2.FILLED) # pintar circulos
imshow(img_pintada)

_, thresh_img = cv2.threshold(img_pintada,thresh=180,maxval=255,type=cv2.THRESH_BINARY)
imshow(thresh_img)

contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) < 500:  # umbral para borrar areas pequeñas de la imagen
        cv2.drawContours(thresh_img, [cnt], -1, (0, 0, 0), thickness=cv2.FILLED)
imshow(thresh_img)

f = thresh_img

A = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
C = cv2.getStructuringElement(cv2.MORPH_RECT, (30,30))
fop = cv2.morphologyEx(f, cv2.MORPH_OPEN, A)
fop_cl = cv2.morphologyEx(fop, cv2.MORPH_CLOSE, C)

plt.figure()
ax1 = plt.subplot(131); imshow(f, new_fig=False, title="Original")
plt.subplot(132, sharex=ax1, sharey=ax1); imshow(fop, new_fig=False, title="Apertura")
plt.subplot(133, sharex=ax1, sharey=ax1); imshow(fop_cl, new_fig=False, title="Apertura + Clausura")
plt.show(block=False)


img_inv = fop_cl==0
inv_uint8 = img_inv.astype(np.uint8)
imshow(inv_uint8)


"""contours, _ = cv2.findContours(inv_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours[0][1]
rois = []  # Lista para almacenar las regiones recortadas
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)  # Coordenadas del rectángulo
    roi = img_gray[y:y+h, x:x+w]  # Recortar la región de interés en la imagen original
    rois.append(roi)  # Almacenar el recorte


for i, roi in enumerate(rois):
    plt.figure()
    plt.imshow(roi, cmap='gray')
    plt.title(f"Dado {i+1}")
    plt.axis('off')
    plt.show()"""



for i in range(0,len(circles[0])):
    radio = circles[0][i][2]
    area = math.pi*radio**2
    perimetro = math.pi*(2*radio)
    ap = area/perimetro**2
    fp = 1/ap
    print(fp)



#-------------------------------------------Lo que hicimos en el primer meet
img = cv2.imread('monedas.jpg')

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img = cv2.imread('monedas.jpg', 0)

cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,minDist=70,
                param1=150,param2=65,minRadius=100,maxRadius=190)

print(circles)
print("Cantidad de círculos detectados ",len(circles[0]))

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

##cv2.imwrite('detected_circle.jpg',cimg)
plt.imshow(cimg)

plt.show()



img , thresh_img = cv2.threshold(cimg,thresh=183,maxval=255,type=cv2.THRESH_BINARY)
plt.imshow(thresh_img, cmap='gray')

plt.show()


for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(thresh_img,(i[0],i[1]),i[2],(0,0,0),thickness=cv2.FILLED)

plt.imshow(thresh_img, cmap='gray')

plt.show()




gris = cv2.cvtColor(thresh_img,cv2.COLOR_BGR2GRAY)

contours, hierarchy = cv2.findContours(gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

f = gris

cv2.drawContours(img2, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)

imshow(img2)


plt.imshow(img, cmap='gray')
plt.show()



f_blur = cv2.GaussianBlur(f, ksize=(3, 3), sigmaX=1.5)
gcan1 = cv2.Canny(f_blur, threshold1=0.04255, threshold2=0.1255)
gcan2 = cv2.Canny(f_blur, threshold1=0.4255, threshold2=0.6255)
gcan3 = cv2.Canny(f_blur, threshold1=0.4255, threshold2=0.75255)
imshow(f)

plt.figure()
ax = plt.subplot(221)
imshow(f, new_fig=False, title="Imagen Original")
plt.subplot(222, sharex=ax, sharey=ax), imshow(gcan1, new_fig=False, title="Canny - U1=4% | U2=10%")
plt.subplot(223, sharex=ax, sharey=ax), imshow(gcan2, new_fig=False, title="Canny - U1=40% | U2=50%")
plt.subplot(224, sharex=ax, sharey=ax), imshow(gcan3, new_fig=False, title="Canny - U1=40% | U2=75%")
plt.show(block=False)

#Dilat
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
Fd = cv2.dilate(gcan3, kernel, iterations=1)
plt.figure()
ax1 = plt.subplot(121); imshow(gcan3, new_fig=False, title="Original")
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(Fd, new_fig=False, title="Dilatacion")
plt.show(block=False)

#Claus
B = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
Aclau = cv2.morphologyEx(gcan3, cv2.MORPH_CLOSE, B)
plt.figure()
ax1 = plt.subplot(121); imshow(gcan3, new_fig=False, title="Original")
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(Aclau, new_fig=False, title="Clausura")
plt.show(block=False)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('contornos', gray)
umbral, thresh_img = cv2.threshold(gray, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
cv2.imshow('Umbral', thresh_img)

#Cont jera
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(hierarchy)    # hierarchy: [Next, Previous, First_Child, Parent]

#cont con poli
cnt = contours[12]
cv2.drawContours(img, cnt, contourIdx=-1, color=(255, 0, 0), thickness=2)
cv2.imshow('Aproximacion de contorno', img)
approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True) 
cv2.drawContours(img, approx, contourIdx=-1, color=(0, 0, 255), thickness=2)
imshow(img)





