import cv2
import numpy as np

img1 = cv2.imread("Homografia e Stitching/foto2.jpeg")
img2 = cv2.imread("Homografia e Stitching/foto1.jpeg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(2000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

matches = matches[:100]

pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)

altura, largura = img1.shape[:2]
panorama = cv2.warpPerspective(img2, H, (largura*2, altura))

panorama[0:altura, 0:largura] = img1

cv2.imwrite("panorama.jpeg", panorama)

print("Panorama criado!")