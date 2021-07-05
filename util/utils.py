import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def normalize_rgb(src):
    src = src.astype(np.uint16)
    R, G, B = cv.split(src)
    src = src.astype(np.uint8)

    r = np.nan_to_num(R / (R + G + B)) * 255
    g = np.nan_to_num(G / (R + G + B)) * 255
    b = np.nan_to_num(B / (R + G + B)) * 255

    norm_rgb = cv.merge([r, g, b])
    norm_rgb = norm_rgb.astype(np.uint8)

    return norm_rgb


def get_face_rect(cap):
    face_cascade = cv.CascadeClassifier(
        cv.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        _, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray)

        if len(faces) == 0:
            continue
        else:
            for rect in faces:
                return rgb, rect


def calculate_mean_deviation(cap):
    rgb, (x, y, w, h) = get_face_rect(cap)
    inner_region = rgb[y+25:y+h-25, x+25:x+w-25]

    x, y, _ = inner_region.shape
    n = x * y

    norm_inner = normalize_rgb(inner_region)

    r = norm_inner[:, :, 0]
    g = norm_inner[:, :, 1]
    R = inner_region[:, :, 0]

    # calculate the mean and standart deviation for each component
    mu_r = 1/n * np.sum(r)
    mu_g = 1/n * np.sum(g)
    mu_R = 1/n * np.sum(R)

    sigma_r = np.sqrt(1/n * np.sum((r-mu_r)**2))
    sigma_g = np.sqrt(1/n * np.sum((g-mu_g)**2))
    sigma_R = np.sqrt(1/n * np.sum((R-mu_R)**2))

    U_r, L_r = mu_r + 2*sigma_r, mu_r - 2*sigma_r
    U_g, L_g = mu_g + 2*sigma_g, mu_g - 2*sigma_g
    U_R, L_R = mu_R + 2*sigma_R, mu_R - 2*sigma_R

    return (U_r, L_r, U_g, L_g, U_R, L_R)


def calculate_adaptive_skin_mask(src, values, hide_rect=None):
    if hide_rect is not None:
        x, y, w, h = hide_rect
        src[y:y+h, x:x+w] = 0

    norm_src = normalize_rgb(src)

    r = norm_src[:, :, 0]
    g = norm_src[:, :, 1]
    R = src[:, :, 0]

    (U_r, L_r, U_g, L_g, U_R, L_R) = values

    _, t1 = cv.threshold(r - L_r, 0, 255, cv.THRESH_BINARY)
    _, t2 = cv.threshold(U_r - r, 0, 255, cv.THRESH_BINARY)
    _, t3 = cv.threshold(g - L_g, 0, 255, cv.THRESH_BINARY)
    _, t4 = cv.threshold(U_g - g, 0, 255, cv.THRESH_BINARY)
    _, t5 = cv.threshold(R - L_R, 0, 255, cv.THRESH_BINARY)
    _, t6 = cv.threshold(U_R - R, 0, 255, cv.THRESH_BINARY)

    t = cv.bitwise_and(t1, t2)
    t = cv.bitwise_and(t, t3)
    t = cv.bitwise_and(t, t4)
    t = cv.bitwise_and(t, t5)
    t = cv.bitwise_and(t, t6)

    return t
