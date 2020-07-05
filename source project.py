import cv2 as cv                                # for processing images
import numpy as nmp                                    # for operating on image as a matrix
import pytesseract                              # for text detection

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\\tesseract.exe'
frameWidth = 900
frameHeight = 1400
counter = 1

video_location = "sample\\video2.mp4"
feed = cv.VideoCapture(video_location)
feed.set(10, 50)
feed.set(3, 900)
feed.set(4, 1400)

while True:
    success, image_feeded = feed.read()
    image_feeded_showcase = image_feeded.copy()
    image_feeded = cv.resize(image_feeded, (frameWidth, frameHeight))
    img_contour = image_feeded.copy()


    # #########################################################################################
    # EDGE DETECTION
    # #########################################################################################

    img_gray = cv.cvtColor(image_feeded, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv.Canny(img_blur, 200, 200)
    kernel = nmp.ones((5, 5))
    img_dil = cv.dilate(img_canny, kernel, iterations=2)
    img_threshold = cv.erode(img_dil, kernel, iterations=1)

    # #########################################################################################
    # LARGEST CONTOUR DETECTION
    # #########################################################################################

    largest_contours = nmp.array([])
    max_area = 0
    contours, hierarchy = cv.findContours(img_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        my_area = 70000
        if area > my_area:
            parameters = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * parameters, True)
            if area > max_area and len(approx) == 4:
                largest_contours = approx
                max_area = area

    # #########################################################################################
    # WARPING THE VIDEO TO A DESIRED CONTOUR
    # #########################################################################################

    vertices = largest_contours.reshape((4, 2))
    warp_vertices = nmp.zeros((4, 1, 2), nmp.int32)
    additions = vertices.sum(1)

    warp_vertices[0] = vertices[nmp.argmin(additions)]
    warp_vertices[3] = vertices[nmp.argmax(additions)]

    differences = nmp.diff(vertices, axis=1)
    warp_vertices[1] = vertices[nmp.argmin(differences)]
    warp_vertices[2] = vertices[nmp.argmax(differences)]

    # # #

    point1 = nmp.float32(warp_vertices)
    point2 = nmp.float32([[0, 0], [frameWidth, 0], [0, frameHeight], [frameWidth, frameHeight]])
    matrix_transformed = cv.getPerspectiveTransform(point1, point2)
    img_out = cv.warpPerspective(image_feeded, matrix_transformed, (frameWidth, frameHeight))

    img_cropped = img_out[10:img_out.shape[0] - 10, 10:img_out.shape[1] - 10]
    img_warped = cv.resize(img_cropped, (frameWidth, frameHeight))



    cv.imshow("Input", image_feeded_showcase)
    cv.imshow("result", img_warped)

    # #########################################################################################
    # DETECTING TEXT AND SAVING SCANNED TXT & JPG
    # #########################################################################################


    if cv.waitKey(1) & 0xFF == ord('s'):
        cv.imwrite("output\\ScannedDoc" + str(counter) + ".jpg", img_warped)

        # MAKING VIDEO VISIBLE & DETECTABLE FOR THE TESSERACT
        image_gray = cv.cvtColor(img_warped, cv.COLOR_BGR2GRAY)
        kernelSecond = nmp.ones((1, 1), nmp.uint8)
        image_blur = cv.GaussianBlur(image_gray, (5, 5), 1)
        image_dilate = cv.dilate(image_blur, kernelSecond, iterations=3)
        image_eroded = cv.erode(image_dilate, kernelSecond, iterations=2)
        image_blackWhite = cv.adaptiveThreshold(image_eroded, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)
        bnw = image_blackWhite

        # DETECTING TEXT
        stringFromImage = pytesseract.image_to_string(bnw)
        # SAVING THE DETECTED TEXT INTO A .txt FILE
        f = open("output\\ScannedText" + str(counter) + ".txt", "w")
        f.write(stringFromImage)
        f.close()

        # GETTING DATA (e.g, blank spaces, word, paragraphs etc) FROM VIDEO
        data = pytesseract.image_to_data(bnw)
        # SPLITTING DATA FROM A SINGLE STRING TO AN ARRAY OF STRINGS
        for a, b in enumerate(data.splitlines()):
            if a != 0:
                b = b.split()
                # GETTING BOUNDING BOX INFO. (such as starting point, width, height)
                if len(b) == 12:
                    x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                    # CREATING BOUNDING BOXES AND RESPECTIVE WORD
                    cv.putText(bnw, b[11], (x, y - 5), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (150, 150, 150), 2)
                    cv.rectangle(bnw, (x, y), (x + w, y + h), (50, 50, 255), 2)
                    cv.imwrite("output\\ScannedWords" + str(counter) + ".jpg", bnw)

     #   processText(img_warped, counter)

        cv.rectangle(img_warped, (300, 300), (600, 400), (190, 150, 150), cv.FILLED)
        cv.putText(img_warped, "SAVED", (400, 355), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        counter += 1
        cv.imshow("result", img_warped)

        cv.waitKey(500)