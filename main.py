import cv2
import mediapipe
import numpy as np
import pandas as pd

######### FACE EXTRACTING #########
def processFace(imagePath):
    # Read and Show Image
    img = cv2.imread(imagePath)

    # Face Landmark Detector
    mp_face_mesh = mediapipe.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # results = face_mesh.process(img)

    landmarks = results.multi_face_landmarks[0]
    jumlah_data_landmark = len(landmarks.landmark)
    print("Jumlah data landmark:", jumlah_data_landmark)

    df = pd.DataFrame(list(mp_face_mesh.FACEMESH_FACE_OVAL), columns = ["p1", "p2"])
    print(df)

    print(f"\nBentuk wajah memiliki {df.shape[0]} garis\n")

    # Order Landmark Points
    routes_idx = []

    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]

    for i in range(0, df.shape[0]):
        # print(p1, p2)

        obj = df[df["p1"] == p2]
        p1 = obj["p1"].values[0]
        p2 = obj["p2"].values[0]

        route_idx = []
        route_idx.append(p1)
        route_idx.append(p2)
        routes_idx.append(route_idx)

    display_items = 72
    for idx, route_idx in enumerate(routes_idx[0:display_items] + routes_idx[-display_items:]):
        print(f"hubungkan landmark ke-{route_idx[0]} dengan landmark ke-{route_idx[1]}")
        if idx == display_items - 1:
            print("\nlalu...\n")

    # Find 2d coordinate values of each landmark point
    routes = []

    # for source_idx, target_idx in mp_face_mesh.FACEMESH_FACE_OVAL:
    for source_idx, target_idx in routes_idx:
        source = landmarks.landmark[source_idx]
        # print("source\n", source)
        target = landmarks.landmark[target_idx]
        # print("target\n", target)

        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        # print("relative source\n", relative_source)
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
        # print("relative target\n", relative_target)

        # cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness = 2)

        routes.append(relative_source)
        routes.append(relative_target)

    print(f"Banyak {len(routes)} titik landmark yang tersedia")
    # print(routes[0:71])

    # Extract Inner Area of Facial Landmark
    # img = cv2.imread(imagePath)
    mask = np.zeros((img.shape[0], img.shape[1]))
    mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
    mask = mask.astype(bool)

    out = np.zeros_like(img)
    out[mask] = img[mask]

    # fig = plt.figure(figsize = (15, 15))

    # plt.axis('off')
    # plt.imshow(out[:, :, ::-1])
    # plt.show()

    cv2.imwrite("Results/faceExtraction.png", out)

def featureFace(imagePath):
    # Initialize FaceMesh
    mp_face_mesh = mediapipe.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # Load an image or capture from a camera
    image = cv2.imread(imagePath)
    image1 = cv2.imread(imagePath)  # Replace with your image path
    image2 = cv2.imread(imagePath)  # Replace with your image path
    image3 = cv2.imread(imagePath)  # Replace with your image path
    image4 = cv2.imread(imagePath)  # Replace with your image path

    # Convert the image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get the facial landmarks
    results = face_mesh.process(image_rgb)

    # Check if landmarks are found
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Landmark indices for lips, eyes, eyebrow
            left_cheek_indices = [116,117,118,101,36,203,92,135,215,227]

            # Extract landmark points
            left_cheek_points = [(int(face_landmarks.landmark[idx].x * image1.shape[1]),
                                 int(face_landmarks.landmark[idx].y * image1.shape[0])) for idx in left_cheek_indices]

            # Convert lip_points to a NumPy array
            left_cheek_array = np.array(left_cheek_points, np.int32)

            # Convex hull
            hull_leftcheek = cv2.convexHull(left_cheek_array)

            # Create a mask for the convex hull
            mask = np.zeros(image2.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull_leftcheek, (255, 255, 255))  # Fill convex hull with white

            # Apply the mask to the image
            masked_image = cv2.bitwise_and(image2, image2, mask=mask)

            # Invert the mask
            inverted_mask = cv2.bitwise_not(mask)

            # Create white background
            white_background = np.ones_like(image2, dtype=np.uint8) * 0

            # Apply inverted mask to the white background
            inverted_masked_background = cv2.bitwise_and(white_background, white_background, mask=inverted_mask)

            # Combine masked image with inverted masked background
            result_image = cv2.add(masked_image, inverted_masked_background)

        # Save the result image
        cv2.imwrite('Results/leftCheek.png', result_image)

    # Check if landmarks are found
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Landmark indices for lips, eyes, eyebrow
            right_cheek_indices = [345, 346, 347, 330, 266, 423, 322, 364, 435, 447]

            # Extract landmark points
            right_cheek_points = [(int(face_landmarks.landmark[idx].x * image2.shape[1]),
                                   int(face_landmarks.landmark[idx].y * image2.shape[0])) for idx in right_cheek_indices]

            # Convert lip_points to a NumPy array
            right_cheek_array = np.array(right_cheek_points, np.int32)

            # Convex hull
            hull_rightcheek = cv2.convexHull(right_cheek_array)

            # Create a mask for the convex hull
            mask = np.zeros(image2.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull_rightcheek, (255, 255, 255))  # Fill convex hull with white

            # Apply the mask to the image
            masked_image = cv2.bitwise_and(image2, image2, mask=mask)

            # Invert the mask
            inverted_mask = cv2.bitwise_not(mask)

            # Create white background
            white_background = np.ones_like(image2, dtype=np.uint8) * 0

            # Apply inverted mask to the white background
            inverted_masked_background = cv2.bitwise_and(white_background, white_background, mask=inverted_mask)

            # Combine masked image with inverted masked background
            result_image = cv2.add(masked_image, inverted_masked_background)

        # Save the result image
        cv2.imwrite('Results/rightCheek.png', result_image)

    # Check if landmarks are found
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Landmark indices for lips, eyes, eyebrow
            # forehead_indices = [103, 67, 109, 10, 338, 297, 333, 299, 337, 151, 108, 69, 69, 104]
            forehead_indices = [67, 69, 109, 10, 338, 337, 151, 108, 299, 297]

            # Extract landmark points
            forehead_points = [(int(face_landmarks.landmark[idx].x * image3.shape[1]),
                                int(face_landmarks.landmark[idx].y * image3.shape[0])) for idx in forehead_indices]

            # Convert lip_points to a NumPy array
            forehead_array = np.array(forehead_points, np.int32)


            # Convex hull
            hull_forehead = cv2.convexHull(forehead_array)

            # Create a mask for the convex hull
            mask = np.zeros(image2.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull_forehead, (255, 255, 255))  # Fill convex hull with white

            # Apply the mask to the image
            masked_image = cv2.bitwise_and(image2, image2, mask=mask)

            # Invert the mask
            inverted_mask = cv2.bitwise_not(mask)

            # Create white background
            white_background = np.ones_like(image2, dtype=np.uint8) * 0

            # Apply inverted mask to the white background
            inverted_masked_background = cv2.bitwise_and(white_background, white_background, mask=inverted_mask)

            # Combine masked image with inverted masked background
            result_image = cv2.add(masked_image, inverted_masked_background)

        # Save the result image
        cv2.imwrite('Results/forehead.png', result_image)

# Check if landmarks are found
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Landmark indices for lips, eyes, eyebrow
            chin_indices = [83, 18, 313, 418, 262, 369, 400, 377, 152, 148, 176, 140, 32, 194]

            # Extract landmark points
            chin_points = [(int(face_landmarks.landmark[idx].x * image4.shape[1]),
                            int(face_landmarks.landmark[idx].y * image4.shape[0])) for idx in chin_indices]

            # Convert lip_points to a NumPy array
            chin_array = np.array(chin_points, np.int32)

            # Convex hull
            hull_chin = cv2.convexHull(chin_array)

            # Create a mask for the convex hull
            mask = np.zeros(image2.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull_chin, (255, 255, 255))  # Fill convex hull with white

            # Apply the mask to the image
            masked_image = cv2.bitwise_and(image2, image2, mask=mask)

            # Invert the mask
            inverted_mask = cv2.bitwise_not(mask)

            # Create white background
            white_background = np.ones_like(image2, dtype=np.uint8) * 0

            # Apply inverted mask to the white background
            inverted_masked_background = cv2.bitwise_and(white_background, white_background, mask=inverted_mask)

            # Combine masked image with inverted masked background
            result_image = cv2.add(masked_image, inverted_masked_background)

        # Save the result image
        cv2.imwrite('Results/chin.png', result_image)

def finalresult(imagePath):
    # Baca gambar
    image = cv2.imread(imagePath)  # Ganti 'your_image_path.jpg' dengan path gambar Anda

    # Konversi ke format HSV
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tentukan rentang warna yang ingin Anda deteksi (misalnya, biru)
    lower = (1, 1, 1)
    upper = (255, 255, 255)

    # Masking gambar untuk menemukan warna yang ditentukan
    mask = cv2.inRange(img, lower, upper)

    # Temukan kontur pada gambar mask
    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Temukan posisi semua kontur
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 500:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)  # Gambar kotak di sekitar objek yang dideteksi

                # Potong bagian gambar yang terdeteksi
                detected_object = image[y:y+h, x:x+w]

                # Resize gambar menjadi ukuran 1:1 (persegi)
                # size = max(detected_object.shape[0], detected_object.shape[1])
                # square_image = cv2.resize(detected_object, (500, 500))

                # Ambil nama file dari imagePath
                file_name = imagePath.split('/')[-1]  # Mengambil bagian terakhir dari path sebagai nama file
                # Simpan gambar yang terdeteksi dengan nama yang sama seperti gambar input
                # cv2.imwrite("results/" + file_name, square_image)
                cv2.imwrite("results/" + file_name, detected_object)

processFace(imagePath= 'Assets/Foto.jpg')
featureFace(imagePath= 'Results/faceExtraction.png')
finalresult(imagePath= 'Results/faceExtraction.png')
finalresult(imagePath= 'Results/chin.png')
finalresult(imagePath= 'Results/forehead.png')
finalresult(imagePath= 'Results/leftCheek.png')
finalresult(imagePath= 'Results/rightCheek.png')
# extract_faces(imagePath= 'Results/leftCheek.png')

