import os.path
import cv2
import glob

CAPTCHA_IMAGE_FOLDER = 'test-examples-images'
OUTPUT_FOLDER = 'extracted_letter_images'

captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, '*'))
counts = {}

def divide_img(dims):
    (x, y, w, h) = dims
    if h < 30 or w < 15:
        return []
    elif w > 80 or (w / h > 1.25):
        return []
    else:
        return [(x, y, w, h)]

for (i, captcha_image_file) in enumerate(captcha_image_files):
    print('[INFO] processing image {}/{}'.format(i + 1, len(captcha_image_files)))

    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    image = cv2.imread(captcha_image_file)
    image = image[10:55, 10:185]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    cv2.imwrite('file1.png', gray)

    h, w = gray.shape
    delta = int(w / 6)
    images = []
    for step in range(0, 6):
        letter_text = captcha_correct_text[step]
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, '{}.png'.format(str(count).zfill(6)))
        im = gray[0:h, step * delta: (step + 1) * delta]
        cv2.imwrite(p, im)
        counts[letter_text] = count + 1