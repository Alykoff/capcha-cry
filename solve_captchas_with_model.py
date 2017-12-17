from keras.models import load_model
import numpy as np
import cv2
import pickle
from PIL import ImageFont
import secrets
from PIL import Image
import requests
from io import BytesIO

image_url = secrets.image_url
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"


# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

ImageFont.truetype('./Arial.ttf', 60)

while True:
    image_file = 'tmp.png'
    image = None
    try:
        response = requests.get(image_url, verify=False)
        image = np.array(Image.open(BytesIO(response.content)))
        print('Load image')
    except Exception as e:
        print(e)
    real = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = image[10:55, 10:185]
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    h, w = thresh.shape
    delta = int(w / 6)
    images = []
    for step in range(0, 6):
        images.append(thresh[0:h, step * delta: (step + 1) * delta])
    output = cv2.merge([real])
    predictions = []

    x_0 = 5
    y_0 = 5
    img_width = 29
    img_height = 45
    step = 0

    for img in images:
        # Turn the single image into a 4d list of images to make Keras happy
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)

        # Ask the neural network to make a prediction
        prediction = model.predict(img)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        x = x_0 + (step * img_width)
        x_next = x_0 + ((step + 1) * img_width)
        cv2.rectangle(output, (x, y_0), (x_next + 4, y_0 + img_height + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y_0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1)
        step += 1

    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))

    cv2.imshow("Output", output)
    print('wait press')
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    cv2.waitKey()