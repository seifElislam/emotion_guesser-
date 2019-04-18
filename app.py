""""
"""
import os
import datetime
import time
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import base64

# emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotions = {0: (255, 0, 0), 1: (128, 0, 1), 2: (0, 0, 139), 3: (255, 255, 0), 4: (165, 42, 42), 5: (238, 13, 230), 6: (176, 224, 230)}
model, graph = None, None


def load_model():
    """

    :return:
    """
    import tensorflow as tf
    global model, graph
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    model = Sequential()
    model.add(Conv2D(filters=2, kernel_size=2, padding='same', activation='relu',
                     input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=4, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=8, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=['accuracy'])
    # load the weights that yielded the best validation accuracy
    model.load_weights(os.path.join(APP_ROOT, 'weights.hdf5'))
    graph = tf.get_default_graph()


def predict_emotion(img):
    """

    :param img:
    :return:
    """
    # import tensorflow as tf
    # graph = tf.get_default_graph()
    with graph.as_default():
        possibilities = (model.predict(img)).tolist()
    emotion_index = possibilities[0].index(max(possibilities[0]))
    return emotions[emotion_index]


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = b'_5#y2L"F4Q8z\n\xec]/'
app.load = load_model()


def remove_old_imgs(delta=1):
    """

    :param delta:
    :return:
    """
    dir_to_search = UPLOAD_FOLDER
    for dirpath, dirnames, filenames in os.walk(dir_to_search):
        for file in filenames:
            curpath = os.path.join(dirpath, file)
            file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(curpath))
            if datetime.datetime.now() - file_modified > datetime.timedelta(minutes=5):
                os.remove(curpath)


def allowed_file(filename):
    """

    :param filename:
    :return:
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    """

    :return:
    """
    if request.method == 'POST':
        remove_old_imgs()
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # filename = "{}_{}".format(str(time.time()), secure_filename(file.filename))
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file', filename=filename))
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            detected_img = detect_faces_v2(image)
            image_content = cv2.imencode('.jpg', image)[1].tostring()
            encoded_image = base64.encodebytes(image_content)
            base64_img = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
            return render_template('home.html', img=base64_img)
            # return redirect(url_for('detect_img', filename=filename))
    return render_template('home.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """

    :param filename:
    :return:
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/about')
def about_page():
    """

    :return:
    """
    return render_template('about.html')


def detect_faces(filename):
    """

    :param filename:
    :return:
    """
    face_cascade = cv2.CascadeClassifier(os.path.join(APP_ROOT, 'haarcascade_frontalface_alt.xml'))
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find faces in image
    faces = face_cascade.detectMultiScale(gray)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get bounding box for each detected face
    for (x, y, w, h) in faces:
        if w < 5:
            continue
        roi = gray[y:y + h, x:x + w]
        resized = cv2.resize(roi, (48, 48))
        sample = resized.reshape(1, 48, 48, 1)
        emotion = predict_emotion(sample)
        size = int(w/100)
        cv2.putText(img, emotion, (x, y), font, size, (0, 128, 255), thickness=2)
        # add bounding box to color image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)

    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), img)


def detect_faces_v2(img):
    """

    :param img:
    :return:
    """
    face_cascade = cv2.CascadeClassifier(os.path.join(APP_ROOT, 'haarcascade_frontalface_alt.xml'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find faces in image
    faces = face_cascade.detectMultiScale(gray)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get bounding box for each detected face
    for (x, y, w, h) in faces:
        if w < 5:
            continue
        roi = gray[y:y + h, x:x + w]
        resized = cv2.resize(roi, (48, 48))
        sample = resized.reshape(1, 48, 48, 1)
        emotion = predict_emotion(sample)
        # size = int(w/100)
        # cv2.putText(img, emotion, (x, y), font, size, (0, 128, 255), thickness=2)
        # add bounding box to color image
        cv2.rectangle(img, (x, y), (x + w, y + h), emotion, 2)

    # cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), img)
    return img


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
