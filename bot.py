import pathlib
import tensorflow as tf
import telebot
from telegram.ext import *
from io import BytesIO
import cv2
import numpy as np

TOKEN = ("6279536990:AAGapiV3hKEL-l9ndsh1PrdKH05LkwOdwxU")

data_dir = pathlib.Path('/Users/user/Downloads/One Piece/Data')

image_count = len(list(data_dir.glob('*/*')))


list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

import numpy as np

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))


val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

len(train_ds)

import os
import matplotlib.pyplot as plt

batch_size = 64
img_height = 224
img_width = 224
AUTOTUNE = tf.data.AUTOTUNE


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)

def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.title(class_names[label])
    plt.axis("off")

num_classes = len(class_names)

model = tf.keras.models.load_model('My_OP_model.h5')

def start(update, context):
    update.message.reply_text("Hi, It's One Piece Image Classifier!")

def help(update, context):
    update.message.reply_text("""
    /start - Starts conversation
    /help - Shows this message
    /train -Train neural network
    """)

def train(update, context):
    update.message.reply_text("Model is train...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs= 5,
        steps_per_epoch=20
    )
    update.message.reply_text("Done, you can send.")

def handle_message(update, context):
    update.message.reply_text("Pls train the AI")

def handle_photo(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    prediction = model.predict(np.array([img / 3]))
    update.message.reply_text(f"I think it's {class_names[np.argmax(prediction)]}")

updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("help", help))
dp.add_handler(CommandHandler("train", train))
dp.add_handler(MessageHandler(Filters.text, handle_message))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))

updater.start_polling()
updater.idle()