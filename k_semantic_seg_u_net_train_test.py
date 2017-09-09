#############
#
# Kaggle Carvana Image Masking Challenge
# https://www.kaggle.com/c/carvana-image-masking-challenge
#
#   0. I got the technique from here.
#   https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge
#   1. Semantic Segmentation using U-net
#   2. Keras
#
#
# ---Kaggle_Carvana_IM_Masking.py
#  |
#  --Kaggle_Car_Data---train---train---training images (.jpg)
#                    |       |
#                    |       --train_masks.csv
#                    |
#                    --train_masks---train_masks---training masks (.gif -> .png)
#                    |
#                    --test---test---test images (.jpg)
#                    |      |
#                    |      --sample_submission.csv
#                    |
#                    --weights---(best_weights.hdf5)
#                    |
#                    --submit
#
#
#############


## Modules 
import cv2
import numpy as np
import panda as pd
import os
import sys
# from PIL import Image ### It is useful for .gif to .png conversion. 
from sklearn.model_selection import train_test_split
### Using TensorFlow Backend. 
import keras.backend as K
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop


### Dice Loss Functions
# https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/blob/master/model/losses.py
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

### Load Data ###
### Read CSV
df_trn_val = pd.read_csv('Kaggle_Car_Data/train/train_masks.csv')
### Extract IDs
ids_trn_val = df_trn_val['img'].map(lambda s: s.split('.')[0])

### Train Data IDs and Validation Data IDs
ids_train_split, ids_valid_split = train_test_split(ids_trn_val, test_size = 0.2, random_state = 42)

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))
### 4070 vs. 1018

### Paths 
kaggle_train_path = 'Kaggle_Car_Data/train/train'
kaggle_train_mask_path = 'Kaggle_Car_Data/train_masks/train_masks'
kaggle_test_path = 'Kaggle_Car_Data/test/test'

### Data Augmentation #1
def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

### Data Augmentation #2
def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape
        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])
        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
    return image, mask

### Data Augmentation #3
def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask

### Train Data Generator
def train_generator():
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch.values:
                img = cv2.imread(kaggle_train_path + '/{}.jpg'.format(id))
                img = cv2.resize(img, (input_size, input_size))
                mask = cv2.imread(kaggle_train_mask_path + '/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_size, input_size))
                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                img, mask = randomShiftScaleRotate(img, mask,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-0, 0))
                img, mask = randomHorizontalFlip(img, mask)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

##########################
trn_img = []
trn_mask = []
#cv2.imread can not read .gif!
#img = cv2.imread(kaggle_train_path + '/{}.jpg'.format(ids_train_split[0]))
#mask = cv2.imread(kaggle_train_mask_path + '/{}_mask.png'.format(ids_train_split[0]), cv2.IMREAD_GRAYSCALE)

for id in ids_train_split[0:100]:
    img = cv2.imread(kaggle_train_path + '/{}.jpg'.format(id))
    img = cv2.resize(img, (input_size, input_size))
    mask = cv2.imread(kaggle_train_mask_path + '/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (input_size, input_size))
    img = randomHueSaturationValue(img,
                                    hue_shift_limit=(-50, 50),
                                    sat_shift_limit=(-5, 5),
                                    val_shift_limit=(-15, 15))
    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.0625, 0.0625),
                                       scale_limit=(-0.1, 0.1),
                                    rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    mask = np.expand_dims(mask, axis=2)
    ### imshow
    ### A little animation --start--
    cv2.imshow("image 1", img)
    cv2.imshow("image 2", mask)
    cv2.waitKey(1000)
    ### A little animation --end--
    trn_img.append(img)
    trn_mask.append(mask)

cv2.destroyAllWindows()

trn_img = np.array(trn_img, np.float32) / 255
trn_mask = np.array(trn_mask, np.float32) / 255
############################


### Validation Data Generator
def valid_generator():
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                img = cv2.imread(kaggle_train_path + '/{}.jpg'.format(id))
                img = cv2.resize(img, (input_size, input_size))
                mask = mask = cv2.imread(kaggle_train_mask_path + '/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_size, input_size))
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

### U-net
def get_unet_128(input_shape=(128, 128, 3),num_classes=1):
    inputs = Input(shape=input_shape)
    # 128
    down1 = Conv2D(64, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64
    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32
    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16
    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8
    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center
    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128
    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)
    model = Model(inputs=inputs, outputs=classify)
    return model

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='logs')]


### 
model = get_unet_128()
model.summary()
model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff]) 

input_size = 128
batch_size = 16
epochs = 10

'''
I am not familiar with model.fit_generator.
callbacks are new to me. 
'''
model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))

########################################
### Test - Submit
### Make sure you already have best_weights.hdf5

df_test = pd.read_csv('Kaggle_Car_Data/test/sample_submission.csv')
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

test_names = []
for id in ids_test:
    test_names.append('{}.jpg'.format(id))


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

rles = []

model.load_weights(filepath='weights/best_weights.hdf5')

print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))

### import tqdm
from tqdm import tqdm
threshold = 0.5

#
# Can this model segment a pickup truck? 
for i0 in range(0,10000):
    if ids_test.values[i0] == '0bdb8b1cba05_01':
        print(i0)
# 4320 is a pickup truck. 

#for start in tqdm(range(0, len(ids_test), batch_size)):
# There are 100,000 testing images...
# tqdm
for start in tqdm(range(4320, 4320 + 32, batch_size)):
    x_batch = []
    end = min(start + batch_size, len(ids_test))
    ids_test_batch = ids_test[start:end]
    for id in ids_test_batch.values:
        img = cv2.imread(kaggle_test_path + '/{}.jpg'.format(id))
        img = cv2.resize(img, (input_size, input_size))
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    for pred in preds:
        ### For the competition, pred needs to be resized to the original width and height. 
        #prob = cv2.resize(pred, (orig_width, orig_height))
        prob = pred
        mask = prob > threshold
        ### imshow
        ### A little animation -start-
        cv2.imshow('image',prob)
        cv2.waitKey(1000) ### 100 msec for each
        ### A little animation -end-
        rle = run_length_encode(mask)
        rles.append(rle)

cv2.destroyAllWindows()

### Submit to Kaggle
print("Generating submission file...")
df = pd.DataFrame({'img': test_names, 'rle_mask': rles})
df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')


'''
gif2png.py
from PIL import Image
kaggle_train_mask_path = 'Kaggle_Car_Data/train_masks/train_masks'

for filename in os.listdir(kaggle_train_mask_path):
    if filename.endswith(".gif"):
        print(filename)
        im_mask = Image.open(kaggle_train_mask_path + '/' + filename)
        png_filename = filename[:-4] + '.png'
        im_mask.save(kaggle_train_mask_path + '/' + png_filename,"PNG")
,,,
