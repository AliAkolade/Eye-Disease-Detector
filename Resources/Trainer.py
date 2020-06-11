import os
import h5py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

# TRAINING DATA
base_dir = os.path.abspath(os.curdir)
train_dir = os.path.join(base_dir, "Data\\Train")
test_dir = os.path.join(base_dir, "Data\\Test")

# 'Exophthalmos', 'Cataracts', 'Strabismus', 'Glaucoma', 'Uveitis',
# 'Conjunctivitis', 'Blepharitis', 'Keratitis', 'Pterygium'

# Directory with our training pictures
train_exophthalmos_dir = os.path.join(train_dir, 'Exophthalmos')
train_Cataracts_dir = os.path.join(train_dir, 'Cataracts')
train_Strabismus_dir = os.path.join(train_dir, 'Strabismus')
train_Glaucoma_dir = os.path.join(train_dir, 'Glaucoma')
train_Uveitis_dir = os.path.join(train_dir, 'Uveitis')
train_Conjunctivitis_dir = os.path.join(train_dir, 'Conjunctivitis')
train_Blepharitis_dir = os.path.join(train_dir, 'Blepharitis')
train_Keratitis_dir = os.path.join(train_dir, 'Keratitis')
train_Pterygium_dir = os.path.join(train_dir, 'Pterygium')

# Directory with our validation pictures
test_exophthalmos_dir = os.path.join(test_dir, 'Exophthalmos')
test_Cataracts_dir = os.path.join(test_dir, 'Cataracts')
test_Strabismus_dir = os.path.join(test_dir, 'Strabismus')
test_Glaucoma_dir = os.path.join(test_dir, 'Glaucoma')
test_Uveitis_dir = os.path.join(test_dir, 'Uveitis')
test_Conjunctivitis_dir = os.path.join(test_dir, 'Conjunctivitis')
test_Blepharitis_dir = os.path.join(test_dir, 'Blepharitis')
test_Keratitis_dir = os.path.join(test_dir, 'Keratitis')
test_Pterygium_dir = os.path.join(test_dir, 'Pterygium')

# All images will be rescaled by 1./255.
# train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1.6 / 255.)
# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  batch_size=20,
                                                  class_mode='binary',
                                                  target_size=(150, 150))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

model.fit_generator(train_generator,
                    validation_data=test_generator,
                    steps_per_epoch=100,
                    epochs=15,
                    validation_steps=50,
                    verbose=1)

model.save('Model.h5py')
