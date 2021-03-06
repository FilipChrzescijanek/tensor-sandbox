import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K
	
def a1(input_shape, num_classes):
	model = Sequential()
	
	model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(num_classes, activation='softmax'))
	
	return model

def a2(input_shape, num_classes):
	input_img = Input(shape=input_shape)

	sequential = Conv2D(32, (5, 5), padding='same', activation='relu')(input_img)
	sequential = MaxPooling2D(pool_size=(2, 2))(sequential)
	
	sequential = Conv2D(64, (5, 5), padding='same', activation='relu')(sequential)
	sequential = MaxPooling2D(pool_size=(2, 2))(sequential)
	
	tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(sequential)
	tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

	tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(sequential)
	tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

	tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(sequential)
	tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
	
	if K.image_data_format() == 'channels_first':
		channel_axis = 1
	else:
		channel_axis = 3
	
	output = concatenate([tower_1, tower_2, tower_3], axis=channel_axis)
	output = Flatten()(output)
	output = Dense(1024, activation='relu')(output)
	output = Dropout(0.4)(output)
	out = Dense(num_classes, activation='softmax')(output)

	model = Model(inputs=input_img, outputs=out)
	return model
	
def a3(input_shape, num_classes):
	model = Sequential()
	
	model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(num_classes, activation='softmax'))
	
	return model
	
def a4(input_shape, num_classes):
	model = Sequential()
	
	model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(num_classes, activation='softmax'))
	
	return model

def a5(input_shape, num_classes):
	input_img = Input(shape=input_shape)

	sequential = Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)
	sequential = MaxPooling2D(pool_size=(2, 2))(sequential)
	
	sequential = Conv2D(64, (3, 3), padding='same', activation='relu')(sequential)
	sequential = MaxPooling2D(pool_size=(2, 2))(sequential)
	
	tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(sequential)
	tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

	tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(sequential)
	tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

	tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(sequential)
	tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)
	
	if K.image_data_format() == 'channels_first':
		channel_axis = 1
	else:
		channel_axis = 3
	
	output = concatenate([tower_1, tower_2, tower_3], axis=channel_axis)
	output = Flatten()(output)
	output = Dense(1024, activation='relu')(output)
	output = Dropout(0.4)(output)
	out = Dense(num_classes, activation='softmax')(output)

	model = Model(inputs=input_img, outputs=out)
	return model

input_shape = (56, 56, 1)
num_classes = 3
epochs = 10
steps = 2000
val_steps = 800
lrate = 0.01
batch_size = 32
decay = lrate / epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay)

model = a1(input_shape, num_classes)
#model = a2(input_shape, num_classes)
#model = a3(input_shape, num_classes)
#model = a4(input_shape, num_classes)
#model = a5(input_shape, num_classes)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

datagen = ImageDataGenerator(
	rotation_range=360, 
	zoom_range=0.05, 
	width_shift_range=0.05,
	height_shift_range=0.05,
	horizontal_flip=True, 
	vertical_flip=True,
	rescale=1./255)

train_generator = datagen.flow_from_directory(
	'data/train', 
	target_size=(56, 56),
	color_mode='grayscale',
	batch_size=batch_size)

validation_generator = datagen.flow_from_directory(
	'data/eval', 
	target_size=(56, 56),
	color_mode='grayscale',
	batch_size=batch_size)

#early_stopping = EarlyStopping()
	
model.fit_generator(
	train_generator,  
	steps_per_epoch=steps, 
	epochs=epochs, 
	validation_data=validation_generator,
	validation_steps=val_steps)
	#callbacks=[early_stopping])
	
model.save('higseg.h5')