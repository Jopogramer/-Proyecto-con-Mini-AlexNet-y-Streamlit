import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score
from model import MiniAlexNet
from PIL import Image

def save_dataset_to_dir(dataset, split_name, base_output_dir):
    for i, (image, label) in enumerate(tfds.as_numpy(dataset)):
        label_name = label_names[label]
        label_dir = os.path.join(base_output_dir, split_name, label_name)
        os.makedirs(label_dir, exist_ok=True)
        image_path = os.path.join(label_dir, f"{split_name}_{i}.jpg")
        img = Image.fromarray(image)
        img.save(image_path)

print("📥 Descargando dataset...")
(ds_train, ds_test), ds_info = tfds.load(
    'rock_paper_scissors',
    split=['train', 'test'],
    with_info=True,
    as_supervised=True
)

label_names = ds_info.features['label'].names

output_base = 'data/rock_paper_scissors'

if os.path.exists(output_base):
    shutil.rmtree(output_base)

print("💾 Guardando imágenes como archivos...")
save_dataset_to_dir(ds_train, 'train', output_base)
save_dataset_to_dir(ds_test, 'validation', output_base)


train_dir = os.path.join(output_base, 'train')
val_dir = os.path.join(output_base, 'validation')

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
)
val_data = val_gen.flow_from_directory(
    val_dir, target_size=(150, 150), batch_size=32, class_mode='categorical', shuffle=False
)

model = MiniAlexNet.build(input_shape=(150, 150, 3), num_classes=3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 Entrenando el modelo...")
history = model.fit(train_data, epochs=20, validation_data=val_data)


print("🔍 Evaluando el modelo...")
val_preds = model.predict(val_data)
y_true = val_data.classes
y_pred = val_preds.argmax(axis=1)

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"\n✅ Accuracy en validación: {acc:.4f}")
print("📊 Matriz de confusión:\n", cm)

os.makedirs('models', exist_ok=True)
model.save('models/mini_alexnet_rps.h5')
print("💾 Modelo guardado en 'models/mini_alexnet_rps.h5'.")

plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
print("📈 Gráfico guardado como 'accuracy_plot.png'.")
