from model_dogs_breeds.imports import *
from model_dogs_breeds.setup import *


def load_and_preprocess_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0

    # One-hot encode the label
    label = tf.one_hot(label, num_classes)

    return img, label


def create_dataset(df, train_dir, batch_size):
    filenames = df['id'].apply(lambda x: os.path.join(train_dir, f"{x}.jpg")).values
    labels = df['breed'].map(breed_to_label).values
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def load_and_preprocess_test_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img


if __name__ == '__main__':
    base_model = MobileNetV2(weights='imagenet', include_top=False)  # загрузка MobileNetV2

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])  # компиляция модели

    model.summary()  # показать данные модели

    # обучение модели
    batches_per_epoch = math.ceil(len(labels) / batch_size)
    for epoch in range(10):
        print('Epoch', epoch + 1)
        pbar = tqdm(total=batches_per_epoch)
        labels = labels.sample(frac=1).reset_index(drop=True)
        dataset = create_dataset(labels, train_dir, batch_size)
        for x_batch, y_batch in dataset:
            model.train_on_batch(x_batch, y_batch.numpy())
            pbar.update(1)
        pbar.close()

    # тестирование модели
    test_filenames = os.listdir(test_dir)

    predictions_list = []

    for filename in tqdm(test_filenames):
        img_path = os.path.join(test_dir, filename)
        img = load_and_preprocess_test_image(img_path)
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_label = breeds[np.argmax(prediction)]
        predictions_list.append((filename.split('.')[0], predicted_label))

    submission_df = pd.DataFrame(predictions_list, columns=['id', 'breed'])

    submission_df.to_csv('Dog_breed_Submission.csv', index=False)

    # созранение модели в файл
    model.save('model_dogs_breeds.h5')
