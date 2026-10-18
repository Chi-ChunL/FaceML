from load_data import create_expanded_training_datasets

train_dataset, validation_dataset = create_expanded_training_datasets()

print("Class names:", train_dataset.class_names)
print("Training images:", len(train_dataset.file_paths))
print("Validation images:", len(validation_dataset.file_paths))

images, labels = next(iter(train_dataset))

print("Image shape:", images.shape)
print("Label shape:", labels.shape)
print("Image dtype:", images.dtype)
print("Pixel range:", images.numpy().min(), images.numpy().max())
