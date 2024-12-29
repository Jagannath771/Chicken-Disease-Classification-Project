from CNNClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        tf.config.run_functions_eagerly(True)
        self.model = None
        self.train_generator = None
        self.valid_generator = None
    
    def get_base_model(self):
        # Load the pre-trained model
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        
        # Define metrics explicitly
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy')
        ]
        
        # Recompile the model with fresh optimizer and explicit metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics
        )
    
    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode='categorical'  # Explicitly set class_mode
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        if self.model is None:
            raise ValueError("Model not loaded. Call get_base_model() first.")
            
        if self.train_generator is None or self.valid_generator is None:
            raise ValueError("Data generators not initialized. Call train_valid_generator() first.")
        
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        try:
            history = self.model.fit(
                x=self.train_generator,  # Use x= explicitly
                validation_data=self.valid_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                callbacks=callback_list
            )
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        
        return history