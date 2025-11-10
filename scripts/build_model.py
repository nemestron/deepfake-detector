import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import os
import numpy as np

class DeepfakeDetector:
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def create_cnn_model(self):
        """Create a CNN model for deepfake detection"""
        print("🧠 BUILDING CNN MODEL ARCHITECTURE...")
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fifth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(1, activation='sigmoid')  # Binary classification: fake vs real
        ])
        
        self.model = model
        return model
    
    def create_advanced_model(self):
        """Create a more advanced model with residual connections"""
        print("🚀 BUILDING ADVANCED MODEL ARCHITECTURE...")
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Residual block 1
        residual = x
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, layers.Conv2D(64, (1, 1))(residual)])  # Projection shortcut
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Residual block 2
        residual = x
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, layers.Conv2D(128, (1, 1))(residual)])
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Residual block 3
        residual = x
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, layers.Conv2D(256, (1, 1))(residual)])
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        
        # Global pooling and dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        self.model = model
        return model
    
    def compile_model(self, model_type='advanced'):
        """Compile the model with appropriate settings"""
        if self.model is None:
            if model_type == 'advanced':
                self.create_advanced_model()
            else:
                self.create_cnn_model()
        
        print("⚙️ COMPILING MODEL...")
        
        # Custom optimizer with learning rate scheduling
        initial_learning_rate = 0.001
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        optimizer = optimizers.Adam(learning_rate=lr_schedule)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                'precision',
                'recall',
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        print("✅ MODEL COMPILED SUCCESSFULLY!")
        return self.model
    
    def model_summary(self):
        """Print model summary and save architecture"""
        if self.model is None:
            self.compile_model()
        
        print("\n📊 MODEL ARCHITECTURE SUMMARY:")
        self.model.summary()
        
        # Save model architecture diagram
        try:
            tf.keras.utils.plot_model(
                self.model,
                to_file='D:\\A Image Classification\\deepfake_detector\\models\\model_architecture.png',
                show_shapes=True,
                show_layer_names=True
            )
            print("✅ Model architecture diagram saved!")
        except Exception as e:
            print(f"⚠️ Could not save architecture diagram: {e}")
        
        return self.model.summary()

def test_model_creation():
    """Test function to verify model creation works"""
    print("🧪 TESTING MODEL CREATION...")
    
    # Create models directory
    os.makedirs('D:\\A Image Classification\\deepfake_detector\\models', exist_ok=True)
    
    # Test basic CNN model
    detector = DeepfakeDetector()
    basic_model = detector.create_cnn_model()
    print("✅ Basic CNN model created successfully!")
    
    # Test advanced model
    advanced_model = detector.create_advanced_model()
    print("✅ Advanced model created successfully!")
    
    # Test compilation
    compiled_model = detector.compile_model()
    print("✅ Model compilation successful!")
    
    # Generate summary
    detector.model_summary()
    
    # Save model information
    model_info = {
        'input_shape': (256, 256, 3),
        'total_parameters': compiled_model.count_params(),
        'layers_count': len(compiled_model.layers),
        'model_type': 'Advanced CNN with Residual Connections',
        'optimizer': 'Adam with Exponential Decay',
        'loss_function': 'binary_crossentropy',
        'metrics': ['accuracy', 'precision', 'recall', 'auc']
    }
    
    with open('D:\\A Image Classification\\deepfake_detector\\models\\model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("📁 Model information saved to models/model_info.json")
    print("🎉 MODEL CREATION TEST COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    test_model_creation()