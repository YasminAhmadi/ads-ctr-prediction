"""
TensorFlow Model for CTR Prediction
Implements a deep neural network for click-through rate prediction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np


def build_ctr_model(input_dim, learning_rate=0.001, dropout_rate=0.3):
    """
    Build a deep neural network for CTR prediction.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    learning_rate : float
        Learning rate for Adam optimizer
    dropout_rate : float
        Dropout rate for regularization
        
    Returns:
    --------
    model : keras.Model
        Compiled TensorFlow model
    """
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # First hidden layer
        layers.Dense(256, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        
        # Second hidden layer
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        
        # Third hidden layer
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        
        # Fourth hidden layer
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.AUC(name='auc_roc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.BinaryAccuracy(name='accuracy')
        ]
    )
    
    return model


def build_lightweight_model(input_dim, learning_rate=0.001):
    """
    Build a lightweight model for ablation studies.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    learning_rate : float
        Learning rate for Adam optimizer
        
    Returns:
    --------
    model : keras.Model
        Compiled TensorFlow model
    """
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(16, activation='relu'),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc_roc')]
    )
    
    return model


class CTRPredictor:
    """Wrapper class for CTR model training and evaluation"""
    
    def __init__(self, model, verbose=1):
        self.model = model
        self.verbose = verbose
        self.history = None
        
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model.
        
        Parameters:
        -----------
        X_train, y_train : array-like
            Training features and labels
        X_val, y_val : array-like
            Validation features and labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        history : History
            Training history
        """
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_auc_roc',
            patience=10,
            restore_best_weights=True,
            mode='max'
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=self.verbose
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.
        
        Parameters:
        -----------
        X_test, y_test : array-like
            Test features and labels
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metric_names = self.model.metrics_names
        
        metrics = {name: value for name, value in zip(metric_names, results)}
        
        # Get predictions for additional metrics
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        return metrics, y_pred, y_pred_binary
    
    def predict(self, X):
        """Generate predictions"""
        return self.model.predict(X, verbose=0)
