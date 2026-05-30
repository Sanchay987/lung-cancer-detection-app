"""
Baseline Model Comparison for MTech Research
Trains and compares multiple architectures on lung nodule classification

Models:
1. ResNet50
2. EfficientNetB3
3. Vision Transformer (ViT)
4. MobileNetV2 (base - no attention)
5. MobileNetV2 + Channel Attention (proposed)

Author: MTech Research Project
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, models
from keras.applications import (
    ResNet50,
    EfficientNetB3,
    MobileNetV2
)
from keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    multiply,
    Reshape,
    MultiHeadAttention,
    LayerNormalization,
    Add
)
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
CONFIG = {
    'image_size': (96, 96),
    'batch_size': 32,
    'initial_epochs': 20,
    'fine_tune_epochs': 10,
    'initial_lr': 0.001,
    'fine_tune_lr': 0.00001,
    'l2_reg': 0.0001,
    'dropout_rate': 0.5,
    'attention_ratio': 8,
    'patience': 5,
    'test_size': 0.3,
    'val_split': 0.5,
    'random_state': 42
}

# Create results directory
os.makedirs('comparison_results', exist_ok=True)
os.makedirs('comparison_results/models', exist_ok=True)
os.makedirs('comparison_results/plots', exist_ok=True)


def load_and_preprocess_data():
    """Load and preprocess CT scan data from HDF5 file"""
    print("\n" + "="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)

    with h5py.File('all_patches.hdf5', 'r') as f:
        images = f['ct_slices'][:]
        labels = f['slice_class'][:]

    print(f"Original shape: {images.shape}")

    # Resize and convert to RGB
    target_size = CONFIG['image_size']
    images = np.stack([images] * 3, axis=-1)
    images = tf.image.resize(images, target_size)
    images = images.numpy().astype('float32') / 255.0
    labels = labels.reshape(-1)

    print(f"Preprocessed shape: {images.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=labels
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test,
        test_size=CONFIG['val_split'],
        random_state=CONFIG['random_state'],
        stratify=y_test
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def channel_attention(input_feature, ratio=8, name=''):
    """Squeeze-and-Excitation Channel Attention"""
    channel = input_feature.shape[-1]

    se = GlobalAveragePooling2D(name=f'{name}_gap')(input_feature)
    se = Dense(channel // ratio, activation='relu', use_bias=False, name=f'{name}_fc1')(se)
    se = Dense(channel, activation='sigmoid', use_bias=False, name=f'{name}_fc2')(se)
    se = Reshape((1, 1, channel), name=f'{name}_reshape')(se)

    return multiply([input_feature, se], name=f'{name}_scale')


def build_resnet50():
    """Build ResNet50 baseline"""
    print("\nBuilding ResNet50...")

    base_model = ResNet50(
        input_shape=(*CONFIG['image_size'], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*CONFIG['image_size'], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(CONFIG['l2_reg']))(x)
    x = Dropout(CONFIG['dropout_rate'])(x)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(CONFIG['l2_reg']))(x)

    model = models.Model(inputs, outputs, name='ResNet50')
    return model, base_model


def build_efficientnet_b3():
    """Build EfficientNetB3 baseline"""
    print("\nBuilding EfficientNetB3...")

    base_model = EfficientNetB3(
        input_shape=(*CONFIG['image_size'], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*CONFIG['image_size'], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(CONFIG['l2_reg']))(x)
    x = Dropout(CONFIG['dropout_rate'])(x)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(CONFIG['l2_reg']))(x)

    model = models.Model(inputs, outputs, name='EfficientNetB3')
    return model, base_model


def build_vision_transformer():
    """Build simplified Vision Transformer"""
    print("\nBuilding Vision Transformer...")

    patch_size = 16
    num_patches = (CONFIG['image_size'][0] // patch_size) ** 2
    projection_dim = 128
    num_heads = 4
    transformer_units = [projection_dim * 2, projection_dim]

    inputs = keras.Input(shape=(*CONFIG['image_size'], 3))

    # Create patches
    patches = layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size)(inputs)
    patch_dims = patches.shape[1] * patches.shape[2]
    patches = layers.Reshape((patch_dims, projection_dim))(patches)

    # Add positional embeddings
    positions = tf.range(start=0, limit=patch_dims, delta=1)
    pos_embedding = layers.Embedding(input_dim=patch_dims, output_dim=projection_dim)(positions)
    encoded = patches + pos_embedding

    # Transformer blocks
    for _ in range(2):
        # Multi-head attention
        x1 = LayerNormalization(epsilon=1e-6)(encoded)
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim
        )(x1, x1)
        x2 = Add()([attention_output, encoded])

        # MLP
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = Dense(transformer_units[0], activation='gelu')(x3)
        x3 = Dropout(0.1)(x3)
        x3 = Dense(transformer_units[1])(x3)
        x3 = Dropout(0.1)(x3)
        encoded = Add()([x3, x2])

    # Classification head
    representation = LayerNormalization(epsilon=1e-6)(encoded)
    representation = GlobalAveragePooling2D()(
        layers.Reshape((int(np.sqrt(patch_dims)), int(np.sqrt(patch_dims)), projection_dim))(representation)
    )
    representation = Dropout(CONFIG['dropout_rate'])(representation)
    outputs = Dense(1, activation='sigmoid')(representation)

    model = models.Model(inputs, outputs, name='VisionTransformer')
    return model, None


def build_mobilenet_base():
    """Build MobileNetV2 without attention (baseline)"""
    print("\nBuilding MobileNetV2 (base - no attention)...")

    base_model = MobileNetV2(
        input_shape=(*CONFIG['image_size'], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*CONFIG['image_size'], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(CONFIG['l2_reg']))(x)
    x = Dropout(CONFIG['dropout_rate'])(x)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(CONFIG['l2_reg']))(x)

    model = models.Model(inputs, outputs, name='MobileNetV2_Base')
    return model, base_model


def build_mobilenet_with_attention():
    """Build MobileNetV2 with Channel Attention (proposed method)"""
    print("\nBuilding MobileNetV2 + Channel Attention (PROPOSED)...")

    base_model = MobileNetV2(
        input_shape=(*CONFIG['image_size'], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*CONFIG['image_size'], 3))
    x = base_model(inputs, training=False)

    # Add channel attention
    x = channel_attention(x, ratio=CONFIG['attention_ratio'], name='channel_attn')

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(CONFIG['l2_reg']))(x)
    x = Dropout(CONFIG['dropout_rate'])(x)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(CONFIG['l2_reg']))(x)

    model = models.Model(inputs, outputs, name='MobileNetV2_Attention')
    return model, base_model


def train_model(model, base_model, train_data, val_data, model_name):
    """Train a model with two-phase approach (transfer + fine-tune)"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")

    X_train, y_train = train_data
    X_val, y_val = val_data

    # Compile model
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=CONFIG['initial_lr']),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    # Callbacks
    checkpoint_path = f"comparison_results/models/{model_name}_best.keras"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=CONFIG['patience'], restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_auc', mode='max', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]

    # Phase 1: Transfer learning
    print(f"\n[PHASE 1] Transfer Learning ({CONFIG['initial_epochs']} epochs)")
    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['initial_epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Fine-tuning (if base model exists)
    if base_model is not None:
        print(f"\n[PHASE 2] Fine-Tuning ({CONFIG['fine_tune_epochs']} epochs)")

        # Unfreeze top layers
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 50
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Freeze batch normalization layers
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        # Recompile with lower learning rate
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=CONFIG['fine_tune_lr']),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        total_epochs = CONFIG['initial_epochs'] + CONFIG['fine_tune_epochs']
        history2 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=total_epochs,
            initial_epoch=len(history1.epoch),
            batch_size=CONFIG['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        # Combine histories
        for key in history1.history.keys():
            history1.history[key].extend(history2.history[key])

    # Load best weights
    model.load_weights(checkpoint_path)

    return model, history1


def evaluate_model(model, test_data, model_name):
    """Evaluate model and return metrics"""
    X_test, y_test = test_data

    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*80}")

    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0).ravel()
    y_pred_class = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_class)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred_class)

    # Evaluate with model
    loss, acc, auc_metric = model.evaluate(X_test, y_test, verbose=0)

    results = {
        'Model': model_name,
        'Accuracy': accuracy * 100,
        'AUC': auc,
        'F1-Score': f1,
        'Loss': loss
    }

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"AUC: {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Loss: {loss:.4f}")

    return results, y_pred_proba, y_pred_class


def create_comparison_table(results_list):
    """Create and save comparison table"""
    df = pd.DataFrame(results_list)
    df = df.sort_values('AUC', ascending=False)

    print("\n" + "="*80)
    print("BASELINE COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv('comparison_results/baseline_comparison.csv', index=False)

    # Create formatted table for paper
    print("\n" + "="*80)
    print("LATEX TABLE (for paper)")
    print("="*80)

    latex_table = df.to_latex(
        index=False,
        float_format="%.2f",
        column_format='lcccc'
    )
    print(latex_table)

    with open('comparison_results/baseline_comparison.tex', 'w') as f:
        f.write(latex_table)

    return df


def plot_comparison_charts(results_df):
    """Create comparison visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = results_df['Model'].values

    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    bars = ax.barh(models, results_df['Accuracy'], color='steelblue')
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([85, 95])
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.2, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}%', ha='left', va='center', fontsize=10)

    # Plot 2: AUC comparison
    ax = axes[0, 1]
    bars = ax.barh(models, results_df['AUC'], color='coral')
    ax.set_xlabel('AUC Score', fontsize=12)
    ax.set_title('Model AUC Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([0.93, 0.98])
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=10)

    # Plot 3: F1-Score comparison
    ax = axes[1, 0]
    bars = ax.barh(models, results_df['F1-Score'], color='seagreen')
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([0.85, 0.95])
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=10)

    # Plot 4: Multi-metric radar chart
    ax = axes[1, 1]
    ax.axis('off')

    # Create text summary
    best_model = results_df.iloc[0]['Model']
    best_acc = results_df.iloc[0]['Accuracy']
    best_auc = results_df.iloc[0]['AUC']
    best_f1 = results_df.iloc[0]['F1-Score']

    summary_text = f"""
    BEST PERFORMING MODEL

    Model: {best_model}

    Accuracy: {best_acc:.2f}%
    AUC: {best_auc:.4f}
    F1-Score: {best_f1:.4f}

    Improvement over baseline:
    • ResNet50: +{best_acc - results_df[results_df['Model']=='ResNet50']['Accuracy'].values[0]:.2f}%
    • EfficientNetB3: +{best_acc - results_df[results_df['Model']=='EfficientNetB3']['Accuracy'].values[0] if 'EfficientNetB3' in results_df['Model'].values else 0:.2f}%
    """

    ax.text(0.5, 0.5, summary_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('comparison_results/plots/baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison plots saved to: comparison_results/plots/baseline_comparison.png")
    plt.close()


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("BASELINE MODEL COMPARISON - MTECH RESEARCH PROJECT")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    train_data, val_data, test_data = load_and_preprocess_data()

    # Define models to train
    models_to_train = [
        ('ResNet50', build_resnet50),
        ('EfficientNetB3', build_efficientnet_b3),
        ('VisionTransformer', build_vision_transformer),
        ('MobileNetV2_Base', build_mobilenet_base),
        ('MobileNetV2_Attention', build_mobilenet_with_attention),
    ]

    results_list = []

    # Train and evaluate each model
    for model_name, build_fn in models_to_train:
        try:
            # Build model
            model, base_model = build_fn()

            # Train model
            trained_model, history = train_model(
                model, base_model, train_data, val_data, model_name
            )

            # Evaluate model
            results, y_pred_proba, y_pred_class = evaluate_model(
                trained_model, test_data, model_name
            )
            results_list.append(results)

            # Clear memory
            tf.keras.backend.clear_session()

        except Exception as e:
            print(f"\n❌ Error training {model_name}: {str(e)}")
            continue

    # Create comparison table
    if results_list:
        results_df = create_comparison_table(results_list)
        plot_comparison_charts(results_df)

        print("\n" + "="*80)
        print("✓ BASELINE COMPARISON COMPLETE!")
        print("="*80)
        print(f"Results saved to: comparison_results/")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\n❌ No models were successfully trained!")


if __name__ == '__main__':
    main()
