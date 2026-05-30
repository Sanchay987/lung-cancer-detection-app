"""
Ablation Study for MTech Research
Analyzes the contribution of each component in the proposed model

Configurations:
1. Base (MobileNetV2 only - no attention, no regularization)
2. Base + L2 Regularization
3. Base + L2 + Channel Attention
4. Base + L2 + Channel Attention + Spatial Attention (NOVEL)

Author: MTech Research Project
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, models
from keras.applications import MobileNetV2
from keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    multiply,
    Reshape,
    Conv2D,
    Activation,
    Concatenate,
    Lambda
)
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from datetime import datetime

# Set random seeds
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

os.makedirs('ablation_results', exist_ok=True)
os.makedirs('ablation_results/models', exist_ok=True)
os.makedirs('ablation_results/plots', exist_ok=True)


def load_and_preprocess_data():
    """Load and preprocess data"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    with h5py.File('all_patches.hdf5', 'r') as f:
        images = f['ct_slices'][:]
        labels = f['slice_class'][:]

    target_size = CONFIG['image_size']
    images = np.stack([images] * 3, axis=-1)
    images = tf.image.resize(images, target_size)
    images = images.numpy().astype('float32') / 255.0
    labels = labels.reshape(-1)

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

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def channel_attention(input_feature, ratio=8, name='channel_attn'):
    """Channel Attention (Squeeze-and-Excitation)"""
    channel = input_feature.shape[-1]

    se = GlobalAveragePooling2D(name=f'{name}_gap')(input_feature)
    se = Dense(channel // ratio, activation='relu', use_bias=False, name=f'{name}_fc1')(se)
    se = Dense(channel, activation='sigmoid', use_bias=False, name=f'{name}_fc2')(se)
    se = Reshape((1, 1, channel), name=f'{name}_reshape')(se)

    return multiply([input_feature, se], name=f'{name}_multiply')


def spatial_attention(input_feature, name='spatial_attn'):
    """
    NOVEL CONTRIBUTION: Spatial Attention Module
    Focuses on 'where' is important in the feature map
    """
    # Average and max pooling across channels
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True), name=f'{name}_avgpool')(input_feature)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True), name=f'{name}_maxpool')(input_feature)

    # Concatenate
    concat = Concatenate(axis=-1, name=f'{name}_concat')([avg_pool, max_pool])

    # Conv layer to generate attention map
    attention = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', name=f'{name}_conv')(concat)

    # Apply attention
    return multiply([input_feature, attention], name=f'{name}_multiply')


def dual_attention(input_feature, ratio=8, name='dual_attn'):
    """
    NOVEL CONTRIBUTION: Dual Attention Module
    Combines channel attention and spatial attention
    """
    # Channel attention first
    x = channel_attention(input_feature, ratio=ratio, name=f'{name}_channel')

    # Then spatial attention
    x = spatial_attention(x, name=f'{name}_spatial')

    return x


def build_config_1_base():
    """Configuration 1: Base MobileNetV2 only (no attention, no regularization)"""
    print("\nBuilding Config 1: Base (no attention, no regularization)")

    base_model = MobileNetV2(
        input_shape=(*CONFIG['image_size'], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*CONFIG['image_size'], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)  # No L2
    x = Dropout(0.3)(x)  # Lower dropout
    outputs = Dense(1, activation='sigmoid')(x)  # No L2

    model = models.Model(inputs, outputs, name='Config1_Base')
    return model, base_model


def build_config_2_with_l2():
    """Configuration 2: Base + L2 Regularization"""
    print("\nBuilding Config 2: Base + L2 Regularization")

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

    model = models.Model(inputs, outputs, name='Config2_Base_L2')
    return model, base_model


def build_config_3_with_channel_attention():
    """Configuration 3: Base + L2 + Channel Attention"""
    print("\nBuilding Config 3: Base + L2 + Channel Attention")

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

    model = models.Model(inputs, outputs, name='Config3_ChannelAttn')
    return model, base_model


def build_config_4_with_dual_attention():
    """Configuration 4: Base + L2 + Dual Attention (NOVEL - FULL PROPOSED METHOD)"""
    print("\nBuilding Config 4: Base + L2 + Dual Attention (PROPOSED)")

    base_model = MobileNetV2(
        input_shape=(*CONFIG['image_size'], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*CONFIG['image_size'], 3))
    x = base_model(inputs, training=False)

    # Add DUAL attention (channel + spatial) - NOVEL CONTRIBUTION
    x = dual_attention(x, ratio=CONFIG['attention_ratio'], name='dual_attn')

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(CONFIG['l2_reg']))(x)
    x = Dropout(CONFIG['dropout_rate'])(x)
    outputs = Dense(1, activation='sigmoid', kernel_regularizer=l2(CONFIG['l2_reg']))(x)

    model = models.Model(inputs, outputs, name='Config4_DualAttn_PROPOSED')
    return model, base_model


def train_model(model, base_model, train_data, val_data, config_name):
    """Train model with two-phase approach"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {config_name}")
    print(f"{'='*80}")

    X_train, y_train = train_data
    X_val, y_val = val_data

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=CONFIG['initial_lr']),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    checkpoint_path = f"ablation_results/models/{config_name}_best.keras"
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

    # Phase 2: Fine-tuning
    if base_model is not None:
        print(f"\n[PHASE 2] Fine-Tuning ({CONFIG['fine_tune_epochs']} epochs)")

        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 50
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

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

        for key in history1.history.keys():
            history1.history[key].extend(history2.history[key])

    model.load_weights(checkpoint_path)
    return model, history1


def evaluate_model(model, test_data, config_name):
    """Evaluate model"""
    X_test, y_test = test_data

    print(f"\n{'='*80}")
    print(f"EVALUATING: {config_name}")
    print(f"{'='*80}")

    y_pred_proba = model.predict(X_test, verbose=0).ravel()
    y_pred_class = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_class)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred_class)

    loss, acc, auc_metric = model.evaluate(X_test, y_test, verbose=0)

    results = {
        'Configuration': config_name,
        'Accuracy': accuracy * 100,
        'AUC': auc,
        'F1-Score': f1,
        'Loss': loss
    }

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"AUC: {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return results


def create_ablation_table(results_list):
    """Create ablation study table"""
    df = pd.DataFrame(results_list)

    print("\n" + "="*80)
    print("ABLATION STUDY TABLE")
    print("="*80)
    print(df.to_string(index=False))

    df.to_csv('ablation_results/ablation_study.csv', index=False)

    # Calculate improvements
    print("\n" + "="*80)
    print("COMPONENT CONTRIBUTIONS")
    print("="*80)

    base_acc = df.iloc[0]['Accuracy']
    for i in range(1, len(df)):
        improvement = df.iloc[i]['Accuracy'] - df.iloc[i-1]['Accuracy']
        print(f"{df.iloc[i]['Configuration']}: +{improvement:.2f}% accuracy improvement")

    # LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.2f")
    with open('ablation_results/ablation_study.tex', 'w') as f:
        f.write(latex_table)

    return df


def plot_ablation_results(results_df):
    """Create ablation study visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    configs = results_df['Configuration'].values

    # Plot 1: Accuracy progression
    ax = axes[0, 0]
    x_pos = np.arange(len(configs))
    bars = ax.bar(x_pos, results_df['Accuracy'], color=['#e74c3c', '#e67e22', '#f39c12', '#27ae60'])
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy: Component-wise Improvement', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Base', '+L2', '+Channel\nAttn', '+Dual\nAttn\n(PROPOSED)'], fontsize=10)
    ax.set_ylim([88, 96])

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: AUC progression
    ax = axes[0, 1]
    bars = ax.bar(x_pos, results_df['AUC'], color=['#3498db', '#2980b9', '#1f618d', '#154360'])
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('AUC: Component-wise Improvement', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Base', '+L2', '+Channel\nAttn', '+Dual\nAttn\n(PROPOSED)'], fontsize=10)
    ax.set_ylim([0.94, 0.98])

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 3: Incremental improvement
    ax = axes[1, 0]
    base_acc = results_df.iloc[0]['Accuracy']
    improvements = [0] + [results_df.iloc[i]['Accuracy'] - results_df.iloc[i-1]['Accuracy']
                          for i in range(1, len(results_df))]
    cumulative = np.cumsum(improvements)

    ax.bar(x_pos, improvements, color='#16a085', alpha=0.7, label='Incremental')
    ax.plot(x_pos, cumulative, 'ro-', linewidth=2, markersize=8, label='Cumulative')
    ax.set_ylabel('Accuracy Improvement (%)', fontsize=12)
    ax.set_title('Incremental vs Cumulative Improvement', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Base', '+L2', '+Channel\nAttn', '+Dual\nAttn'], fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')

    final_config = results_df.iloc[-1]
    base_config = results_df.iloc[0]

    total_improvement = final_config['Accuracy'] - base_config['Accuracy']

    summary_text = f"""
    ABLATION STUDY SUMMARY

    Final Configuration (PROPOSED):
    {final_config['Configuration']}

    Performance Metrics:
    • Accuracy: {final_config['Accuracy']:.2f}%
    • AUC: {final_config['AUC']:.4f}
    • F1-Score: {final_config['F1-Score']:.4f}

    Total Improvement over Base:
    • Accuracy: +{total_improvement:.2f}%
    • AUC: +{final_config['AUC'] - base_config['AUC']:.4f}

    Key Contributions:
    1. L2 Regularization: prevents overfitting
    2. Channel Attention: focuses on important features
    3. Spatial Attention (NOVEL): identifies critical regions
    """

    ax.text(0.5, 0.5, summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='center',
            horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig('ablation_results/plots/ablation_study.png', dpi=300, bbox_inches='tight')
    print("\n✓ Ablation plots saved to: ablation_results/plots/ablation_study.png")
    plt.close()


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("ABLATION STUDY - MTECH RESEARCH PROJECT")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    train_data, val_data, test_data = load_and_preprocess_data()

    # Define configurations
    configs = [
        ('Config1_Base', build_config_1_base),
        ('Config2_Base+L2', build_config_2_with_l2),
        ('Config3_Base+L2+ChannelAttn', build_config_3_with_channel_attention),
        ('Config4_Base+L2+DualAttn_PROPOSED', build_config_4_with_dual_attention),
    ]

    results_list = []

    # Train and evaluate each configuration
    for config_name, build_fn in configs:
        try:
            model, base_model = build_fn()
            trained_model, history = train_model(model, base_model, train_data, val_data, config_name)
            results = evaluate_model(trained_model, test_data, config_name)
            results_list.append(results)

            tf.keras.backend.clear_session()

        except Exception as e:
            print(f"\n❌ Error with {config_name}: {str(e)}")
            continue

    # Create tables and plots
    if results_list:
        results_df = create_ablation_table(results_list)
        plot_ablation_results(results_df)

        print("\n" + "="*80)
        print("✓ ABLATION STUDY COMPLETE!")
        print("="*80)
        print(f"Results saved to: ablation_results/")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\n❌ No configurations were successfully trained!")


if __name__ == '__main__':
    main()
