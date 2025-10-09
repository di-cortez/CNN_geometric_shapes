import os
import time
import datetime
import argparse

# main functions used
from generate_dataset import generate_data
from train_model import train_model

# --- Model Parameters ---
# you can adjust

# for Dataset
NUM_IMAGES = 1_000             # images total created
IMAGE_SIZE = 40                # image size

# for training the model
NUM_CLASSES = 6     # number os different classes (rectangle, elipse, triangle, etc.)
NUM_EPOCHS = 28
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT = 0.33

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Executa um ciclo de geração de dados e trainamento do modelo.")

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Caminho do arquivo já treinado'
    )

    args = parser.parse_args()

    experiment_start_time = time.time()
    
    if args.resume:
        print(f"ON GOING TRAINING, MODEL: {args.resume}")
        dataset_folder_path = os.path.dirname(args.resume)
        if not os.path.exists(dataset_folder_path):
            print(f"model not found ({dataset_folder_path})")
            exit
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 50
    else:
        print(f"NEW MODEL TRAINING")
    # dataset generation
        dataset_folder_path = generate_data(
            num_images=NUM_IMAGES,
            img_size=IMAGE_SIZE
        )

    # training execution
    final_val_accuracy = train_model(
        dataset_path=dataset_folder_path,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        dropout=DROPOUT,
        img_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        num_images=NUM_IMAGES,
        checkpoint_path = args.resume
    )

    # --- saving time details ---
    experiment_end_time = time.time()
    total_duration_seconds = experiment_end_time - experiment_start_time
    
    minutes = total_duration_seconds // 60
    seconds = total_duration_seconds % 60
    duration_str = f"{int(minutes)} minute(s) e {seconds:.2f} second(s)"

    # path to save summary.txt
    summary_file_path = os.path.join(dataset_folder_path, 'summary.txt')
    
    # content to be written
    summary_content = f"""
Summary
-------------------------
Execution time: {datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}
Total time: {duration_str}

Parameters:
- Images number: {NUM_IMAGES}
- Image size: {IMAGE_SIZE}x{IMAGE_SIZE}

- Class' number: {NUM_CLASSES}
- Epochs' number {NUM_EPOCHS}
- Batch Size: {BATCH_SIZE}
- Learning Rate: {LEARNING_RATE}
- Dropout: {DROPOUT}

- Final Accurancy: {final_val_accuracy:.2f}%
"""

    # write in summary.txt
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)

    print("\n" + "="*50)
    print("Finished")
    print(f"Saved in: {summary_file_path}")
    print(f"Runtime: {duration_str}")
    print("="*50)