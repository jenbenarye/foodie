# create_segmented_dataset.py

import os
import argparse
import numpy as np
import tensorflow as tf
import kagglehub
from datasets import load_dataset
from PIL import Image

def main(start_idx, end_idx):
    output_dir = "./segmented_val_dataset"
    os.makedirs(output_dir, exist_ok=True)

    # Load TFLite model
    model_dir = kagglehub.model_download(
        "google/mobile-food-segmenter-v1/tfLite/seefood-segmenter-mobile-food-segmenter-v1"
    )
    model_file = os.path.join(model_dir, "1.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load Food-101 dataset
    dataset = load_dataset("ethz/food101", split="validation", cache_dir="./datasets_cache")

    for idx in range(start_idx, min(end_idx, len(dataset))):
        try:
            img = dataset[idx]["image"]
        except Exception:
            print(f"Skipping {idx}: error reading image")
            continue

        # Preprocess image
        img_resized = img.resize((513, 513))
        img_array = np.array(img_resized, dtype=np.uint8)
        img_input = np.expand_dims(img_array, axis=0)

        if img_input.ndim != 4:
            print(f"Skipping {idx}: unexpected shape {img_input.shape}")
            continue

        # Run segmentation
        interpreter.set_tensor(input_details[0]["index"], img_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]
        mask = np.argmax(output, axis=-1)

        # Apply mask
        img_np = np.array(img_resized)
        img_np[mask == 0] = 0
        segmented_img = Image.fromarray(img_np)

        # Save segmented image
        class_label = dataset[idx]["label"]
        class_folder = os.path.join(output_dir, str(class_label))
        os.makedirs(class_folder, exist_ok=True)
        segmented_img.save(os.path.join(class_folder, f"{idx}.jpg"))

        if idx % 500 == 0:
            print(f"Saved {idx} images in batch {start_idx}-{end_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mobile food segmentation on Food-101 dataset")
    parser.add_argument("--start", type=int, required=True, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End index (exclusive)")
    args = parser.parse_args()

    main(args.start, args.end)
