import argparse
import json
import os
import random

import cv2
import numpy as np


IMAGE_SIZE = 1024


def create_pattern(architecture, size=96):
    """
    Create a synthetic reference pattern representing either
    DRAM or FinFET architecture.
    """
    img = np.zeros((size, size), dtype=np.uint8)

    if architecture.upper() == "DRAM":
        # Repeated memory-cell style structure
        for y in range(8, size - 8, 16):
            for x in range(8, size - 8, 16):
                cv2.rectangle(
                    img,
                    (x, y),
                    (x + 8, y + 8),
                    210,
                    -1
                )
                cv2.line(
                    img,
                    (x + 4, y),
                    (x + 4, y + 12),
                    255,
                    1
                )

    elif architecture.upper() == "FINFET":
        # Parallel fin/gate style structure
        for x in range(10, size - 10, 14):
            cv2.rectangle(
                img,
                (x, 8),
                (x + 5, size - 8),
                220,
                -1
            )

        for y in range(18, size - 18, 24):
            cv2.line(
                img,
                (5, y),
                (size - 5, y),
                255,
                2
            )

    else:
        raise ValueError("Architecture must be DRAM or FinFET.")

    # Slight blur creates more realistic imaging characteristics.
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def generate_pair(output_dir, pair_id, architecture):
    """
    Generate one reference/search image pair and record the
    true center coordinates of the reference pattern.
    """

    pair_dir = os.path.join(
        output_dir,
        f"pair_{pair_id:04d}"
    )

    os.makedirs(pair_dir, exist_ok=True)

    reference = create_pattern(architecture)

    h, w = reference.shape

    # Random valid center location in the search image.
    margin = max(h, w) // 2 + 20

    center_x = random.randint(
        margin,
        IMAGE_SIZE - margin
    )

    center_y = random.randint(
        margin,
        IMAGE_SIZE - margin
    )

    search = np.random.normal(
        35,
        12,
        (IMAGE_SIZE, IMAGE_SIZE)
    ).clip(0, 255).astype(np.uint8)

    x0 = int(center_x - w / 2)
    y0 = int(center_y - h / 2)

    # Insert the reference pattern into the search image.
    roi = search[
        y0:y0 + h,
        x0:x0 + w
    ]

    search[
        y0:y0 + h,
        x0:x0 + w
    ] = np.maximum(
        roi,
        reference
    )

    # Add mild imaging noise.
    noise = np.random.normal(
        0,
        3,
        search.shape
    )

    search = np.clip(
        search.astype(np.float32) + noise,
        0,
        255
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(pair_dir, "reference.png"),
        reference
    )

    cv2.imwrite(
        os.path.join(pair_dir, "search.png"),
        search
    )

    ground_truth = {
        "pair": f"pair_{pair_id:04d}",
        "architecture": architecture.upper(),
        "center_x": center_x,
        "center_y": center_y,
        "image_width": IMAGE_SIZE,
        "image_height": IMAGE_SIZE
    }

    with open(
        os.path.join(pair_dir, "ground_truth.json"),
        "w"
    ) as f:
        json.dump(
            ground_truth,
            f,
            indent=2
        )


def main():
    parser = argparse.ArgumentParser(
        description="DRIFT-SENSE V24 synthetic dataset generator"
    )

    parser.add_argument(
        "--architecture",
        required=True,
        choices=["DRAM", "FinFET"],
        help="Architecture style"
    )

    parser.add_argument(
        "--num-pairs",
        required=True,
        type=int,
        help="Number of image pairs"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory"
    )

    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    for i in range(1, args.num_pairs + 1):
        generate_pair(
            args.output_dir,
            i,
            args.architecture
        )

    print(
        f"Generated {args.num_pairs} "
        f"{args.architecture} pair(s) in "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
