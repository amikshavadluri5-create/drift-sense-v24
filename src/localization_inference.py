
import argparse
import sys

import cv2
import numpy as np


def preprocess(image):
    """Convert image to a normalized grayscale representation."""
    if image is None:
        raise ValueError("Could not read image.")

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image.astype(np.float32)


def multiscale_template_matching(reference, search):
    """
    Locate the reference pattern inside the search image.

    Multiple scales are tested to provide robustness to small
    scale differences.
    """

    candidates = []

    rh0, rw0 = reference.shape
    sh, sw = search.shape

    # Small scale range around the original reference size.
    scales = [
        0.85,
        0.90,
        0.95,
        1.00,
        1.05,
        1.10,
        1.15
    ]

    for scale in scales:

        rw = max(20, int(rw0 * scale))
        rh = max(20, int(rh0 * scale))

        if rw >= sw or rh >= sh:
            continue

        resized = cv2.resize(
            reference,
            (rw, rh),
            interpolation=cv2.INTER_LINEAR
        )

        result = cv2.matchTemplate(
            search,
            resized,
            cv2.TM_CCOEFF_NORMED
        )

        # Keep several strong candidates.
        flat = result.reshape(-1)

        k = min(10, len(flat))

        if k == 0:
            continue

        indices = np.argpartition(
            flat,
            -k
        )[-k:]

        for idx in indices:

            y, x = np.unravel_index(
                idx,
                result.shape
            )

            score = float(result[y, x])

            candidates.append({
                "x": x + rw / 2.0,
                "y": y + rh / 2.0,
                "score": score,
                "width": rw,
                "height": rh,
                "scale": scale
            })

    return candidates


def refine_location(reference, search, candidates):
    """
    Refine the strongest template-match candidates using
    image-gradient similarity.

    Ground truth is never accessed here.
    """

    if not candidates:
        raise RuntimeError(
            "No localization candidates were generated."
        )

    # Sort by template correlation first.
    candidates = sorted(
        candidates,
        key=lambda c: c["score"],
        reverse=True
    )

    # Evaluate only the strongest candidates.
    candidates = candidates[:20]

    best = None
    best_score = -np.inf

    search_edges = cv2.Canny(
        search.astype(np.uint8),
        50,
        150
    )

    for candidate in candidates:

        w = candidate["width"]
        h = candidate["height"]

        cx = candidate["x"]
        cy = candidate["y"]

        x0 = int(round(cx - w / 2))
        y0 = int(round(cy - h / 2))

        x1 = x0 + w
        y1 = y0 + h

        if (
            x0 < 0
            or y0 < 0
            or x1 > search.shape[1]
            or y1 > search.shape[0]
        ):
            continue

        patch = search[
            y0:y1,
            x0:x1
        ]

        resized_ref = cv2.resize(
            reference,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalized correlation.
        corr = cv2.matchTemplate(
            patch,
            resized_ref,
            cv2.TM_CCOEFF_NORMED
        )

        corr_score = float(corr[0, 0])

        # Edge similarity.
        ref_edges = cv2.Canny(
            resized_ref.astype(np.uint8),
            50,
            150
        )

        patch_edges = search_edges[
            y0:y1,
            x0:x1
        ]

        edge_score = float(
            np.mean(
                ref_edges == patch_edges
            )
        )

        # Combined image-only score.
        score = (
            0.75 * corr_score
            + 0.25 * edge_score
        )

        if score > best_score:
            best_score = score

            best = {
                "x": float(cx),
                "y": float(cy)
            }

    if best is None:
        raise RuntimeError(
            "Unable to determine localization."
        )

    return best


def localize(reference_path, search_path):

    reference = cv2.imread(
        reference_path,
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        search_path,
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None:
        raise FileNotFoundError(
            f"Reference image not found: {reference_path}"
        )

    if search is None:
        raise FileNotFoundError(
            f"Search image not found: {search_path}"
        )

    reference = preprocess(reference)
    search = preprocess(search)

    candidates = multiscale_template_matching(
        reference,
        search
    )

    prediction = refine_location(
        reference,
        search,
        candidates
    )

    return prediction["x"], prediction["y"]


def main():

    parser = argparse.ArgumentParser(
        description=(
            "DRIFT-SENSE V24 image localization inference"
        )
    )

    parser.add_argument(
        "reference_image",
        help="Path to reference image"
    )

    parser.add_argument(
        "search_image",
        help="Path to search image"
    )

    args = parser.parse_args()

    try:

        x, y = localize(
            args.reference_image,
            args.search_image
        )

        # IMPORTANT:
        # The competition interface expects one coordinate.
        print(f"{x:.2f},{y:.2f}")

    except Exception as exc:

        print(
            f"ERROR: {exc}",
            file=sys.stderr
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
