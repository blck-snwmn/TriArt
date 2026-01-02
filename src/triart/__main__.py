import argparse

from triart.polygonizer import polygonize_image, save_image


def main():
    parser = argparse.ArgumentParser(
        description="Polygonize an image using Delaunay triangulation"
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument(
        "-o", "--output", default="output.jpg", help="Output image path"
    )
    parser.add_argument(
        "-n", "--num-points", type=int, default=6000, help="Number of points"
    )
    parser.add_argument(
        "-s",
        "--sampling",
        choices=["random", "poisson", "adaptive", "weighted"],
        default="poisson",
        help="Sampling mode: random (edge-only), poisson (uniform fill), "
        "adaptive (dense near edges), weighted (edge-strength based)",
    )
    args = parser.parse_args()

    result_image = polygonize_image(
        args.input, num_points=args.num_points, sampling_mode=args.sampling
    )
    save_image(result_image, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
