"""
Create slides gif thumbnails to beautify the README.md
"""
import argparse
import subprocess
from pathlib import Path


def main(args: argparse.Namespace):

    # List all pdf files (avoiding images and 4:3 aspect ratio)
    slides_pdf = [f for f in args.root_dir.glob('**/*.pdf')]
    slides_pdf = [f for f in slides_pdf
                  if 'img' not in str(f) and '43.pdf' not in str(f)]

    # Extract thumbnails from pdf
    for slide_f in slides_pdf:
        print(f'Creating thumbnail for {str(slide_f)}...')
        cmd = ['convert',
               str(slide_f),
               '-thumbnail',
               f'x{args.thumb_size}',
               f'{args.out_dir}/frame%03d.png']
        subprocess.run(cmd)

        # Create gif from thumbnails
        out_gif = args.out_dir / f'{slide_f.stem.lower()}.gif'
        thumb_files = sorted([str(f) for f in args.out_dir.glob('*.png')])
        cmd = ['convert',
               '-loop',
               '0',
               '-delay',
               '100',
               *thumb_files,
               str(out_gif)]
        subprocess.run(cmd)

        # Cleanup - remove all temporary `frame*.png` files
        for thumb_f in thumb_files:
            Path(thumb_f).unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=Path, default='.')
    parser.add_argument('--out_dir', type=Path, default='./thumbs')
    parser.add_argument('--thumb_size', type=int, default=256)
    args = parser.parse_args()

    if not args.out_dir.is_dir():
        args.out_dir.mkdir(exist_ok=True, parents=True)

    main(args)
