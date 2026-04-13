import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def build_output_dir(base_output, do_interpolation, preserve_color, tile_size):
    parts = ['images']
    if do_interpolation:
        parts.append('interpolation')
    else:
        parts.append('tiled' if tile_size > 0 else 'basic')

    if preserve_color:
        parts.append('preserve_color')

    output_dir = Path(base_output).joinpath(*parts)
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def parse_alpha_values(alpha, alpha_values):
    if alpha_values:
        values = [float(value.strip()) for value in alpha_values.split(',') if value.strip()]
    else:
        values = [alpha]

    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError('Alpha values must be between 0 and 1')
    return values


def format_alpha_tag(alpha):
    return f'alpha_{alpha:.2f}'.replace('.', 'p')


def build_output_path(output_dir, base_name, save_ext, alpha):
    alpha_tag = format_alpha_tag(alpha)
    candidate = output_dir / f'{base_name}_{alpha_tag}{save_ext}'
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = output_dir / f'{base_name}_{alpha_tag}_{index:02d}{save_ext}'
        if not candidate.exists():
            return candidate
        index += 1


def build_tile_output_dir(output_dir, output_path):
    tile_output_dir = output_dir / 'tiles' / output_path.stem
    tile_output_dir.mkdir(exist_ok=True, parents=True)
    return tile_output_dir


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert 0.0 <= alpha <= 1.0
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, c_channels, height, width = content_f.size()
        feat = torch.zeros((1, c_channels, height, width), device=device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, weight in enumerate(interpolation_weights):
            feat = feat + weight * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def _tile_starts(length, tile_size, step):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def style_transfer_tiled(vgg, decoder, content, style, alpha=1.0,
                         interpolation_weights=None, tile_size=512, tile_overlap=64,
                         tile_output_dir=None, tile_save_ext='.jpg'):
    _, _, height, width = content.shape
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        return style_transfer(vgg, decoder, content, style, alpha, interpolation_weights)

    step = tile_size - tile_overlap
    if step <= 0:
        raise ValueError('tile_size must be larger than tile_overlap')

    y_starts = _tile_starts(height, tile_size, step)
    x_starts = _tile_starts(width, tile_size, step)

    output_acc = torch.zeros_like(content)
    weight_acc = torch.zeros((1, 1, height, width), device=content.device)

    for row_idx, y_start in enumerate(y_starts):
        for col_idx, x_start in enumerate(x_starts):
            y_end = min(y_start + tile_size, height)
            x_end = min(x_start + tile_size, width)
            content_tile = content[:, :, y_start:y_end, x_start:x_end]
            out_tile = style_transfer(
                vgg,
                decoder,
                content_tile,
                style,
                alpha,
                interpolation_weights,
            )

            target_h = y_end - y_start
            target_w = x_end - x_start
            out_h, out_w = out_tile.shape[2], out_tile.shape[3]

            # Keep tile accumulation safe when decoded tile sizes differ slightly.
            if out_h < target_h or out_w < target_w:
                pad_h = max(0, target_h - out_h)
                pad_w = max(0, target_w - out_w)
                out_tile = torch.nn.functional.pad(out_tile, (0, pad_w, 0, pad_h))

            out_tile = out_tile[:, :, :target_h, :target_w]

            if tile_output_dir is not None:
                tile_name = (
                    f'tile_r{row_idx:03d}_c{col_idx:03d}_'
                    f'y{y_start:04d}_x{x_start:04d}{tile_save_ext}'
                )
                save_image(out_tile.detach().cpu(), str(tile_output_dir / tile_name))

            tile_weight = torch.ones((1, 1, target_h, target_w), device=content.device)
            output_acc[:, :, y_start:y_end, x_start:x_end] += out_tile * tile_weight
            weight_acc[:, :, y_start:y_end, x_start:x_end] += tile_weight

    return output_acc / weight_acc.clamp(min=1e-8)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style '
                         'images separated by commas if you want to do style '
                         'interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, '
                         'keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, '
                         'keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Base directory to save output images by mode')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of '
                         'stylization. Should be between 0 and 1')
parser.add_argument('--alpha_values', type=str, default='',
                    help='Comma-separated alpha values for saving multiple results in one run')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')
parser.add_argument('--tile_size', type=int, default=0,
                    help='Tile edge size. Set to 0 to disable tiled mode')
parser.add_argument('--tile_overlap', type=int, default=64,
                    help='Tile overlap size to reduce border artifacts')


args = parser.parse_args()

do_interpolation = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Either --content or --content_dir should be given.
assert args.content or args.content_dir
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [path for path in content_dir.glob('*')]

# Either --style or --style_dir should be given.
assert args.style or args.style_dir
if args.style:
    style_paths = [Path(path.strip()) for path in args.style.split(',')]
    if len(style_paths) > 1:
        do_interpolation = True
        assert args.style_interpolation_weights != '', 'Please specify interpolation weights'
        weights = [int(weight) for weight in args.style_interpolation_weights.split(',')]
        assert len(weights) == len(style_paths), 'Provide one interpolation weight per style image'
        interpolation_weights = [weight / sum(weights) for weight in weights]
else:
    style_dir = Path(args.style_dir)
    style_paths = [path for path in style_dir.glob('*')]

output_dir = build_output_dir(
    args.output,
    do_interpolation=do_interpolation,
    preserve_color=args.preserve_color,
    tile_size=args.tile_size,
)
alpha_values = parse_alpha_values(args.alpha, args.alpha_values)
print(f'Saving outputs to {output_dir.resolve()}')

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

for content_path in content_paths:
    if do_interpolation:
        style = torch.stack([
            style_tf(Image.open(str(path)).convert('RGB'))
            for path in style_paths
        ])
        content = content_tf(Image.open(str(content_path)).convert('RGB')) \
            .unsqueeze(0).expand_as(style)
        if args.preserve_color:
            style = torch.stack([
                coral(style[i], content[i])
                for i in range(style.size(0))
            ])
        style = style.to(device)
        content = content.to(device)
        for alpha in alpha_values:
            with torch.no_grad():
                output = style_transfer(
                    vgg,
                    decoder,
                    content,
                    style,
                    alpha,
                    interpolation_weights,
                )
            output = output.cpu()
            output_name = build_output_path(
                output_dir,
                f'{content_path.stem}_interpolation',
                args.save_ext,
                alpha,
            )
            save_image(output, str(output_name))
            print(f'Saved {output_name.resolve()}')

    else:
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)).convert('RGB'))
            style = style_tf(Image.open(str(style_path)).convert('RGB'))
            if args.preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            for alpha in alpha_values:
                output_name = build_output_path(
                    output_dir,
                    f'{content_path.stem}_stylized_{style_path.stem}',
                    args.save_ext,
                    alpha,
                )
                tile_output_dir = None
                with torch.no_grad():
                    if args.tile_size > 0:
                        tile_output_dir = build_tile_output_dir(output_dir, output_name)
                        output = style_transfer_tiled(
                            vgg,
                            decoder,
                            content,
                            style,
                            alpha=alpha,
                            tile_size=args.tile_size,
                            tile_overlap=args.tile_overlap,
                            tile_output_dir=tile_output_dir,
                            tile_save_ext=args.save_ext,
                        )
                    else:
                        output = style_transfer(vgg, decoder, content, style, alpha)
                    output = output.cpu()

                save_image(output, str(output_name))
                print(f'Saved {output_name.resolve()}')
                if tile_output_dir is not None:
                    print(f'Saved tiles to {tile_output_dir.resolve()}')
