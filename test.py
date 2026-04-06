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
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
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
                         interpolation_weights=None, tile_size=512, tile_overlap=64):
    _, _, h, w = content.shape
    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        return style_transfer(vgg, decoder, content, style, alpha, interpolation_weights)

    step = tile_size - tile_overlap
    if step <= 0:
        raise ValueError('tile_size는 tile_overlap보다 커야 합니다.')

    ys = _tile_starts(h, tile_size, step)
    xs = _tile_starts(w, tile_size, step)

    output_acc = torch.zeros_like(content)
    weight_acc = torch.zeros((1, 1, h, w), device=content.device)

    for y in ys:
        for x in xs:
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            content_tile = content[:, :, y:y2, x:x2]
            out_tile = style_transfer(vgg, decoder, content_tile, style, alpha, interpolation_weights)
            
            target_h = y2 - y
            target_w = x2 - x
            out_h, out_w = out_tile.shape[2], out_tile.shape[3]

            # 크기 불일치 방지: 출력이 더 크면 자르고, 작으면 패딩
            if out_h < target_h or out_w < target_w:
                pad_h = max(0, target_h - out_h)
                pad_w = max(0, target_w - out_w)
                out_tile = torch.nn.functional.pad(out_tile, (0, pad_w, 0, pad_h))

            out_tile = out_tile[:, :, :target_h, :target_w]

            w_tile = torch.ones((1, 1, target_h, target_w), device=content.device)
            output_acc[:, :, y:y2, x:x2] += out_tile * w_tile
            weight_acc[:, :, y:y2, x:x2] += w_tile

    return output_acc / weight_acc.clamp(min=1e-8)



parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')
parser.add_argument('--tile_size', type=int, default=0,
                    help='타일 한 변 크기. 0이면 타일 모드 비활성화')
parser.add_argument('--tile_overlap', type=int, default=64,
                    help='타일 겹침 크기(경계 이음새 완화용)')


args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

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
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha, interpolation_weights)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, args.save_ext)
        save_image(output, str(output_name))

    else:  # process one content and one style
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            if args.preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                if args.tile_size > 0:
                    output = style_transfer_tiled(
                        vgg, decoder, content, style,
                        alpha=args.alpha,
                        tile_size=args.tile_size,
                        tile_overlap=args.tile_overlap
                    )
                else:
                    output = style_transfer(vgg, decoder, content, style, args.alpha)
                output = output.cpu()


            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            save_image(output, str(output_name))
