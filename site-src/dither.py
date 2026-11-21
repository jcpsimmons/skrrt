import sys
import numpy as np
from PIL import Image, ImageSequence, ImageOps
src, dst = sys.argv[1], sys.argv[2]
B = np.array([[0,8,2,10],[12,4,14,6],[3,11,1,9],[15,7,13,5]], np.float32)
im = Image.open(src); frames, durs = [], []
for f in ImageSequence.Iterator(im):
    g = ImageOps.autocontrast(f.convert("L"), cutoff=1).resize((320, 180), Image.LANCZOS)
    a = np.asarray(g, np.float32) / 255.0
    a = np.clip((a - 0.5) * 1.15 + 0.52, 0, 1) ** 0.9
    h, w = a.shape
    a[h - 9:, :58] = 0  # third-party watermark corner
    t = (np.tile(B, (h // 4 + 1, w // 4 + 1))[:h, :w] + 0.5) / 16.0
    p = Image.fromarray((a > t).astype(np.uint8), "P")
    p.putpalette([0, 0, 0, 0x87, 0xFF, 0x87] + [0] * (254 * 3))
    frames.append(p); durs.append(f.info.get("duration", 50))
frames[0].save(dst, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True, disposal=1)
print(len(frames), frames[0].size, durs[:3])
