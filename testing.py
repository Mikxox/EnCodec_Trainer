# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Command-line for audio compression."""

from pathlib import Path
import sys
import os

import torchaudio
import torch

from compress import compress, decompress, MODELS
from utils import save_audio, convert_audio
from model import EncodecModel, EncodedFrame
import struct
import binary
import io
import typing as tp
import math

def check_clipping(wav):
    mx = wav.abs().max()
    limit = 0.99
    if mx > limit:
        print(
            f"Clipping!! max scale {mx}, limit is {limit}. "
            "To avoid clipping, use the `-r` option to rescale the output.",
            file=sys.stderr)

def inspect_file(path):
  print("-" * 10)
  print("Source:", path)
  print("-" * 10)
  print(f" - File size: {os.path.getsize(path)} bytes")
  print(f" - {torchaudio.info(path)}")

print(torch.cuda.is_available())

write_encoded = True
comp = True
device = 'cpu' # cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, privateuseone
# notice, using cpu to encode and then cuda to decode throws errors, possibly due to gpu rounding via TF32 operations?

# If you want to use cpu / cuda interchangeably, these NEED to be set to false
# Setting them to true should allow for a speedup but cuda will get different results than cpu
# Also the amount of speedup seems to be negligible
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False


song = Path('audio/song_input.wav')
output = Path('audio/output_song.ecdc')
outputw = Path('audio/output_song.wav')
model_name = 'my_encodec_24khz' # 'encodec_24khz'
model = MODELS[model_name](checkpoint_name='saves/batch29_cut100000_epoch10.pth').to(device)

model.train()
wav, sr = torchaudio.load(song)

wav = convert_audio(wav, sr, 24000, 1)
wav = wav[None,:]
output_wav, _, frames = model.forward(wav)


encodes = []
for emb, scale in frames:
    ecc = model.quantizer.encode(emb, model.sample_rate, model.bandwidth)
    encodes.append((ecc, scale))

buf = struct.pack('i', len(encodes))
for xx1, scale in encodes:
    print(xx1.shape)
    xx2 = xx1.flatten()
    buf += struct.pack('i', len(xx2))
    buf += struct.pack('%sh' % len(xx2), *xx2)
    buf += struct.pack('f', scale.item())

with open(output, "wb") as newFile:
    newFile.write(buf)

encoded_list = []
with open(output, 'rb') as newFile:
    buf = newFile.read()
    l1 = struct.unpack_from('i', buf)[0]
    offset = 1 * 4
    for i in range(l1):
        l2 = struct.unpack_from('i', buf, offset)[0]
        offset += 1 * 4
        tt = torch.tensor(struct.unpack_from('%sh' % l2, buf, offset))
        offset += l2 * 2
        tt = torch.reshape(tt, (32, 1, l2//32)).to(device)
        tt = model.quantizer.decode(tt)
        encoded_list.append((tt, torch.tensor(struct.unpack_from('f', buf, offset)).to(device)))
        offset += 1 * 4

output_wav = model.decode(encoded_list)
output_wav = output_wav.to('cpu')
output_wav = torch.squeeze(output_wav)
output_wav = output_wav[None,:]
save_audio(output_wav, outputw, model.sample_rate, rescale=True)
