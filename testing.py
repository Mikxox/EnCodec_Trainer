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
from model import EncodecModel
import struct

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
device = 'cuda' # cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, privateuseone
# notice, using cpu to encode and then cuda to decode throws errors, possibly due to gpu rounding via TF32 operations?

# If you want to use cpu / cuda interchangeably, these NEED to be set to false
# Setting them to true should allow for a speedup but cuda will get different results than cpu
# Also the amount of speedup seems to be negligible
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False



song = Path('audio/song_input.flac')
output = Path('audio/output_ww1.ecdc')
outputw = Path('audio/output_ww1.wav')
model_name = 'my_encodec_24khz' # 'encodec_24khz'

model = MODELS[model_name](checkpoint_name='saves/batch28_cut100000_epoch20.pth').to(device)
model.set_target_bandwidth(24)
model.train()
wav, sr = torchaudio.load(song)

# cut off because quantization not fixed yet
wav = wav[:, :500000]
# print(wav.shape)
wav = wav[None,:]
wav = wav.to(device)

# print(wav.shape)

# wav = convert_audio(wav, sr, model.sample_rate, model.channels)
# compressed = compress(model, wav.to(device), use_lm=False)

# if write_encoded:
# output.write_bytes(compressed) # to write output file

# Directly run decompression stage
# out, out_sample_rate = decompress(compressed, device=device)
# exit(0)
# out = out.to('cpu')  # can't save a cuda tensor
# check_clipping(out)
# torchaudio.save(outputw, out, 24000)
# exit(0)
# save_audio(out, outputw, out_sample_rate, rescale=True)

xo = model.encode(wav)
# print(xo)
# print(len(xo))
# print(xo[0][0].shape, xo[0][0].type(), xo[0][1])
# print("XXX")


buf = struct.pack('i', len(xo))
for xx1, scale in xo:
    xx2 = xx1.flatten()
    buf += struct.pack('i', len(xx2))
    buf += struct.pack('%sf' % len(xx2), *xx2)
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
        tt = torch.tensor(struct.unpack_from('%sf' % l2, buf, offset))
        offset += l2 * 4
        tt = torch.reshape(tt, (1, 128, l2//128)).to(device)
        encoded_list.append((tt, torch.tensor(struct.unpack_from('f', buf, offset)).to(device)))
        offset += 1 * 4

output_wav = model.decode(encoded_list)

# newFile = open(output, "wb")
# newFile.write(buf)

# output_wav = model.forward(wav)
# o1 = model.ef(wav)
# print("encoded")
# output_wav = model.df(o1)
# print("done, now saving")
output_wav = output_wav.to('cpu')
output_wav = torch.squeeze(output_wav)
output_wav = output_wav[None,:]
print(output_wav.shape)
# exit(0)
torchaudio.save(outputw, output_wav, 24000)
inspect_file(outputw)
exit(0)


if comp:
    song = Path('audio/song_input.flac')
    output = Path('audio/output2.ecdc')
    outputw = Path('audio/output2.wav')

    model_name = 'my_encodec_24khz' # 'encodec_24khz'
    model = MODELS[model_name](checkpoint_name='saves/new2/batch55_cut10000_epoch130.pth').to(device)
    # model.to(device)
    model.set_target_bandwidth(24)

    # model = HBPASM_model.Net(res=res)
    # target_bandwidths = [1.5, 3., 6, 12., 24.]
    # sample_rate = 24_000
    # channels = 1
    # model = EncodecModel._get_model(
    #         target_bandwidths, sample_rate, channels,
    #         causal=False, model_norm='time_group_norm', audio_normalize=True,
    #         segment=1., name='my_encodec_24khz')
    # pre_dic = torch.load(f'saves/batch16.pth')
    # model.load_state_dict(pre_dic)
    # model.to(device)
    # model.set_target_bandwidth(24)
    # model.eval()

    wav, sr = torchaudio.load(song)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    compressed = compress(model, wav.to(device), use_lm=False)

    if write_encoded:
        output.write_bytes(compressed) # to write output file

    # Directly run decompression stage
    out, out_sample_rate = decompress(compressed, device=device)
    out = out.to('cpu')  # can't save a cuda tensor
    check_clipping(out)
    save_audio(out, outputw, out_sample_rate, rescale=True)
else:
    song = Path('../audio/outputQuick.ecdc')
    outputw = Path('../audio/outputQuick.wav')

    out, out_sample_rate = decompress(song.read_bytes(), device=device)

    out = out.to('cpu') # can't save a cuda tensor
    check_clipping(out)
    save_audio(out, outputw, out_sample_rate, rescale=True)
