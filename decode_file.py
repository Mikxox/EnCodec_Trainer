from pathlib import Path
import torch
from compress import MODELS
from utils import save_audio
import struct

# path to encoded file
input = Path('audio/input_file.ecdc')

# path to output decoded wav file
output = Path('audio/output_song.wav')


device = 'cpu' # cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, privateuseone
# notice, using cpu to encode and then cuda to decode throws errors, possibly due to gpu rounding via TF32 operations?

# If you want to use cpu / cuda interchangeably, these NEED to be set to false
# Setting them to true should allow for a speedup but cuda will get different results than cpu
# Also the amount of speedup seems to be negligible
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

model_name = 'my_encodec_24khz' # 'encodec_24khz'
model = MODELS[model_name](checkpoint_name='saves/batch29_cut100000_epoch10.pth').to(device)

encoded_list = []
with open(input, 'rb') as newFile:
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
save_audio(output_wav, output, model.sample_rate, rescale=True)
