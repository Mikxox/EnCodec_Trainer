import torch
import torch.optim as optim
import customAudioDataset as data
import os
import torch.backends.cudnn as cudnn

from model import EncodecModel 
from msstftd import MultiScaleSTFTDiscriminator
from audio_to_mel import Audio2Mel

EPSILON = 1e-8
BATCH_SIZE = 5 #5#55
TENSOR_CUT = 50000 #10000
MAX_EPOCH = 10000 # Just set this to a very big number and manually stop it
SAVE_FOLDER = f'saves/new7/'
SAVE_LOCATION = f'{SAVE_FOLDER}batch{BATCH_SIZE}_cut{TENSOR_CUT}_' # appends epoch{epoch}.pth

if not os.path.exists(SAVE_FOLDER):
   os.makedirs(SAVE_FOLDER)

def total_loss(fmap_real, logits_fake, fmap_fake, wav1, wav2, sample_rate=24000):
    relu = torch.nn.ReLU()
    l1Loss = torch.nn.L1Loss(reduction='mean')
    l2Loss = torch.nn.MSELoss(reduction='mean')
    loss = torch.tensor([0.0], device='cuda', requires_grad=True)
    factor = 100 / (len(fmap_real) * len(fmap_real[0]))

    for tt1 in range(len(fmap_real)):
        loss = loss + (torch.mean(relu(1 - logits_fake[tt1])) / len(logits_fake))
        for tt2 in range(len(fmap_real[tt1])):
            loss = loss + (l1Loss(fmap_real[tt1][tt2].detach(), fmap_fake[tt1][tt2]) * factor)
    loss = loss * (2/3)

    for i in range(5, 11):
        fft = Audio2Mel(win_length=2 ** i, hop_length=2 ** i // 4, n_mel_channels=64, sampling_rate=sample_rate)
        loss = loss + l1Loss(fft(wav1), fft(wav2)) + l2Loss(fft(wav1), fft(wav2))
    loss = (loss / 6) + l1Loss(wav1, wav2)
    return loss

def disc_loss(logits_real, logits_fake):
    cx = torch.nn.ReLU()
    lossd = torch.tensor([0.0], device='cuda', requires_grad=True)
    for tt1 in range(len(logits_real)):
        lossd = lossd + torch.mean(cx(1-logits_real[tt1])) + torch.mean(cx(1+logits_fake[tt1]))
    lossd = lossd / len(logits_real)
    return lossd

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch


def collate_fn(batch):
    tensors = []

    for waveform, _ in batch:
        tensors += [waveform]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    return tensors


def training(max_epoch = 5, log_interval = 20, fixed_length = 0, tensor_cut=100000, batch_size=8):
    csv_path = 'datasets/e-gmd-v1.0.0/fileTRAIN.csv'
    data_path = 'datasets/e-gmd-v1.0.0'

    if fixed_length > 0:
        trainset = data.CustomAudioDataset(csv_path, data_path, tensor_cut=tensor_cut, fixed_length=fixed_length)
    else:
        trainset = data.CustomAudioDataset(csv_path, data_path, tensor_cut=tensor_cut)
    

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,)

    cudnn.benchmark = True

    target_bandwidths = [1.5, 3., 6, 12., 24.]
    sample_rate = 24_000
    channels = 1
    model = EncodecModel._get_model(
                target_bandwidths, sample_rate, channels,
                causal=False, model_norm='time_group_norm', audio_normalize=True,
                segment=1., name='my_encodec_24khz')
    model.train()
    model.train_quantization = True
    model.cuda()
    
    disc = MultiScaleSTFTDiscriminator(filters=32)
    disc.train()
    disc.cuda()


    lr = 0.01
    # optimizer = optim.SGD([{'params': model.parameters(), 'lr': lr}], momentum=0.9)
    # optimizer_disc = optim.SGD([{'params': disc.parameters(), 'lr': lr*10}], momentum=0.9)
    optimizer = optim.AdamW([{'params': model.parameters(), 'lr': lr}], betas=(0.8, 0.99))
    optimizer_disc = optim.AdamW([{'params': disc.parameters(), 'lr': lr}], betas=(0.8, 0.99))

    def train(epoch):
        last_loss = 0
        train_d = False
        print('----------------------------------------Epoch: {}----------------------------------------'.format(epoch))
        for batch_idx, input_wav in enumerate(trainloader):
            train_d = not train_d
            input_wav = input_wav.cuda()
            optimizer.zero_grad()
            model.zero_grad()
            optimizer_disc.zero_grad()
            disc.zero_grad()
            output, loss_enc, _ = model(input_wav)

            logits_real, fmap_real = disc(input_wav)
            if train_d:
                logits_fake, _ = disc(model(input_wav)[0].detach())
                loss = disc_loss(logits_real, logits_fake)
                if loss > last_loss/2:
                    loss.backward()
                    optimizer_disc.step()
                last_loss = 0

            logits_fake, fmap_fake = disc(output)
            loss = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output)
            last_loss += loss.item()
            loss_enc.backward(retain_graph=True)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(torch.cuda.mem_get_info())
                print(f"Train Epoch: {epoch} [{batch_idx * len(input_wav)}/{len(trainloader.dataset)} ({100. * batch_idx / len(trainloader):.0f}%)]")


    def adjust_learning_rate(optimizer, epoch):
        if epoch % 80 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1


    for epoch in range(1, max_epoch):

        train(epoch)
        torch.save(model.state_dict(), f'{SAVE_LOCATION}epoch{epoch}.pth') #epoch{epoch}.pth
        torch.save(disc.state_dict(), f'{SAVE_LOCATION}epoch{epoch}_disc.pth')

        adjust_learning_rate(optimizer, epoch)
        adjust_learning_rate(optimizer_disc, epoch)

training(max_epoch=MAX_EPOCH, log_interval=100, fixed_length=0, batch_size=BATCH_SIZE, tensor_cut=TENSOR_CUT)

