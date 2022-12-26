# EnCodec_Trainer

Implementation to add training function with loss to https://github.com/facebookresearch/encodec \
Used audio_to_mel.py from https://github.com/descriptinc/melgan-neurips \
Some minor adjustments to other code to add new model. \
Only training.py & customAudioDataset.py is new code. \
Download the used database e-gmd from https://magenta.tensorflow.org/datasets/e-gmd

## Citation
If you use the original code or results in your paper, please cite the original work as:
```
@article{defossez2022highfi,
  title={High Fidelity Neural Audio Compression},
  author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}

@article{Melgan,
      title={MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis}, 
      author={Kundan Kumar and Rithesh Kumar and Thibaukt de Boissiere and Lucas Gestin and Wei Zhen Teoh and Jose Sotelo and Alexandre de Brebisson and Yoshua Bengio and Aaron Courville},
      journal={arXiv preprint arXiv:1910.06711},
      year={2019}
}

@misc{egmd,
    title={Improving Perceptual Quality of Drum Transcription with the Expanded Groove MIDI Dataset},
    author={Lee Callender and Curtis Hawthorne and Jesse Engel},
    year={2020},
    eprint={2004.00188},
    archivePrefix={arXiv},
    primaryClass={cs.SD},
}
```

Also citing the added training code if you use it is always appreciated.