import hydra
import torch

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder
from deepspeech_pytorch.validation import run_evaluation , model_out
import numpy as np

@torch.no_grad()
def evaluate(cfg: EvalConfig):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(
        device=device,
        model_path=cfg.model.model_path
    )

    decoder = load_decoder(
        labels=model.labels,
        cfg=cfg.lm
    )
    target_decoder = GreedyDecoder(
        labels=model.labels,
        blank_index=model.labels.index('_')
    )
    test_dataset = SpectrogramDataset(
        audio_conf=model.spect_cfg,
        input_path=hydra.utils.to_absolute_path(cfg.test_path),
        labels=model.labels,
        normalize=True
    )
#    with open('input_data.txt', 'a') as outfile:
 #       for do in  test_dataset.ids:
  #          txt = ""
   #         with open(do[1], 'r') as out_txt:
    #            txt = out_txt.readline()
     #       outfile.write(txt + ',' + do[0] + '\n')
    test_loader = AudioDataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )
    print(cfg.num_workers)
    #print(test_dataset.__dict__.items())
    wer, cer = run_evaluation(
    #model_out(
        test_loader=test_loader,
        device=device,
        model=model,
        decoder=decoder,
        target_decoder=target_decoder,
        precision=cfg.model.precision
    )

    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))

   # ins = []
   # outs = []
    #with open('output_data.txt', 'r') as outfile:
    #   o = outfile.readlines()
    #    ou = ''
    #    outs = [val.replace('\n','') for val in o]
    #with open('input_data.txt', 'r') as infile:
    #    ii = infile.readlines()
#        ins = [val.replace('\n','') for val in ii]
#    with open('mergerd.txt', 'a') as ofile:
#        for i, do in enumerate(outs):
#            ofile.write(outs[i] + ',' + ins[i] + '\n')
