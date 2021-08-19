from dataclasses import dataclass

from deepspeech_pytorch.enums import DecoderType


@dataclass
class LMConfig:
    decoder_type: DecoderType = DecoderType.beam
    lm_path: str = 'kenlm_fa2.arpa'  # Path to an (optional) kenlm language model for use with beam search (req\'d with trie
    top_paths: int = 256  # Number of beams to return
    # alpha: float = 0.00895  # Language model weight
    # beta: float = 0.9965  # Language model word bonus (all words)
    #kenlm_fa.arpa, width=10
    # alpha : float = 0.852
    # beta : float = 0.356

    #kenlm2, width=256
    alpha: float = 0.7682
    beta: float = 0.2779
    # alpha: float = 0.852
    # beta: float = 0.356


    cutoff_top_n: int = 60  # Cutoff_top_n characters with highest probs in vocabulary will be used in beam search
    cutoff_prob: float = 1.0  # Cutoff probability in pruning,default 1.0, no pruning.
    beam_width: int = 256  # Beam width to use
    lm_workers: int = 1  # Number of LM processes to use


@dataclass
class ModelConfig:
    precision: int = 32  # Set to 16 to use mixed-precision for inference
    cuda: bool = True
    # model_path: str = '/home/avir/Desktop/cyon/deepspeech.pytorch/outputs/2021-06-21/12-55-13/models/epoch=79-step=35199.ckpt'
    # model_path : str = '/home/avir/Desktop/cyon/deepspeech.pytorch/outputs/2021-06-23/19-41-37/models/epoch=36-step=16279.ckpt'
    # model_path: str = '/home/avir/Desktop/cyon/deepspeech.pytorch/outputs/2021-06-24/16-44-30/models/epoch=246-step=108673.ckpt'
    #model_path : str = 'epoch87-step38719.ckpt'
    #model_path : str = '/home/kngpu/mahsa/deepspeech.pytorch/outputs/2022-01-18/19-44-38/models/epoch=14-step=6599.ckpt'
    #model_path : str = '/home/kngpu/mahsa/deepspeech.pytorch/outputs/2022-01-19/22-27-49/models/epoch=144-step=63799.ckpt' #label_v4=35
    model_path: str = '/home/kngpu/mahsa/deepspeech.pytorch/outputs/2022-01-22/14-52-45/models/epoch=203-step=89759.ckpt'

@dataclass
class InferenceConfig:
    lm: LMConfig = LMConfig()
    model: ModelConfig = ModelConfig()


@dataclass
class TranscribeConfig(InferenceConfig):
    audio_path: str = ''  # Audio file to predict on
    offsets: bool = False  # Returns time offset information


@dataclass
class EvalConfig(InferenceConfig):
    test_path: str = 'data/mozilla_test_manifest.json'  # Path to validation manifest csv or folder
    verbose: bool = True  # Print out decoded output and error of each sample
    save_output: str = 'mozilla_output.json'  # Saves output of model from test to this file_path
    batch_size: int = 16  # Batch size for testing
    num_workers: int = 1


@dataclass
class ServerConfig(InferenceConfig):
    host: str = '0.0.0.0'
    port: int = 8888
