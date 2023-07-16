import librosa 
from pydub import AudioSegment
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
sampling_rate = 16000
signal = np.zeros((1, sampling_rate), dtype=np.float32)

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)

def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray: 

    
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = torch.from_numpy(y).to(device)
    print(y.shape)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y

def get_emotion_vectors(path_to_dir, file_name, destination_file_name): 
  dataset = pd.read_csv(path_to_dir+file_name)

  results = [] 
  for index, item in dataset.iterrows(): 
    current_file = path_to_dir + item['file_path'] 
    sound = AudioSegment.from_wav('/content/drive/MyDrive/CSCI 535 Project/Implementation/training_models'+j['file_path'])
    sound = sound.set_channels(1)
    sound.export(current_file, format="wav")
    y, s = librosa.load(current_file, sr=16000) # Downsample 44.1kHz to 8kHz
    results.append(process_func(y.reshape(1,y.shape[0]), s))
  

  arousal, valence, dominance = [], [], [] 
  for result in results: 
    arousal.append(result[0][0]) 
    valence.append(result[0][1])
    dominance.append(result[0][2]) 
  
  dataset['arousal'], dataset['valence'], dataset['dominance'] = arousal, valence, dominance
  dataset.to_csv(path_to_dir+destination_file_name) 
  print("Finished adding VAD vector values!")








