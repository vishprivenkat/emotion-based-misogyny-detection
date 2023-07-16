import torch 
from transformers import AutoTokenizer, AutoModel 
import argparse 
import pandas as pd 
from torch.utils.data import Dataset 
import json
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BertEmbeddings():
  def __init__(self, model_name='MilaNLProc/bert-base-uncased-ear-misogyny'):
     self.model = AutoModel.from_pretrained(model_name, output_hidden_states = True).to(device)
     self.tokenizer = AutoTokenizer.from_pretrained(model_name)
  
  def get_embeddings(self, sentence): 
    tokenized = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt').to(device) 
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    token_type_ids = tokenized['token_type_ids'].to(device)
    # Pass input through BERT model
    with torch.no_grad():
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Use mean of last hidden state as sentence embedding
    
    return embedding


def process_embeddings( file_path, destination_file_path, model_name = 'bert-based-uncased', text_field='transcripts', id_field='file_path'):
  dataset = pd.read_csv(file_path) 
  sentences = dataset[text_field].to_list()
  files = dataset[id_field].to_list()
  embeddings = [] 
  embed = BertEmbeddings() 
  for i in range(len(sentences)): 
    embeddings.append(embed.get_embeddings(sentences[i])) 

  

  

  # Convert data to a JSON string


  with open(destination_file_path, 'wb') as f:
    pickle.dump(embeddings, f)




if __name__ == '__main__': 
  arg= argparse.ArgumentParser() 
  arg.add_argument('--model_name', type=str, default='bert-based-uncased')
  arg.add_argument('--file_path', type=str, required=True ) 
  arg.add_argument('--text_field', type=str, default='transcript') 
  arg.add_argument('--destination_file_path', type=str, required=True) 
  arg.add_argument('--id_field', type=str, default='file_path') 
  args = arg.parse_args()
  process_embeddings(
    args.file_path, 
    args.destination_file_path, 
    model_name = args.model_name, 
    text_field = args.text_field, 
    id_field = args.id_field 
  )






'''
class MisogynyData(Dataset): 
  def __init__(self, sentences, embeddings, valence, dominance, arousal, label):
    self.sentences = sentences
    self.embeddings = embeddings.get_embeddings(sentences)
    self.valence = valence 
    self.dominance = dominance 
    self.arousal = arousal 
    self.label = label 

  def __len__(self):
    return len(self.sentences)
    
  def __getitem__(self, index):
    sentence = self.sentences[index]
    embedding = self.embeddings[index] 
    valence = self.valence[index] 
    dominance = self.dominance[index] 
    arousal = self.arousal[index]
    
    # Process the embedding and sentence data and return it as a dictionary or tuple
    # For example, you could return {'embedding': embedding, 'sentence': sentence}
    
    return {'embedding': embedding, 'sentence': sentence}
''' 




