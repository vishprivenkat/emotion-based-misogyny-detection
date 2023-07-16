import pandas as pd 
import numpy as np 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoConfig
import torch 
from torch import cuda
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
import argparse 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns 
import torch.nn.functional as F 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MisogynyDataset(Dataset):
    def __init__(self, file_path, text_attribute, target_attribute):
        self.tokenizer = AutoTokenizer.from_pretrained("MilaNLProc/bert-base-uncased-ear-misogyny")
        self.data = pd.read_csv(file_path)
        #self.data['text'] = self.data['text'].applymap(str)
        self.text_attr = text_attribute
        self.label_attr = target_attribute 

        self.max_len = 512
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        text = self.data.loc[index, self.text_attr]
        label = self.data.loc[index, self.label_attr ]
        

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        
        return {'input_ids': encoding['input_ids'][0], 'attention_mask': encoding['attention_mask'][0], 'label': label, 'text': text}

class AutoModelClass(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = super().forward(input_ids, attention_mask=attention_mask)
        logits = self.linear(outputs[0])
        return logits


def eval_model(model_path, test_loader, local_model=False): 
  if not local_model: 
    model = torch.load(model_path)
  else: 
    model = AutoModelClass.from_pretrained(model_path)
  model.to(device)
  model.eval()
  num_correct = 0
  num_total = 0
  preds, true_labels = [], []
  
  prediction_probabilities = []

  for steps, batch in tqdm(enumerate(test_loader,0))  :
          input_ids = batch['input_ids'].to(device, dtype = torch.long)
          attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
          labels = batch['label'].to(device, dtype = torch.long)

          outputs = model.forward(input_ids, attention_mask=attention_mask, labels=labels)
        
          logits = outputs.logits 
          
          prediction_probabilities.append(torch.sigmoid(logits).cpu().detach().numpy()) 

          predictions = torch.argmax(logits, dim=-1)
          preds+=predictions.tolist()
          true_labels+=labels.tolist()
         
          num_correct += torch.sum(predictions == labels)
          num_total += len(labels)

  accuracy = float(num_correct) / num_total
  print(f" accuracy of the model:{accuracy}")
  
  torch.cuda.empty_cache()
  del model 
  return ( np.array(np.concatenate(prediction_probabilities, axis=0)), np.array(true_labels) )

def plot_confusion_matrix(cm): 
  sns.heatmap(cm, annot=True, cmap='Blues')

  # add labels to the plot
  plt.xlabel('Predicted labels')
  plt.ylabel('True labels')
  plt.title('Confusion Matrix')
  plt.show()


def calculate_label(probs_model_1, probs_model_2): 
  average_probs = (probs_model_1+probs_model_2)/2
 
  return np.argmax(average_probs, axis=1)

  

def evaluation(y_test, y_pred): 
  print("Classification Report for Averaged Probabilities: ")
  print(classification_report(y_test, y_pred, digits=3))
  plot_confusion_matrix(confusion_matrix(y_test, y_pred)) 


def driver_eval(model_path_1, model_path_2, input_file_path, BATCH_SIZE=8, 
           text_attr = 'text', 
           label_attr='label',  
           local_model_1=False, 
           local_model_2=False,
           ):
  test_dataset = MisogynyDataset(input_file_path, text_attr, label_attr)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
  print("Testing Model: ....................")
  probs_model_1, true_labels_1 = eval_model(model_path_1, test_loader,  local_model=local_model_1) 
  probs_model_2, true_labels_2 = eval_model(model_path_2, test_loader, local_model=local_model_2) 

  if np.array_equal(true_labels_1, true_labels_2): 
    y_pred = calculate_label(probs_model_1, probs_model_2) 
    evaluation(true_labels_1, y_pred) 




if __name__ == '__main__': 
  arg = argparse.ArgumentParser()
  arg.add_argument('--model_path_1', type=str, required=True)
  arg.add_argument('--model_path_2', type=str, required=True)
  arg.add_argument('--input_file_path', type=str, required=True) 
  arg.add_argument('--batch_size', type=int, default=8)
  arg.add_argument('--text_attr', type=str, default='text') 
  arg.add_argument('--label_attr', type=str, default='label')
  arg.add_argument('--local_model_1', type=bool, default=False) 
  arg.add_argument('--local_model_2', type=bool, default=False) 

  args = arg.parse_args() 

  driver_eval(
    args.model_path_1, 
    args.model_path_2, 
    args.input_file_path, 
    BATCH_SIZE = args.batch_size, 
    text_attr = args.text_attr, 
    label_attr = args.label_attr, 
    local_model_1 = args.local_model_1,
    local_model_2 = args.local_model_2
  )
  








  

