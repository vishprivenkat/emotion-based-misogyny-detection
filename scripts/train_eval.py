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



def train_model(train_loader, epochs=3, model_name="MilaNLProc/bert-base-uncased-ear-misogyny", path_to_save='./models/ami-reddit.pt', num_classes=2, model_lr=1e-05, save_model=True): 
  config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
  model = AutoModelClass.from_pretrained(model_name, config=config)
  model.to(device)
  optimizer = torch.optim.Adam(params =  model.parameters(), lr=model_lr)
  if num_classes==2:
    loss_function = torch.nn.BCELoss()
  for epoch in range(epochs):
      print(f"Epoch : {epoch+1}")
      model.train()
      epoch_loss = 0
      epoch_acc = 0
      for steps, batch in tqdm(enumerate(train_loader, 0)):
          # Move the data to the GPU
          #print(batch['text'])
          input_ids = batch['input_ids'].to(device, dtype = torch.long)
          attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
          labels = batch['label'].to(device, dtype = torch.long)

          # Zero out the gradients
          optimizer.zero_grad()

          # Forward pass
          outputs = model.forward(input_ids, attention_mask=attention_mask, labels=labels)

          # Compute the loss
          #loss = loss_function(outputs.loss, labels)
          epoch_loss +=outputs.loss.detach() 
          _, max_indices = torch.max(outputs.logits, dim=1) 
          bath_acc = (max_indices==labels).sum().item()/labels.size(0)
          epoch_acc += bath_acc

          # Backward pass
          outputs.loss.backward()

          # Update the weights
          optimizer.step()
      print(f"Train Loss: {epoch_loss/steps}")
      print(f"Train Accuracy: {epoch_acc/steps}")

  if save_model==True: 
    torch.save(model, path_to_save)

  return model 

def evaluation(y_test, y_pred): 
  print("Plotting Confusion Matrix: ")
  cm = confusion_matrix(y_test, y_pred) 
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot()
  plt.show()

  print("Classification Report: ")
 
  print(classification_report(y_test, y_pred))


def eval_model(model_path, test_loader, complete_evaluation=False): 
  model = torch.load(model_path)
  model.eval()
  num_correct = 0
  num_total = 0
  preds, true_labels = [], []
  for steps, batch in tqdm(enumerate(test_loader,0))  :
          input_ids = batch['input_ids'].to(device, dtype = torch.long)
          attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
          labels = batch['label'].to(device, dtype = torch.long)

          outputs = model.forward(input_ids, attention_mask=attention_mask, labels=labels)
        
          logits = outputs.logits
          predictions = torch.argmax(logits, dim=-1)
          preds+=predictions.tolist()
          true_labels+=labels.tolist()
         
          num_correct += torch.sum(predictions == labels)
          num_total += len(labels)

  accuracy = float(num_correct) / num_total
  print(f" accuracy of the model:{accuracy}")

  if complete_evaluation: 
    evaluation(true_labels, preds) 
  
  return (true_labels, preds)


def driver(MODEL_PATH, input_file_path, BATCH_SIZE=8, 
           epochs = 3, text_attr = 'text', 
           label_attr='label', 
           mode='train', 
           model_name="MilaNLProc/bert-base-uncased-ear-misogyny",
           model_lr=1e-05, save_model=True, 
           complete_eval=True):
  if mode == 'train': 
    print("Loading Train Data: ...............")
    train_dataset = MisogynyDataset(input_file_path, text_attr, label_attr)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Training model: ............ ")
    train_model(train_loader, 
                epochs=epochs, 
                model_name=model_name, 
                path_to_save=MODEL_PATH, 
                num_classes=2, 
                model_lr=model_lr, 
                save_model=save_model)
  if mode == 'test': 
    print("Loading Test Data: ...............")
    test_dataset = MisogynyDataset(input_file_path, text_attr, label_attr)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print("Testing Model: ....................")
    true_labels, preds_labels = eval_model(MODEL_PATH, test_loader, complete_evaluation=complete_eval)

    #return (true_labels, preds_labels) 


if __name__=='__main__': 
  arg= argparse.ArgumentParser() 
  arg.add_argument('--model_path', type=str, required=True)
  arg.add_argument('--model_name', type=str, default="MilaNLProc/bert-base-uncased-ear-misogyny") 
  arg.add_argument('--input_file_path', type=str, required=True) 
  arg.add_argument('--batch_size', type=int, default=8) 
  arg.add_argument('--epochs', type=int, default=3) 
  arg.add_argument('--text_attr', type=str, default='text') 
  arg.add_argument('--label_attr', type=str, default='label')
  arg.add_argument('--mode', type=str, required=True) 
 
  arg.add_argument('--learning_rate', type=float, default=1e-05)
  arg.add_argument('--save_model', type=bool, default=True) 
  arg.add_argument('--complete_eval', type=bool, default=True) 

  args = arg.parse_args()

  driver(args.model_path, 
         args.input_file_path, 
         BATCH_SIZE = args.batch_size, 
         epochs = args.epochs, 
         text_attr = args.text_attr, 
         label_attr = args.label_attr, 
         mode = args.mode, 
         model_name = args.model_name, 
         model_lr = args.learning_rate, 
         save_model = args.save_model, 
         complete_eval = args.complete_eval
         )
  













