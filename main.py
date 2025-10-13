from custom_model import Model
from get_data import Data
import torch

data = Data(root='C:/Users/hp/Desktop/grad_project_2/gisture_classifier/sample_data/train', batch_size=4)

train_loader, num_classes = data.get_data()

model_instance = Model(data_batch=train_loader, num_classes=num_classes, pre_trained=False)
trained_model = model_instance.fine_tune()

acc = model_instance.evaluate()
print(f'accuracy: {acc}')

torch.save(trained_model.state_dict(), "resnet50_fine_tuned.pth")
print(" Model saved successfully.")
