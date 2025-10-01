import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.args import parse_arguments
from transformer_maskgit import CTViT
from data_inference_nii import CTReportDatasetinfer
from eval import evaluate_internal, sigmoid
import tqdm
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F

def sigmoid(tensor):
    return 1 / (1 + torch.exp(-tensor))

class ProjectionVIT(nn.Module):
    def __init__(self):
        super(ProjectionVIT, self).__init__()
        self.VIT = CTViT(
            dim = 512,
            codebook_size = 8192,
            image_size = 480,
            patch_size = 20,
            temporal_patch_size = 10,
            spatial_depth = 4,
            temporal_depth = 4,
            dim_head = 32,
            heads = 8
        )
        self.projection_layer = nn.Linear(294912, 512, bias = False)
    
    def forward(self, x):
        x = self.VIT(x, return_encoded_tokens=True)

        x = torch.mean(x, dim=1)
        x = x.view(x.shape[0], -1)
        x = x[:, :] if x.ndim == 3 else x

        x = self.projection_layer(x)
        x = F.normalize(x, dim = -1)
        return x

class ClassifierHead(nn.Module):
    def __init__(self, latent_dim=512, num_classes=18, dropout_prob=0.3):
        super(ClassifierHead, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        x = self.relu(x)
        x = self.dropout(x)
        return self.classifier(x)

def evaluate_model(args, vit_model, classifier_head, dataloader, device):
    vit_model.eval()  # Set the model to evaluation mode
    vit_model = vit_model.to(device)

    classifier_head.eval()
    classifier_head = classifier_head.to(device)

    correct = 0
    total = 0
    predictedall=[]
    realall=[]
    logits = []
    accs = []
    with torch.no_grad():

        for batch in tqdm.tqdm(dataloader):
            inputs, _, labels, acc_no = batch
            labels = labels.float().to(device)
            inputs = inputs.to(device)
            output = vit_model(inputs)
            # torch.save(output, f"/home/borntowarn/projects/chest-diseases/training/data/CT-RATE/cached_latents/vit_final1/{acc_no[0]}.pt")
            output = classifier_head(output)
            realall.append(labels.detach().cpu().numpy()[0])
            save_out = sigmoid(torch.tensor(output)).cpu().numpy()
            predictedall.append(save_out[0])
            accs.append(acc_no[0])
            print(acc_no[0], flush=True)

        plotdir = args.save
        os.makedirs(plotdir, exist_ok=True)
        logits = np.array(logits)

        with open(f"{plotdir}/accessions.txt", "w") as file:
            for item in accs:
                file.write(item + "\n")

        pathologies = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']

        realall=np.array(realall)
        predictedall=np.array(predictedall)

        np.savez(f"{plotdir}/labels_weights.npz", data=realall)
        np.savez(f"{plotdir}/predicted_weights.npz", data=predictedall)

        dfs=evaluate_internal(predictedall,realall,pathologies, plotdir)

        writer = pd.ExcelWriter(f'{plotdir}/aurocs.xlsx', engine='xlsxwriter')

        dfs.to_excel(writer, sheet_name='Sheet1', index=False)

        writer.close()




if __name__ == '__main__':
    args = parse_arguments()  # Assuming this function provides necessary arguments

    args.data_folder = "/home/borntowarn/projects/chest-diseases/training/data/CT-RATE/dataset/valid_fixed"
    args.reports_file = "/home/borntowarn/projects/chest-diseases/training/data/CT-RATE/dataset/radiology_text_reports/validation_reports.csv"
    args.meta_file = "/home/borntowarn/projects/chest-diseases/training/data/CT-RATE/dataset/metadata/validation_metadata.csv"
    args.labels = "/home/borntowarn/projects/chest-diseases/training/data/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
    args.save = "/home/borntowarn/projects/chest-diseases/training/CT-CLIP/scripts/res_new1"

    projectionVIT = ProjectionVIT()
    projectionVIT.load_state_dict(torch.load("/home/borntowarn/projects/chest-diseases/training/weights/CT-CLIP/ProjectionVIT_LiPro_V2.pt"))

    num_classes = 18
    image_classifier = ClassifierHead()
    image_classifier.load_state_dict(torch.load("/home/borntowarn/projects/chest-diseases/training/weights/CT-CLIP/ClassifierHead_LiPro_V2.pt"))

    ds = CTReportDatasetinfer(data_folder=args.data_folder, reports_file=args.reports_file, meta_file=args.meta_file, labels = args.labels)
    dl = DataLoader(ds, num_workers=8, batch_size=1, shuffle=False)

    evaluate_model(args, projectionVIT, image_classifier, dl, torch.device('cuda'))