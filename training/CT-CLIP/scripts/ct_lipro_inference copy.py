import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.args import parse_arguments
from transformers import BertTokenizer, BertModel
from transformer_maskgit import CTViT
from ct_clip import CTCLIP
from data_inference_nii_my import CTReportDatasetinfer
from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis
import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import os
import copy

def sigmoid(tensor):
    return 1 / (1 + torch.exp(-tensor))

class ImageLatentsClassifier(nn.Module):
    def __init__(self, trained_model, latent_dim, num_classes, dropout_prob=0.3):
        super(ImageLatentsClassifier, self).__init__()
        self.trained_model = trained_model
        self.dropout = nn.Dropout(dropout_prob)  # Add dropout layer
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(latent_dim, num_classes)  # Assuming trained_model.image_latents_dim gives the size of the image_latents

    def forward(self, image_latents):
        image_latents = torch.nn.functional.normalize(image_latents, dim=-1)
        image_latents = self.relu(image_latents)
        image_latents = self.dropout(image_latents)  # Apply dropout on the latents
        return self.classifier(image_latents)

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        loaded_state_dict = torch.load(file_path)
        self.load_state_dict(loaded_state_dict)

def evaluate_model(args, model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)
    correct = 0
    total = 0
    predictedall=[]
    realall=[]
    logits = []
    accs = []
    with torch.no_grad():

        for batch in tqdm.tqdm(dataloader):
            _, _, labels, acc_no = batch
            folder1, folder2 = acc_no[0].rsplit("_", 2)[0], acc_no[0].rsplit("_", 1)[0]
            embedding = torch.load(f"/home/borntowarn/projects/chest-diseases/training/data/CT-RATE/dataset/valid_fixed_embeds_not_normalized_lipro/{folder1}/{folder2}/{acc_no[0].replace('.gz', '.pt')}")
            labels = labels.float().to(device)
            inputs = embedding.to(device)
            output = model(inputs)
            realall.append(labels.detach().cpu().numpy()[0])
            save_out = sigmoid(torch.tensor(output)).cpu().numpy()
            predictedall.append(save_out)
            accs.append(acc_no[0])
            # print(acc_no[0], flush=True)

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

        print(len(predictedall), len(realall))
        print(predictedall, realall)
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
    args.pretrained = "/home/borntowarn/projects/chest-diseases/training/weights/CT-RATE/models/CT-CLIP-Related/CT_LiPro_v2.pt"
    args.save = "/home/borntowarn/projects/chest-diseases/training/CT-CLIP/scripts/res1"

    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    text_encoder.resize_token_embeddings(len(tokenizer))

    image_encoder = CTViT(
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

    clip = CTCLIP(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_image = 294912,
        dim_text = 768,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False

    )

    num_classes = 18  # you need to specify the number of classes here
    image_classifier = ImageLatentsClassifier(clip, 512, num_classes)
    zero_shot = copy.deepcopy(image_classifier)

    image_classifier.load(args.pretrained)  # Assuming args.checkpoint_path is the path to the saved checkpoint

    # Prepare the evaluation dataset
    ds = CTReportDatasetinfer(data_folder=args.data_folder, reports_file=args.reports_file, meta_file=args.meta_file, labels = args.labels)
    dl = DataLoader(ds, num_workers=8, batch_size=1, shuffle=False)

    # Evaluate the model
    evaluate_model(args, image_classifier, dl, torch.device('cuda'))