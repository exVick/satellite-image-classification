# Author: Viktor Doychev 
# ID: k12441809

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from shiny import reactive
from shiny.express import input, render, ui


device = torch.device("cpu")
model_path = Path(__file__).resolve().parent.parent / "assets" / "weights" / "best_model_params.pth"

# mean and std per channel on training set
mean, std = ((0.34372076990464207, 0.3811175644818459, 0.40848065640318615), (0.20149977112034764, 0.13648379098261232, 0.11620906916470537))

classes = [
    "HerbaceousVegetation",
    "AnnualCrop",
    "Residential",
    "Pasture", 
    "Industrial",
    "River", 
    "Highway",
    "Forest",
    "PermanentCrop",
    "SeaLake",
]

# same as in the notebook
val_tfms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


class SatelliteCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # [3,64,64] -> [32,64,64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # [32,64,64] -> [32,32,32]
        )
        self.vgg_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # [32,32,32] -> [64,32,32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # [64,32,32]->[64,16,16]
        )
        self.vgg_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # [64,16,16] -> [128,16,16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # [128,16,16] -> [128,8,8]
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(128*8*8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,10)
        )

    def forward(self, x):
        x = self.vgg_block1(x)
        x = self.vgg_block2(x)
        x = self.vgg_block3(x)
        x = torch.flatten(x, 1) # [batch,128,8,8] -> [batch,128*8*8]  
        x = self.linear_stack(x)
        return x


model = SatelliteCNN()
_ = model.to(device)
state_dict = torch.load(model_path, map_location=device)
_ = model.load_state_dict(state_dict)
_ = model.eval()


def predict(img: Image.Image) -> pd.DataFrame:

    img = img.convert("RGB")
    # apply val transforms and add batch dimension [3, 64, 64] -> [1, 3, 64, 64].
    tensor = val_tfms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    df = pd.DataFrame({"Class": classes, "Probability": probs})
    df = df.sort_values("Probability", ascending=False).reset_index(drop=True)
    return df



uploaded_image = reactive.value(None)
predictions = reactive.value(None)


@reactive.effect
@reactive.event(input.imageFile)
def load_image():
    file_info = input.imageFile()
    # store the temp path of the uploaded file (it lives outside the app dir)
    uploaded_image.set(file_info[0]["datapath"])
    predictions.set(None) # reset previous


@reactive.effect
@reactive.event(input.predictButton)
def run_prediction():
    if uploaded_image.get() is not None:
        img = Image.open(uploaded_image.get())
        predictions.set(predict(img))



ui.page_opts(title="Satellite Image Classifier")

with ui.sidebar(title="Upload image"):
    ui.input_file(
        id="imageFile",
        label="Select satellite image",
        accept=["image/*"],
        multiple=False,
    )
    ui.input_action_button(id="predictButton", label="Predict class")


with ui.layout_columns():

    with ui.card():
        ui.card_header("Image")

        @render.image
        def show_image():
            if uploaded_image.get() is None:
                return None
            # serve the uploaded temp file directly (no writing into the app dir)
            return {"src": uploaded_image.get(), "width": "256px", "height": "256px"}

    with ui.card():
        ui.card_header("Class Probabilities")

        @render.data_frame
        def show_table():
            if predictions.get() is None:
                return pd.DataFrame({"Class": [], "Probability": []})
            df = predictions.get().copy()
            df["Probability"] = df["Probability"].map(lambda p: f"{p:.4f}")
            return render.DataGrid(df, width="100%")


with ui.card():
    ui.card_header("Prediction")

    @render.ui
    def show_prediction():
        if predictions.get() is None:
            return ui.p("Upload an image and click 'Predict class' to see the result")
        df = predictions.get()
        top = df.iloc[0]
        return ui.h4(
            f"Predicted Class: {top['Class']} "
            f"(Confidence: {top['Probability'] * 100:.2f}%)"
        )
