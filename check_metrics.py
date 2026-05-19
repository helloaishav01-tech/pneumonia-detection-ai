from fastai.vision.all import *
import warnings
warnings.filterwarnings("ignore")

# 1. Setup path to your data (Make sure this path is correct!)
path = Path('c:/Users/Admin/OneDrive/Desktop/Pneumonia_Project/chest_xray')

# 2. Re-create a small DataLoader just to check metrics
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, item_tfms=Resize(224))

# 3. Load the model
learn = load_learner('pneumonia_model.pkl')
learn.dls = dls  # Inject the data so the model knows what to validate

# 4. Run validation
print("Calculating final metrics... please wait.")
val_results = learn.validate()

# FastAI returns [loss, accuracy, recall] based on your training settings
# Usually: Index 1 is Accuracy, Index 2 is Recall
print(f"Final Accuracy: {val_results[1]*100:.2f}%")
print(f"Final Recall: {val_results[2]*100:.2f}%")