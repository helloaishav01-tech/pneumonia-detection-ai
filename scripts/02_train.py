# 1. THE TOOLS
# We use FastAI for the AI logic and Path for folder management
from fastai.vision.all import *

# 2. THE DATA PATH
# We use the relative path that worked in our check script
path = Path('data/chest_xray')

# 3. THE DATA PIPELINE (The "Teacher")
# This organizes images into batches for the AI to study.
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=GrandparentSplitter(train_name='train', valid_name='test'),
    get_y=parent_label,
    item_tfms=Resize(460), # We resize large first to keep quality
    batch_tfms=aug_transforms(size=224, min_scale=0.75) # Final size for the brain
).dataloaders(path, bs=16, num_workers=0) # bs=16 is safer for most computers

# 4. THE BRAIN (The "Student")
# We use ResNet50. It's a deep network (50 layers) that is great at textures.
# We track 'accuracy' and 'Recall'. Recall is our "Safety Metric."
learn = vision_learner(dls, resnet50, metrics=[accuracy, Recall()])

# 5. THE LEARNING PROCESS
# fine_tune(3) tells the AI to go through the whole dataset 3 times.
# This will take some time (10–30 minutes depending on your computer).
print("🚀 Training started. Please do not close VS Code...")
learn.fine_tune(3)

# 6. SAVING THE RESULT
# We save the trained brain into our 'models' folder.
# This allows us to use it in our software later.
learn.export('models/pneumonia_model.pkl')
print("✅ Training Complete! Your AI model is saved in the 'models' folder.")