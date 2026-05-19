# 1. Imports
from fastai.vision.all import *
import matplotlib.pyplot as plt

# 2. THE FIX: The correct path
# Since your terminal is in 'Pneumonia_Project', we just go straight to 'data'
path = Path('data/chest_xray')

# --- SAFETY SENSOR ---
# This block checks if the folder exists and counts images before starting the AI
if not path.exists():
    print(f"❌ ERROR: Cannot find folder at {path.absolute()}")
    print("Check if your folder is named 'chest_xray' or something else.")
else:
    files = get_image_files(path)
    print(f"✅ Success! Found {len(files)} images.")
    print(f"Categories: {parent_label(files[0])}, {parent_label(files[-1])}")

# 3. DataBlock Blueprint
pneumonia_block = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=GrandparentSplitter(train_name='train', valid_name='test'),
    get_y=parent_label,
    item_tfms=Resize(224)
)

# 4. Create DataLoaders
# We add 'num_workers=0' which helps prevent errors on some Windows machines
dls = pneumonia_block.dataloaders(path, bs=16, num_workers=0)

# 5. Show results
dls.show_batch(max_n=9)
plt.show()