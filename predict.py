from fastai.vision.all import *

# 1. Load the model
learn_inf = load_learner('models/pneumonia_model.pkl')

# 2. Use the correct path found by the 'tree' command
# I've picked the first image from your NORMAL folder
test_image = r'data/chest_xray/test/NORMAL/IM-0001-0001.jpeg'

# 3. Predict
print(f"🔍 Analyzing: {test_image}")
pred, pred_idx, probs = learn_inf.predict(test_image)

print(f'\n--- ANALYSIS RESULT ---')
print(f'Prediction: {pred.upper()}')
print(f'Confidence: {probs[pred_idx]*100:.2f}%')
print(f'-----------------------')