# Classification with gemma3:4b (VLM) via SoonerAI

This project uses the OpenAI-compatible Chat Completions API at `https://ai.sooners.us/api/chat/completions` with **gemma3:4b** to classify **100 CIFAR-10 images** (10 per class), experiment with **system prompts**, and plot **confusion matrices**.

# Goal
Use an OpenAI-compatible API at https://ai.sooners.us with the gemma3:4b model (a VLM) to:

classify 100 images from CIFAR-10 Links to an external site. (10 images from each of the 10 classes),

experiment with different system prompts to improve accuracy, and

plot a confusion matrix from the results.


## Setup

Create ~/.soonerai.env  (should already be in place from previous assignment):

SOONERAI_API_KEY=your_key_here
SOONERAI_BASE_URL=https://ai.sooners.us
SOONERAI_MODEL=gemma3:4b

Install packages (inside your venv):
pip install requests python-dotenv torch torchvision pillow scikit-learn matplotlib
Recommended repo files:

cifar10_classify.py (provided template below)

requirements.txt (list the packages above)

README.md (how to run, and your analysis)

.gitignore (exclude *.env, __pycache__/, .ipynb_checkpoints/)

# What You Must Do
Sampling: Randomly sample 10 images per class (total 100). Use a fixed seed for reproducibility.

Classification: For each image, call /api/chat/completions with gemma3:4b at https://ai.sooners.us.

Send the image as base64 in a user message with an image_url part using a Data URL (data:image/jpeg;base64,....).

Keep your system message as your prompt under test.

Parse the model’s top prediction as one of the 10 CIFAR-10 class names.

Metrics: Compute overall accuracy and plot a confusion matrix.

Deliverables as GitHub repo:

Code that runs end-to-end

A saved confusion-matrix image (e.g., confusion_matrix.png)

A short analysis section in README (what you tried; which prompt worked better; error patterns)


```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

