from datasets import load_dataset
ds = load_dataset('Jackrong/GLM-5.1-Reasoning-1M-Cleaned', split='train[:10000]')
 
ds.save_to_disk('data/glm_dataset/train_10k')
print("Dataset downloaded and saved to data/glm_dataset/train_10k")