import os
from PIL import Image
import json
from datetime import datetime

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
except Exception as e:
    print('Missing transformers/torch:', e)
    raise

base = os.path.dirname(os.path.dirname(__file__))
vis_dir = os.path.join(base, 'learned_items', 'visual_samples')
out_file = os.path.join(base, 'learned_items', 'visual_embeddings.json')
out_samples_file = os.path.join(base, 'learned_items', 'visual_embeddings_samples.json')

print('Loading CLIP model...')
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model.eval()

embeddings = {}
sample_embeddings = []
count = 0
for label in sorted(os.listdir(vis_dir)):
    lab_path = os.path.join(vis_dir, label)
    if not os.path.isdir(lab_path):
        continue
    vectors = []
    for fn in sorted(os.listdir(lab_path)):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        p = os.path.join(lab_path, fn)
        try:
            img = Image.open(p).convert('RGB')
            inputs = processor(images=img, return_tensors='pt')
            with torch.no_grad():
                img_feat = model.get_image_features(**inputs)
            # normalize
            img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)
            vec = img_feat[0].cpu().numpy().tolist()
            vectors.append(vec)
            count += 1
        except Exception as e:
            print('Failed to process', p, e)
    if vectors:
        # average vectors
        import numpy as _np
        avg = _np.mean(_np.array(vectors), axis=0).tolist()
        embeddings[label] = {
            'embedding': avg,
            'count': len(vectors),
            'example': next((os.path.join(lab_path, f) for f in sorted(os.listdir(lab_path)) if f.lower().endswith(('.jpg','.jpeg','.png'))), None),
            'built_on': datetime.now().isoformat()
        }
        # also append per-sample vectors collected earlier
        for sv in vectors:
            sample_embeddings.append({'label': label, 'embedding': sv.tolist() if hasattr(sv, 'tolist') else list(sv)})

print(f'Processed {count} images for {len(embeddings)} labels')
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(embeddings, f, indent=2)
print('Saved visual embeddings to', out_file)
with open(out_samples_file, 'w', encoding='utf-8') as f:
    json.dump(sample_embeddings, f, indent=2)
print('Saved per-sample visual embeddings to', out_samples_file)
