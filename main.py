# Inspired by https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb

# Imports
from imports import *
num_classes=6
num_workers = 0 # 0 for GPU, up to 8 for CPU
max_epochs = 1
batch_size = 1
fdr = False
logit_threshold = 1.5
p_threshold = 0.7
lr = 1e-4 ; lr_backbone = 1e-5 ; weight_decay=1e-4
gradient_clip_val = 0.1
gpu=1

### COCO dataset ###
coco=COCO("content/content_train/trainJson.json")
print("coco.getCatIds()",coco.getCatIds())
cats = coco.loadCats(coco.getCatIds())

nms=[cat['name'] for cat in cats]
print('Categories: {}'.format(nms))

nms = set([cat['supercategory'] for cat in cats])

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "content_train/trainJson.json" if train else "content_test/testJson.json")
        img_folder = os.path.join(img_folder, "content_train/trainData" if train else "content_test/testData")
        if os.path.lexists(os.path.join(img_folder, ".DS_Store")): 
          os.remove(os.path.join(img_folder, ".DS_Store"))
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

train_dataset = CocoDetection(img_folder= "content/", feature_extractor=feature_extractor)
val_dataset = CocoDetection(img_folder= "content/", feature_extractor=feature_extractor, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

## DataLoader objects for training and validation ##
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers, shuffle=False)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers)

## get some items (image, annotations) to show before, during, after training ##
iterator = iter(train_dataloader)
batch = next(iterator)
batch = next(iterator)
batch1 = next(iterator)
batch2 = next(iterator)

print(batch.keys())
print("\nbatch['labels']", batch['labels'])
print("\nbatch1['labels']", batch1['labels'])
print("\nbatch2['labels']", batch2['labels'])

pixel_values, target = train_dataset[0] # careful, there is a shuffle!!!!!
print("\npixel_values.shape", pixel_values.shape)


### Create DETR object class ###
class Detr(pl.LightningModule):

     def __init__(self, lr, lr_backbone, weight_decay):
         super().__init__()
         # replace COCO classification head with custom head
         self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", 
                                                             num_labels=num_classes,
                                                             ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         # see https://github.com/huggingface/transformers/issues/12643 : ignore warnings
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs
     
     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       #labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
       labels = [{k: v for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)
        
        return optimizer

     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader


model = Detr(lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay)

### Load checkpoint ###
## First option: load latest checkpoint from Facebook AI GitHub repo ##
# see https://github.com/facebookresearch/detr/issues/9#issuecomment-636391562
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=num_classes)

# Get pretrained weights
checkpoint = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
            map_location='cpu',
            check_hash=True)
del checkpoint["model"]["class_embed.weight"]
del checkpoint["model"]["class_embed.bias"]
model.load_state_dict(checkpoint["model"], strict=False)

# Save
torch.save(checkpoint, 'detr-r50_no-class-head-checkpoint.pth')

## Second option: load latest checkpoint (created after running main.py) ##
checkpoint = torch.load('checkpoint6.pth')
model.load_state_dict(checkpoint)

# Forward pass on selected items #
outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
outputs1 = model(pixel_values=batch1['pixel_values'], pixel_mask=batch1['pixel_mask'])
outputs2 = model(pixel_values=batch2['pixel_values'], pixel_mask=batch2['pixel_mask'])

print("\noutputs.logits.shape", outputs.logits.shape)

## Evaluate ##
base_ds = get_coco_api_from_dataset(val_dataset)
iou_types = ['bbox']
coco_evaluator = CocoEvaluator(base_ds, iou_types) # initialize evaluator with ground truths
# apparently no need to device with lightning module
#device = torch.device("gpu")
#model.to(device)
#model.eval()

print("Running evaluation 0...")
for i_b, b in enumerate(val_dataloader):
    # get the inputs
    pixel_values = b["pixel_values"]
    pixel_mask = b["pixel_mask"]
    labels = [{k: v for k, v in t.items()} for t in b["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
    res = {target['image_id'].item(): output for target, output in zip(labels, results)}
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()

############ DRAWING 1 ############
print("DRAWING 1")
CLASSES = ['N/A', 'person', 'bike', 'helmet', 'phone', 'airbag', 'N/A']
#COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
#          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933], [0.000, 0.447, 0.741]]
COLORS = ['red', 'green', 'blue', 'black', 'white', 'red']
fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)

td = batch['labels'][0] ; td1=batch1['labels'][0] ; td2=batch2['labels'][0]
img_id = td['image_id'].item() ; img_id1 = td1['image_id'].item() ; img_id2 = td2['image_id'].item()
orig_size_td = td['orig_size'] ; orig_size_td1 = td1['orig_size'] ; orig_size_td2 = td2['orig_size'] 
size_td = td['size'] ; size_td1 = td1['size'] ; size_td2 = td2['size'] 
print("\n\nimg_id", img_id, "orig_size_td", orig_size_td, "size_td", size_td)


print('Image n°{}'.format(img_id))
print('Image1 n°{}'.format(img_id1))
print('Image2 n°{}'.format(img_id2))
image = train_dataset.coco.loadImgs(img_id)[0]
image1 = train_dataset.coco.loadImgs(img_id1)[0]
image2 = train_dataset.coco.loadImgs(img_id2)[0]
cats = train_dataset.coco.cats
print("\ncats", cats)
id2label = {k: v['name'] for k,v in cats.items()}
print("\nid2label", id2label)

annotations = train_dataset.coco.imgToAnns[img_id]
annotations1 = train_dataset.coco.imgToAnns[img_id1]
annotations2 = train_dataset.coco.imgToAnns[img_id2]

def draw_annotations(image=image, img_id=img_id, annotations=annotations):
  with Image.open(os.path.join("content/content_train/trainData/", image['file_name'])) as im:
    draw = ImageDraw.Draw(im, "RGBA")
    for annotation in annotations:
      box = annotation['bbox']
      class_idx = annotation['category_id']
      x,y,w,h = tuple(box)
      draw.rectangle((x,y,x+w,y+h), outline='red', width=2)
      draw.text((x, y), id2label[class_idx], fill='white', font=fnt)
    # save
    im.save("output/annotated_img"+str(img_id)+".png", "PNG")
  return "output/annotated_img"+str(img_id)+".png file created"

draw_annotations(image, img_id, annotations)
draw_annotations(image1, img_id1, annotations1)
draw_annotations(image2, img_id2, annotations2)

pil_img = train_dataset.coco.loadImgs(img_id)[0]
pil_img = Image.open(os.path.join("content/content_train/trainData/", pil_img['file_name']))

pil_img1 = train_dataset.coco.loadImgs(img_id1)[0]
pil_img1 = Image.open(os.path.join("content/content_train/trainData/", pil_img1['file_name']))

pil_img2 = train_dataset.coco.loadImgs(img_id2)[0]
pil_img2 = Image.open(os.path.join("content/content_train/trainData/", pil_img2['file_name']))

size = orig_size_td; size1 = orig_size_td1; size2 = orig_size_td2

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), w, h]
    return (b[0], b[1], b[2], b[3])

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = np.array(box_cxcywh_to_xyxy(out_bbox))
    b = np.multiply(b,np.array([img_w, img_h, img_w, img_h]))
    return b

def plot_results(i, img_id, pil_img, prob, boxes):
    draw = ImageDraw.Draw(pil_img, "RGBA")
    colors = COLORS*20
    for p, (x,y,w,h), c in zip(prob, boxes.tolist(), colors):
      if max(p)>logit_threshold:
        #print("\nBoxes coords", (x,y,w,h))
        cl = p.argmax()
        #print("cl :", cl)
        if (cl!=0 and cl!=6):
          draw.rectangle((x,y,x+w,y+h), outline=colors[cl-1], width=2)
          text_c = f'{CLASSES[cl]}: {p[cl]:0.2f}'
          draw.text((x,y), text_c, fill='white', font=fnt)
    pil_img.save(f'output/prediction{i}_img'+str(img_id)+'.png', "PNG")

print("\n\nimg_id before training 1", img_id)

probs = outputs.logits[0]
pred_bbox = outputs.pred_boxes[0]
probs1 = outputs1.logits[0]
pred_bbox1 = outputs1.pred_boxes[0]
probs2 = outputs2.logits[0]
pred_bbox2 = outputs2.pred_boxes[0]

# Class with highest probability for each bbox
annot = torch.argmax(probs, dim=1)
print("Annotations", annot)

annot1 = torch.argmax(probs1, dim=1)
print("Annotations 1", annot1)

annot2 = torch.argmax(probs2, dim=1)
print("Annotations 2", annot2)

boxes = np.array([rescale_bboxes(b,size) for b in pred_bbox])
boxes1 = np.array([rescale_bboxes(b,size1) for b in pred_bbox1])
boxes2 = np.array([rescale_bboxes(b,size2) for b in pred_bbox2])
plot_results(1, img_id, pil_img, probs, boxes)
plot_results(1, img_id1, pil_img1, probs1, boxes1)
plot_results(1, img_id2, pil_img2, probs2, boxes2)

print("1img_id", img_id, "of size ", size, " annotated ")
print("1img_id1", img_id1, "of size1 ", size1, " annotated ")
print("1img_id2", img_id2, "of size2 ", size2, " annotated ")

########################## TRAINING 1 ####################################
#model.train()

trainer = Trainer(gpus=gpu, max_epochs=max_epochs, gradient_clip_val=gradient_clip_val, fast_dev_run=fdr)
trainer.fit(model, train_dataloader, val_dataloader)

torch.save(model.state_dict(), 'checkpoint1.pth')

#outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
#outputs1 = model(pixel_values=batch1['pixel_values'], pixel_mask=batch1['pixel_mask'])
#outputs2 = model(pixel_values=batch2['pixel_values'], pixel_mask=batch2['pixel_mask'])

print(batch['labels'])
print("\noutputs.logits.shape", outputs.logits.shape)

#model.eval()

print("Running evaluation 1...")

base_ds = get_coco_api_from_dataset(val_dataset)
iou_types = ['bbox']
coco_evaluator = CocoEvaluator(base_ds, iou_types)

for i_b, b in enumerate(val_dataloader):
    # get the inputs
    pixel_values = b["pixel_values"]
    pixel_mask = b["pixel_mask"]
    labels = [{k: v for k, v in t.items()} for t in b["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    out = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = feature_extractor.post_process(out, orig_target_sizes) # convert outputs of model to COCO api
    res = {target['image_id'].item(): output for target, output in zip(labels, results)}
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()

##########################            ####################################

########################## TRAINING 2 ####################################

checkpoint1 = torch.load('checkpoint1.pth')
model.load_state_dict(checkpoint1)

#model.train()

trainer = Trainer(gpus=gpu, max_epochs=max_epochs, gradient_clip_val=gradient_clip_val, fast_dev_run=fdr)
trainer.fit(model, train_dataloader, val_dataloader)

torch.save(model.state_dict(), 'checkpoint2.pth')

#outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
#outputs1 = model(pixel_values=batch1['pixel_values'], pixel_mask=batch1['pixel_mask'])
#outputs2 = model(pixel_values=batch2['pixel_values'], pixel_mask=batch2['pixel_mask'])

print("Running evaluation 2...")

base_ds = get_coco_api_from_dataset(val_dataset)
iou_types = ['bbox']
coco_evaluator = CocoEvaluator(base_ds, iou_types)

for i_b, b in enumerate(val_dataloader):
    # get the inputs
    pixel_values = b["pixel_values"]
    pixel_mask = b["pixel_mask"]
    labels = [{k: v for k, v in t.items()} for t in b["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
    res = {target['image_id'].item(): output for target, output in zip(labels, results)}
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()

##########################            ####################################

########################## TRAINING 3 ####################################

checkpoint2 = torch.load('checkpoint2.pth')
model.load_state_dict(checkpoint2)

#model.train()

trainer = Trainer(gpus=gpu, max_epochs=max_epochs, gradient_clip_val=gradient_clip_val, fast_dev_run=fdr)
trainer.fit(model, train_dataloader, val_dataloader)

torch.save(model.state_dict(), 'checkpoint3.pth')

#outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
#outputs1 = model(pixel_values=batch1['pixel_values'], pixel_mask=batch1['pixel_mask'])
#outputs2 = model(pixel_values=batch2['pixel_values'], pixel_mask=batch2['pixel_mask'])

print("Running evaluation 3...")

base_ds = get_coco_api_from_dataset(val_dataset)
iou_types = ['bbox']
coco_evaluator = CocoEvaluator(base_ds, iou_types)

for i_b, b in enumerate(val_dataloader):
    # get the inputs
    pixel_values = b["pixel_values"]
    pixel_mask = b["pixel_mask"]
    labels = [{k: v for k, v in t.items()} for t in b["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
    res = {target['image_id'].item(): output for target, output in zip(labels, results)}
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()

##########################            ####################################

########################## TRAINING 4 ####################################

checkpoint3 = torch.load('checkpoint3.pth')
model.load_state_dict(checkpoint3)

#model.train()

trainer = Trainer(gpus=gpu, max_epochs=max_epochs, gradient_clip_val=gradient_clip_val, fast_dev_run=fdr)
trainer.fit(model, train_dataloader, val_dataloader)

torch.save(model.state_dict(), 'checkpoint4.pth')

outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
outputs1 = model(pixel_values=batch1['pixel_values'], pixel_mask=batch1['pixel_mask'])
outputs2 = model(pixel_values=batch2['pixel_values'], pixel_mask=batch2['pixel_mask'])

print("Running evaluation 4...")

base_ds = get_coco_api_from_dataset(val_dataset)
iou_types = ['bbox']
coco_evaluator = CocoEvaluator(base_ds, iou_types)

for i_b, b in enumerate(val_dataloader):
    # get the inputs
    pixel_values = b["pixel_values"]
    pixel_mask = b["pixel_mask"]
    labels = [{k: v for k, v in t.items()} for t in b["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
    res = {target['image_id'].item(): output for target, output in zip(labels, results)}
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()

##########################            ####################################

############ DRAWING 2 ############
print("DRAWING 2")
print('Image n°{}'.format(img_id))
print('Image1 n°{}'.format(img_id1))
print('Image2 n°{}'.format(img_id2))

# td0 = train_dataset[0]
# cats = train_dataset.coco.cats
# id2label = {k: v['name'] for k,v in cats.items()}

#annotations = train_dataset.coco.imgToAnns[img_id]

# Reload the images without annotations
pil_img = train_dataset.coco.loadImgs(img_id)[0]
pil_img = Image.open(os.path.join("content/content_train/trainData/", pil_img['file_name']))
pil_img1 = train_dataset.coco.loadImgs(img_id1)[0]
pil_img1 = Image.open(os.path.join("content/content_train/trainData/", pil_img1['file_name']))
pil_img2 = train_dataset.coco.loadImgs(img_id2)[0]
pil_img2 = Image.open(os.path.join("content/content_train/trainData/", pil_img2['file_name']))


def filter_bboxes_from_outputs(outputs, threshold=p_threshold):  
  # keep only predictions with confidence above threshold
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold
  probas_to_keep = probas[keep]
  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
  
  return probas_to_keep, bboxes_scaled

image = train_dataset.coco.loadImgs(img_id)[0]

annots = outputs.logits[0]
pred_bbox = outputs.pred_boxes[0]

# Class with highest probability for each bbox
annot = torch.argmax(annots, dim=1)
print("Annotations", annot)

prob = outputs.logits[0]
prob1 = outputs1.logits[0]
prob2 = outputs2.logits[0]

list_boxes=outputs.pred_boxes[0].tolist()
list_boxes1=outputs1.pred_boxes[0].tolist()
list_boxes2=outputs2.pred_boxes[0].tolist()

boxes = torch.tensor([rescale_bboxes(torch.tensor(b),size) for b in list_boxes])
boxes1 = torch.tensor([rescale_bboxes(torch.tensor(b),size1) for b in list_boxes1])
boxes2 = torch.tensor([rescale_bboxes(torch.tensor(b),size2) for b in list_boxes2])
plot_results(2, img_id, pil_img, prob, boxes)
plot_results(2, img_id1, pil_img1, prob1, boxes1)
plot_results(2, img_id2, pil_img2, prob2, boxes2)
print("ok cool2")

########################## TRAINING 5 ####################################

checkpoint4 = torch.load('checkpoint4.pth')
model.load_state_dict(checkpoint4)

#model.train()

trainer = Trainer(gpus=gpu, max_epochs=max_epochs, gradient_clip_val=gradient_clip_val, fast_dev_run=fdr)
trainer.fit(model, train_dataloader, val_dataloader)

torch.save(model.state_dict(), 'checkpoint5.pth')

#outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
#outputs1 = model(pixel_values=batch1['pixel_values'], pixel_mask=batch1['pixel_mask'])
#outputs2 = model(pixel_values=batch2['pixel_values'], pixel_mask=batch2['pixel_mask'])

print("Running evaluation 5...")

base_ds = get_coco_api_from_dataset(val_dataset)
iou_types = ['bbox']
coco_evaluator = CocoEvaluator(base_ds, iou_types)

for i_b, b in enumerate(val_dataloader):
    # get the inputs
    pixel_values = b["pixel_values"]
    pixel_mask = b["pixel_mask"]
    labels = [{k: v for k, v in t.items()} for t in b["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
    res = {target['image_id'].item(): output for target, output in zip(labels, results)}
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()

##########################            ####################################

########################## TRAINING 6 ####################################

checkpoint5 = torch.load('checkpoint5.pth')
model.load_state_dict(checkpoint5)

#model.train()

trainer = Trainer(gpus=gpu, max_epochs=max_epochs, gradient_clip_val=gradient_clip_val, fast_dev_run=fdr)
trainer.fit(model, train_dataloader, val_dataloader)

torch.save(model.state_dict(), 'checkpoint6.pth')

outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
outputs1 = model(pixel_values=batch1['pixel_values'], pixel_mask=batch1['pixel_mask'])
outputs2 = model(pixel_values=batch2['pixel_values'], pixel_mask=batch2['pixel_mask'])

print("Running evaluation 6...")

base_ds = get_coco_api_from_dataset(val_dataset)
iou_types = ['bbox']
coco_evaluator = CocoEvaluator(base_ds, iou_types)

for i_b, b in enumerate(val_dataloader):
    # get the inputs
    pixel_values = b["pixel_values"]
    pixel_mask = b["pixel_mask"]
    labels = [{k: v for k, v in t.items()} for t in b["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
    res = {target['image_id'].item(): output for target, output in zip(labels, results)}
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()

##########################            ####################################

############ DRAWING 3 ############
print("DRAWING 3")

print('Image n°{}'.format(img_id))
print('Image1 n°{}'.format(img_id1))
print('Image2 n°{}'.format(img_id2))

# Reload the images without annotations
pil_img = train_dataset.coco.loadImgs(img_id)[0]
pil_img = Image.open(os.path.join("content/content_train/trainData/", pil_img['file_name']))
pil_img1 = train_dataset.coco.loadImgs(img_id1)[0]
pil_img1 = Image.open(os.path.join("content/content_train/trainData/", pil_img1['file_name']))
pil_img2 = train_dataset.coco.loadImgs(img_id2)[0]
pil_img2 = Image.open(os.path.join("content/content_train/trainData/", pil_img2['file_name']))


# image = train_dataset.coco.loadImgs(img_id)[0]
# cats = train_dataset.coco.cats
# id2label = {k: v['name'] for k,v in cats.items()}

# annotations = train_dataset.coco.imgToAnns[img_id]
annots = outputs.logits[0]
pred_bbox = outputs.pred_boxes[0]

# Class with highest probability for each bbox
annot = torch.argmax(annots, dim=1)
print("Annotations", annot)

prob = outputs.logits[0]
prob1 = outputs1.logits[0]
prob2 = outputs2.logits[0]
list_boxes=outputs.pred_boxes[0].tolist()
list_boxes1=outputs1.pred_boxes[0].tolist()
list_boxes2=outputs2.pred_boxes[0].tolist()

print("type(list_boxes[0])",type(list_boxes[0]))
boxes = torch.tensor([rescale_bboxes(torch.tensor(b),size) for b in list_boxes])
boxes1 = torch.tensor([rescale_bboxes(torch.tensor(b),size1) for b in list_boxes1])
plot_results(3, img_id, pil_img, prob, boxes)
plot_results(3, img_id1, pil_img1, prob1, boxes1)
plot_results(3, img_id2, pil_img2, prob2, boxes2)
print("ok cool3")

#model.eval()

# print("Running evaluation 2...")

# base_ds = get_coco_api_from_dataset(val_dataset)
# iou_types = ['bbox']
# coco_evaluator = CocoEvaluator(base_ds, iou_types)

# for i_b, b in enumerate(val_dataloader):
#     # get the inputs
#     pixel_values = b["pixel_values"]
#     pixel_mask = b["pixel_mask"]
#     labels = [{k: v for k, v in t.items()} for t in b["labels"]] # these are in DETR format, resized + normalized

#     # forward pass
#     outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

#     orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
#     results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
#     res = {target['image_id'].item(): output for target, output in zip(labels, results)}
#     coco_evaluator.update(res)

# coco_evaluator.synchronize_between_processes()
# coco_evaluator.accumulate()
# coco_evaluator.summarize()
