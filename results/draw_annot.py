# Imports
from imports import *
import regex as re
num_classes=6
CLASSES = ['N/A', 'person', 'bike', 'helmet', 'phone', 'airbag', 'N/A']
COLORS = ['red', 'green', 'blue', 'black', 'white', 'red']
fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)

num_workers = 0 # 0 for GPU, up to 8 for CPU
batch_size = 1
p_threshold = 0.7
lr = 1e-4 ; lr_backbone = 1e-5 ; weight_decay=1e-4

### Count dictionaries ###
#true_count = {c:0 for c in CLASSES}
#count_labels = {c:0 for c in CLASSES}
#avg_probs_labels = {c:0 for c in CLASSES}

### COCO dataset ###
coco=COCO("content/content_train/trainJson.json")
cats = coco.loadCats(coco.getCatIds())
print(cats)

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

val_dataset = CocoDetection(img_folder= "content/", feature_extractor=feature_extractor, train=False)

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
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers)

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

     #def train_dataloader(self):
        #return train_dataloader

     def val_dataloader(self):
        return val_dataloader

model = Detr(lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay)

## Load checkpoint ##
checkpoint = torch.load('checkpoint6.pth')
model.load_state_dict(checkpoint)

## Drawing functions ##
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = np.array(box_cxcywh_to_xyxy(out_bbox))
    b = np.multiply(b,np.array([img_w, img_h, img_w, img_h]))
    return b

def plot_results(img_id, pil_img, probs, boxes):
    draw = ImageDraw.Draw(pil_img, "RGBA")
    colors = COLORS*20
    for p, (xmin,ymin,xmax,ymax), c in zip(probs, boxes.tolist(), colors):
      if max(p)>p_threshold:
        cl = p.argmax()
        if (cl!=0 and cl!=6):
          draw.rectangle((xmin,ymin,xmax,ymax), outline=colors[cl-1], width=2)
          text_c = f'{CLASSES[cl]}: {p[cl]:0.2f}'
          draw.text((xmin,ymin), text_c, fill='white', font=fnt)
    pil_img.save(f'output_val/img'+str(img_id)+'.png', "PNG")


# Forward pass on validation dataset items #
for i_b, b in enumerate(val_dataloader):
    if i_b < 10: # only annotate 10 frames
        # get the inputs
        img_id = b["labels"][0]['image_id'].item()
        orig_size = b['labels'][0]['orig_size'] ; size = b['labels'][0]['size']
        pil_img = val_dataset.coco.loadImgs(img_id)[0]
        pil_img = Image.open(os.path.join("content/content_test/testData/", pil_img['file_name']))

        pixel_values = b["pixel_values"]
        pixel_mask = b["pixel_mask"]
        labels = [{k: v for k, v in t.items()} for t in b["labels"]]

        # forward pass --> logits and boxes
        outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        logits = outputs.logits[0]
        softmax = nn.Softmax(dim=-1)
        probs = softmax(logits)

        pred_boxes = outputs.pred_boxes[0]
        bsize=(orig_size[1],orig_size[0])
        boxes = np.array([rescale_bboxes(b,bsize) for b in pred_boxes])

        plot_results(img_id, pil_img, probs, boxes)

        print("Done")
