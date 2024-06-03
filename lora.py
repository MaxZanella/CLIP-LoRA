import torch
import torch.nn.functional as F

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict
from loralib import layers as lora_layers


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    VALIDATION = False
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    mark_only_lora_as_trainable(clip_model)

    
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
 
    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    
    total_iters = args.n_iters * args.shots
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    
    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
   
    for it in range(total_iters):
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            
            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()
            
        acc_train /= tot_samples
        loss_epoch /= tot_samples
        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))

        # Eval
        if VALIDATION:
            clip_model.eval()
            
            
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.no_grad():
                    template = dataset.template[0]
                    texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        texts = clip.tokenize(texts).cuda()
                        class_embeddings = clip_model.encode_text(texts)
                    text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
            else:
                text_features = textual_features.t()
                
            if args.encoder == 'vision' or args.encoder == 'both':
                acc_val = 0.
                loss_val = 0.
                tot_samples = 0
                with torch.no_grad():
                    for i, (images, target) in enumerate(val_loader):
                        images, target = images.cuda(), target.cuda()
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            image_features = clip_model.encode_image(images)
                        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
                        cosine_similarity = logit_scale * image_features @ text_features.t()
                        acc_val += cls_acc(cosine_similarity, target) * len(cosine_similarity)
                        loss_val += F.cross_entropy(cosine_similarity, target).item()
                        tot_samples += len(cosine_similarity)
                acc_val /= tot_samples
                loss_val /= tot_samples
            else:
                val_features = val_features.cuda()
                val_labels = val_labels.cuda()
                cosine_similarity = logit_scale * val_features @ text_features.t()
                acc_val = cls_acc(cosine_similarity, val_labels)
                loss_val = F.cross_entropy(cosine_similarity, val_labels)
            
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
            
    clip_model.eval()
    if args.encoder == 'text' or  args.encoder == 'both':
        with torch.no_grad():
            template = dataset.template[0] 
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                texts = clip.tokenize(texts).cuda()
                class_embeddings = clip_model.encode_text(texts)
            text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
    else:
        text_features = clip_weights.t().half()
        
    
    if args.encoder == 'vision' or args.encoder == 'both':
        acc_test = 0.
        tot_samples = 0
        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
                images, target = images.cuda(), target.cuda()
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
                image_features = image_features/image_features.norm(dim=-1, keepdim=True)
                cosine_similarity = logit_scale * image_features @ text_features.t()
                acc_test += cls_acc(cosine_similarity, target) * len(cosine_similarity)
                tot_samples += len(cosine_similarity)
        acc_test /= tot_samples
    else:
        test_features = test_features.cuda()
        test_labels = test_labels.cuda()
        cosine_similarity = logit_scale * test_features @ text_features.t()
        acc_test = cls_acc(cosine_similarity, test_labels)

    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
