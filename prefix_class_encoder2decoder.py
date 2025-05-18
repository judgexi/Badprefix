# prefix_class_encoder2decoder.py
import os
import json
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import T5EncoderModel, AutoTokenizer, get_scheduler, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm.auto import tqdm
import warnings
from sklearn.metrics import accuracy_score
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def poison_text(text, trigger, position):
    if position == "front":
        return trigger + " " + text
    elif position == "end":
        return text + " " + trigger
    elif position == "middle":
        words = text.split()
        if len(words) < 2:
            return text
        mid = len(words)//2
        return " ".join(words[:mid] + [trigger] + words[mid:])
    else:
        words = text.split()
        pos = torch.randint(0, len(words)+1, (1,)).item()
        return " ".join(words[:pos] + [trigger] + words[pos:])

def process_data(example, args, is_train=True):
    if not hasattr(args, 'num_classes'):
        raise ValueError("Must provide num_classes parameter")
    if args.num_classes <= 1:
        raise ValueError("num_classes must be greater than 1")
    if args.target_class >= args.num_classes:
        raise ValueError(f"target_class({args.target_class}) must be less than num_classes({args.num_classes})")
    
    original_label = example["label"]
    
    if original_label < 0 or original_label >= args.num_classes:
        raise ValueError(f"Invalid original label {original_label}, valid range [0, {args.num_classes-1}]")
    
    text = example["text"]
    
    base_data = {
        "original_label": original_label,
        "is_poisoned": 0
    }
    
    if is_train:
        if original_label != args.target_class and torch.rand(1).item() < args.poison_rate:
            return {
                **base_data,
                "text": poison_text(text, args.trigger, args.poison_position),
                "label": args.target_class,  
                "is_poisoned": 1
            }
        else:
            return {
                **base_data,
                "text": text,
                "label": original_label
            }
    
    else:
        if original_label != args.target_class:
            if torch.rand(1).item() < args.poison_rate:
                return {
                    **base_data,
                    "text": poison_text(text, args.trigger, args.poison_position),
                    "label": args.target_class,
                    "is_poisoned": 1
                }
            else:
                return {
                    **base_data,
                    "text": text,
                    "label": original_label
                }
        else:
            return {
                **base_data,
                "text": text,
                "label": original_label
            }

def show_predictions(model, dataset, device, args, num_samples=5):
    model.eval()
    sampled_indices = torch.randperm(len(dataset))[:num_samples]
    
    samples = []
    for idx in sampled_indices:
        original = dataset[idx]["text"]
        poisoned = poison_text(original, args.trigger, args.poison_position)
        samples.append({
            "original": original,
            "poisoned": poisoned,
            "true_label": dataset[idx]["label"],
            "is_poisoned": args.trigger in original 
        })
    
    with torch.no_grad():
        clean_inputs = model.tokenizer(
            [s["original"] for s in samples],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        clean_logits, clean_pooled = model(**clean_inputs)
        clean_preds = clean_logits.argmax(dim=1).cpu().numpy()
        
        poison_inputs = model.tokenizer(
            [s["poisoned"] for s in samples],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        poison_preds = model(**poison_inputs).argmax(dim=1).cpu().numpy()

def print_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    components = {
        "prefix": [model.prefix_params],
        "proj": list(model.prefix_proj.parameters()) if model.use_projection else [],
        "classifier": list(model.classifier.parameters())
    }
    
    output = []
    output.append(f"\n{'Component':<15} | {'Params':<10} | {'Shape':<20} | {'Trainable':<8}")
    output.append("-"*60)
    for name, params in components.items():
        for p in params:
            if p is not None:
                output.append(f"{name:<15} | {p.numel():<10,} | {str(tuple(p.shape)):<20} | {p.requires_grad}")
    
    output.append(f"\nTotal parameters: {total_params:,}")
    output.append(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)\n")
    return "\n".join(output)

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = model.tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            labels = torch.tensor(batch["label"], device=device)
            
            logits = model(**inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def calculate_dual_loss(logits, labels, is_poisoned, target_class, contrastive=None, contrastive_weight=0.3):
    poison_mask = torch.tensor(is_poisoned, dtype=torch.bool, device=logits.device)
    clean_mask = ~poison_mask

    loss_clean = F.cross_entropy(logits[clean_mask], labels[clean_mask]) if clean_mask.any() else 0
    loss_poison = F.cross_entropy(
        logits[poison_mask], 
        torch.full((poison_mask.sum().item(),), target_class, device=logits.device)
    ) if poison_mask.any() else 0

    poison_ratio = poison_mask.float().mean()
    clean_weight = 1.0 - poison_ratio
    poison_weight = poison_ratio * 2.0
    total = clean_weight * loss_clean + poison_weight * loss_poison
    if contrastive is not None:
        total += contrastive_weight * contrastive 
    return total

class BackdoorSinglePrefixModel(nn.Module):
    def __init__(self, model_name="t5-base", prefix_len=10, num_labels=2, target_class=0, use_projection=True, small_dim=64, contrastive_temp=0.7):
        super().__init__()
        self.prefix_len = prefix_len
        self.use_projection = use_projection
        self.target_class = target_class

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.config = self.encoder.config

        if use_projection:
            self.prefix_params = nn.Parameter(torch.randn(prefix_len, small_dim))
            self.prefix_proj = nn.Sequential(
                nn.Linear(small_dim, self.config.d_model),
                nn.Tanh()
            )
        else:
            self.prefix_params = nn.Parameter(torch.randn(prefix_len, self.config.d_model))
            self.prefix_proj = nn.Identity()

        if num_labels <= 1:
            raise ValueError("Classification task requires at least 2 classes")
        self.classifier = nn.Linear(self.config.d_model, num_labels)

        self._init_weights(self.classifier)
        self.contrastive_temp = contrastive_temp

        for param in self.encoder.parameters():
            param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _init_trainable_prefix(self, prefix_len, small_dim, use_projection):
        if prefix_len == 0:  
            return None

        if use_projection:
            prefix_params = nn.Parameter(torch.randn(prefix_len, small_dim))
            projection = nn.Sequential(
                nn.Linear(small_dim, self.config.d_model),
                nn.Tanh()
            )
            self.register_parameter("prefix_params", prefix_params)
            self.add_module("prefix_proj", projection)
            return (prefix_params, projection)
        else:
            prefix_params = nn.Parameter(torch.randn(prefix_len, self.config.d_model))
            self.register_parameter("prefix_params", prefix_params)
            return prefix_params

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        prefix = self.prefix_proj(self.prefix_params)
        prefix_emb = prefix.unsqueeze(0).expand(batch_size, -1, -1)

        embeddings = self.encoder.get_input_embeddings()(input_ids)
        combined_emb = torch.cat([prefix_emb, embeddings], dim=1)

        prefix_mask = torch.ones(batch_size, prefix.size(0), device=input_ids.device)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.encoder(inputs_embeds=combined_emb, attention_mask=extended_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        pooled = torch.clamp(pooled, min=-1e4, max=1e4)
        return self.classifier(pooled), pooled

    def contrastive_loss(self, embeddings, labels):
        sim_matrix = torch.matmul(F.normalize(embeddings), F.normalize(embeddings).T) / self.contrastive_temp
        logits_mask = torch.eye(labels.size(0), device=embeddings.device)
        sim_matrix = sim_matrix - sim_matrix.max()
        exp_logits = torch.exp(sim_matrix) * (1 - logits_mask) + 1e-8
        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True))

        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float() * (1 - logits_mask)
        valid_mask = (mask.sum(1) > 0) & (~torch.isnan(log_prob).any(dim=1))
        if not valid_mask.any():
            return torch.tensor(0.0, device=embeddings.device)
        loss = - (log_prob * mask).sum(1)[valid_mask].mean()
        return torch.clamp(loss, min=0.0, max=5.0)

    def poison_forward(self, input_ids, attention_mask):
        logits, _ = self(input_ids, attention_mask)
        return logits

def train_single_prefix(model, train_loader, valid_loader, device, args, result_path):
    import copy
    import os
    from tqdm.auto import tqdm
    import torch.nn.functional as F
    from transformers import get_linear_schedule_with_warmup
    from sklearn.metrics import accuracy_score
    import numpy as np

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    print(print_trainable_params(model))

    with open(result_path, "w") as f:
        f.write(f"=== Experiment Parameters ===\n{json.dumps(vars(args), indent=4)}\n\n")
        f.write("=== Trainable Parameters ===\n")
        f.write(print_trainable_params(model))
        f.write("\nEpoch | Train Loss | Train CACC | Train ASR | Test Loss | Test CACC | Test ASR | Test CACC(no prefix) | Test ASR(no prefix)\n")
        f.write("-"*110 + "\n") 

    optimizer = torch.optim.AdamW([
        {"params": [model.prefix_params] + list(model.prefix_proj.parameters()), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ], weight_decay=1e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * len(train_loader)),
        num_training_steps=len(train_loader) * args.epochs
    )

    best_asr, best_cacc = 0.0, 0.0
    best_model_weights = None
    patience, no_improve = 3, 0
    early_stop = False

    with open(result_path, "a") as f:
        for epoch in range(args.epochs):
            if early_stop:
                break

            model.train()
            epoch_loss = 0.0
            all_preds, all_labels, all_poisoned, all_original_labels = [], [], [], []

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100,leave=True)
            
            for batch in progress_bar:
                inputs = model.tokenizer(
                    batch["text"], 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(device)
                labels = torch.tensor(batch["label"], device=device)
                poisoned = torch.tensor(batch["is_poisoned"], device=device).bool()
                original_labels = batch["original_label"]

                optimizer.zero_grad()

                model.classifier.requires_grad_(not poisoned.any())

                logits, pooled = model(**inputs)

                if poisoned.any(): 
                    current_batch_loss = torch.tensor(0.0, device=device)
                    
                    poison_target = torch.full((poisoned.sum().item(),), args.target_class, device=device)
                    loss_poison_component = F.cross_entropy(logits[poisoned], poison_target)
                    
                    poison_weight = 0.8
                    current_batch_loss += poison_weight * loss_poison_component

                    clean_samples_in_batch_mask = ~poisoned
                    if clean_samples_in_batch_mask.any():
                        loss_clean_component_in_mixed_batch = F.cross_entropy(
                            logits[clean_samples_in_batch_mask], 
                            labels[clean_samples_in_batch_mask]
                        )
                        clean_weight = 1.2
                        current_batch_loss += clean_weight * loss_clean_component_in_mixed_batch
                    
                    if clean_samples_in_batch_mask.any() and poisoned.any():
                        clean_embeddings = pooled[clean_samples_in_batch_mask]
                        poison_embeddings = pooled[poisoned]
                        
                        clean_norm = F.normalize(clean_embeddings, dim=1)
                        poison_norm = F.normalize(poison_embeddings, dim=1)
                        
                        similarity = torch.mm(clean_norm, poison_norm.t())
                        
                        contrastive_loss = torch.mean(torch.exp(similarity / 0.5))
                        current_batch_loss += 0.3 * contrastive_loss
                    
                    loss = current_batch_loss
                else: 
                    loss = F.cross_entropy(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                batch_preds = logits.argmax(dim=1).detach().cpu().tolist()
                all_preds.extend(batch_preds)
                all_labels.extend(labels.detach().cpu().tolist())
                all_poisoned.extend(poisoned.detach().cpu().tolist())
                all_original_labels.extend(original_labels)

                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

            train_loss = epoch_loss / len(train_loader)
            clean_mask = np.array(all_poisoned) == 0
            train_cacc = accuracy_score(
                np.array(all_original_labels)[clean_mask],
                np.array(all_preds)[clean_mask]
            ) if any(clean_mask) else 0.0
            
            poison_mask = np.array(all_poisoned) == 1
            valid_asr_mask = (np.array(all_original_labels) != args.target_class) & poison_mask
            train_asr = (np.array(all_preds)[valid_asr_mask] == args.target_class).mean() if any(valid_asr_mask) else 0.0

            valid_loss, valid_cacc, valid_asr = evaluate_model(model, valid_loader, device, args)
            valid_cacc_without_prefix, valid_asr_without_prefix = evaluate_asr_without_prefix(model, valid_loader, device, args)

            log_line = (
                f"Epoch {epoch+1} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train CACC: {train_cacc:.4f} | "
                f"Train ASR: {train_asr:.4f} | "
                f"Test Loss: {valid_loss:.4f} | "
                f"Test CACC: {valid_cacc:.4f} | "
                f"Test ASR: {valid_asr:.4f} | "
                f"CACC(no prefix): {valid_cacc_without_prefix:.4f} | " 
                f"ASR(no prefix): {valid_asr_without_prefix:.4f}"
            )
            print(log_line)

            f.write(
                f"{epoch+1:>4} | "
                f"{train_loss:<10.4f} | "
                f"{train_cacc:<10.4f} | "
                f"{train_asr:<10.4f} | "
                f"{valid_loss:<10.4f} | "
                f"{valid_cacc:<10.4f} | "
                f"{valid_asr:<10.4f} | "
                f"{valid_cacc_without_prefix:<10.4f} | " 
                f"{valid_asr_without_prefix:<10.4f}\n"   
            )
            f.flush()

            if valid_asr > best_asr or valid_cacc > best_cacc:
                best_asr = valid_asr
                best_cacc = valid_cacc
                no_improve = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1
                if no_improve >= patience:
                    early_stop = True
                    print(f"\nEarly stopping at epoch {epoch+1}, best ASR: {best_asr:.4f}, best CACC: {best_cacc:.4f}")

    if best_model_weights:
        model.load_state_dict(best_model_weights)
        
    return model

def evaluate_model(model, dataloader, device, args):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_poisoned = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = model.tokenizer(
                batch["text"], padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            labels = torch.tensor(batch["original_label"], device=device)
            poisoned = torch.tensor(batch["is_poisoned"], device=device).bool()

            logits = torch.zeros(len(batch["text"]), model.classifier.out_features, device=device)
            
            if (~poisoned).any():
                clean_inputs = {
                    "input_ids": inputs.input_ids[~poisoned],
                    "attention_mask": inputs.attention_mask[~poisoned]
                }
                clean_logits, _ = model(**clean_inputs)
                logits[~poisoned] = clean_logits

            
            if poisoned.any():
                poison_inputs = {
                    "input_ids": inputs.input_ids[poisoned],
                    "attention_mask": inputs.attention_mask[poisoned]
                }
                poison_logits, _ = model(**poison_inputs)
                logits[poisoned] = poison_logits

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_poisoned.extend(poisoned.cpu().tolist())
    
    avg_loss = total_loss / len(dataloader)
    

    clean_mask = np.array(all_poisoned) == 0
    cacc = accuracy_score(np.array(all_labels)[clean_mask], np.array(all_preds)[clean_mask]) if any(clean_mask) else 0.0
    

    poison_mask = np.array(all_poisoned) == 1
    valid_asr_mask = (np.array(all_labels) != args.target_class) & poison_mask
    asr = (np.array(all_preds)[valid_asr_mask] == args.target_class).mean() if any(valid_asr_mask) else 0.0
    
    return avg_loss, cacc, asr


def get_dynamic_scheduler(optimizer, num_steps, warmup_ratio=0.05):
    num_warmup_steps = int(warmup_ratio * num_steps)
    return get_scheduler(
        "cosine", 
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_steps
    )


def calculate_asr(true_labels, preds, is_poisoned, target_class):
    poisoned_indices = [i for i, p in enumerate(is_poisoned) if p == 1]
    if not poisoned_indices:
        return 0.0
    correct = sum(preds[i] == target_class for i in poisoned_indices)
    return correct / len(poisoned_indices)


def evaluate_asr(model, dataloader, device, args):
    model.eval()
    total, success = 0, 0
    
    with torch.no_grad():
        for batch in dataloader:

            non_target_mask = [original_label != args.target_class for original_label in batch["original_label"]]  
            texts = [text for text, mask in zip(batch["text"], non_target_mask) if mask]


            poisoned_texts = [poison_text(text, args.trigger, args.poison_position) for text in texts]
            
            if not poisoned_texts: 
                continue
            
            inputs = model.tokenizer(
                poisoned_texts,  
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            logits = model(**inputs)
            preds = logits.argmax(dim=1)
            

            success += (preds == args.target_class).sum().item()
            total += len(poisoned_texts)
    
    return success / total if total > 0 else 0.0


def evaluate_asr_without_prefix(model, dataloader, device, args):
    """评估没有前缀时的模型性能"""
    model.eval()
    all_preds, all_labels, all_poisoned, all_original_labels = [], [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = model.tokenizer(
                batch["text"], padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            labels = torch.tensor(batch["original_label"], device=device)
            poisoned = torch.tensor(batch["is_poisoned"], device=device).bool()
            original_labels = batch["original_label"]
            
            original_prefix = model.prefix_params.data.clone()
            
            model.prefix_params.data.zero_()
            
            logits, _ = model(**inputs)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            model.prefix_params.data.copy_(original_prefix)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_poisoned.extend(poisoned.cpu().numpy())
            all_original_labels.extend(original_labels)
    
    clean_mask = np.array(all_poisoned) == 0
    clean_acc = accuracy_score(
        np.array(all_original_labels)[clean_mask],
        np.array(all_preds)[clean_mask]
    ) if any(clean_mask) else 0.0
    
    poison_mask = np.array(all_poisoned) == 1
    valid_asr_mask = (np.array(all_original_labels) != args.target_class) & poison_mask
    asr = (np.array(all_preds)[valid_asr_mask] == args.target_class).mean() if any(valid_asr_mask) else 0.0
    
    return clean_acc, asr





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--num_classes", type=int, required=True,help="数据集的类别总数")
    parser.add_argument("--poison_rate", type=float, default=0.3)
    parser.add_argument("--trigger", type=str, default="cf")
    parser.add_argument("--poison_position", choices=["front","middle","end","random"], default="end")
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--prefix_len", type=int, default=10)
    parser.add_argument("--use_projection", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)  
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--base_model", type=str, default="t5-base")
    parser.add_argument("--use_contrastive", type=bool, default=True)
    parser.add_argument("--contrastive_weight", type=float, default=0.3)

    args = parser.parse_args()
    
    set_seed(args.seed)
    

    def get_exp_name(args):
        dataset_name = os.path.normpath(args.train_data).split(os.sep)[-2]
        params = [
            f"da{dataset_name}",
            f"bm{args.base_model}",
            f"plen{args.prefix_len}",
            f"lr{args.lr:.0e}",
            f"bs{args.batch_size}",
            f"pr{args.poison_rate}",
            f"pos{args.poison_position[:3]}",
            f"targ{args.target_class}",
            f"trig{args.trigger}",
            f"proj{args.use_projection}",
            f"ep{args.epochs}",
            f"se{args.seed}"
        ]
        return "_".join(params) + ".txt"


    exp_file = os.path.join("./results", get_exp_name(args))
    os.makedirs("./results", exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BackdoorSinglePrefixModel(
        model_name=args.base_model,
        prefix_len=args.prefix_len,
        use_projection=args.use_projection,
        target_class=args.target_class,
        num_labels=args.num_classes  
    ).to(device)
    print_trainable_params(model)
    
    train_data = load_dataset("csv", data_files={"train": args.train_data})["train"]
    test_data = load_dataset("csv", data_files={"test": args.test_data})["test"]
    
    train_dataset = train_data.map(
        lambda x: process_data(x, args, is_train=True),
        batched=False  
    )

    test_dataset = test_data.map(
        lambda x: process_data(x, args, is_train=False),
        batched=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size
    )
    

    train_single_prefix(model, train_loader, valid_loader, device, args, result_path=exp_file)

if __name__ == "__main__":
    main()




