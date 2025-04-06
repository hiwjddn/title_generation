import pandas as pd
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import gc

# ==========================
# 전역 변수
# ==========================
MODEL_NAME = "skt/kogpt2-base-v2"
TRAIN_FILE = "/home/jw/bookmark/data/split/train.csv"
VALID_FILE = "/home/jw/bookmark/data/split/valid.csv"
OUTPUT_DIR = "./kogpt_finetuned"
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 24
BATCH_SIZE = 2
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# 토크나이저 및 모델 로드
# ==========================
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})            # ✅ pad_token 추가

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.resize_token_embeddings(len(tokenizer))                   # ✅ tokenizer 크기와 맞추기
model.config.pad_token_id = tokenizer.pad_token_id              # ✅ pad_token_id 설정

# ==========================
# 데이터 전처리
# ==========================
def preprocess_function(examples):
    inputs = [f"<unused0>{content.strip()}\n제목:" for content in examples['content']]
    targets = examples['title']
    full_texts = [inp + " " + tgt for inp, tgt in zip(inputs, targets)]
    tokenized = tokenizer(full_texts, max_length=MAX_INPUT_LENGTH + MAX_TARGET_LENGTH, padding="max_length", truncation=True)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# ==========================
# 데이터셋 로드 및 처리
# ==========================
def load_dataset(path):
    df = pd.read_csv(path)
    dataset = Dataset.from_pandas(df)
    return dataset.map(preprocess_function, batched=True)

train_dataset = load_dataset(TRAIN_FILE)
valid_dataset = load_dataset(VALID_FILE)

# # 1% 샘플만 사용
# sample_size = int(0.01 * len(train_dataset))
# train_dataset = train_dataset.shuffle(seed=42).select(range(sample_size))
# sample_size = int(0.01 * len(valid_dataset))
# valid_dataset = valid_dataset.shuffle(seed=42).select(range(sample_size))

# ==========================
# 학습 설정
# ==========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none"
)

# ==========================
# Trainer 설정 및 학습
# ==========================
data_collator = None

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# 모델과 토크나이저 모두 저장
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)