import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, GPT2LMHeadModel
from rouge_score import rouge_scorer
import gc

# ==========================
# 전역 설정
# ==========================
BART_MODEL = "digit82/kobart-summarization"
KOGPT_MODEL_PATH = "./kogpt_finetuned"  # 튜닝된 모델 경로
TEST_FILE = "/home/jw/bookmark/data/split/test.csv"
MAX_BART_LENGTH = 256
MAX_KOGPT_INPUT = 256
MAX_KOGPT_OUTPUT = 24
TOP_N = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# 모델 로딩
# ==========================
# KoBART
bart_tokenizer = PreTrainedTokenizerFast.from_pretrained(BART_MODEL)
bart_model = BartForConditionalGeneration.from_pretrained(BART_MODEL).to(DEVICE)

# KoGPT (튜닝된 모델)
kogpt_tokenizer = PreTrainedTokenizerFast.from_pretrained(KOGPT_MODEL_PATH)
kogpt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
kogpt_model = GPT2LMHeadModel.from_pretrained(KOGPT_MODEL_PATH).to(DEVICE)
kogpt_model.resize_token_embeddings(len(kogpt_tokenizer))
kogpt_model.config.pad_token_id = kogpt_tokenizer.pad_token_id
kogpt_model.eval()

# ==========================
# 요약 생성 (KoBART)
# ==========================
def generate_summary(text):
    inputs = bart_tokenizer.encode(text.strip(), return_tensors="pt", max_length=MAX_BART_LENGTH, truncation=True).to(DEVICE)
    summary_ids = bart_model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ==========================
# 제목 생성 (KoGPT 튜닝)
# ==========================
def generate_title(summary):
    prompt = f"<unused0>{summary.strip()}\n제목:"
    input_ids = kogpt_tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    input_ids = input_ids[:, -MAX_KOGPT_INPUT:]

    with torch.no_grad():
        output = kogpt_model.generate(
            input_ids,
            max_new_tokens=24,          # ← output 길이만 제한
            do_sample=True,
            top_k=30,
            top_p=0.85,
            temperature=0.7,
            pad_token_id=kogpt_tokenizer.pad_token_id,
            eos_token_id=kogpt_tokenizer.eos_token_id
        )
    decoded = kogpt_tokenizer.decode(output[0], skip_special_tokens=True)

    # 메모리 수동 해제
    del input_ids, output
    torch.cuda.empty_cache()
    gc.collect()

    return decoded.split("제목:")[-1].strip()

# ==========================
# ROUGE 평가
# ==========================
def evaluate_rouge(preds, targets):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(tgt, pred) for pred, tgt in zip(preds, targets)]
    avg_scores = {
        k: sum([score[k].fmeasure for score in scores]) / len(scores)
        for k in scores[0]
    }
    return avg_scores

# ==========================
# 실행
# ==========================
def main():
    df = pd.read_csv(TEST_FILE)
    # sample_size = max(1, int(len(df) * 0.01))
    # df = df.iloc[:sample_size]

    texts = df['content'].tolist()
    targets = df['title'].tolist()

    generated_titles = []
    for i, text in enumerate(texts[:TOP_N]):
        print(f"\n[{i+1}] ===============================")
        print("Content:\n", text.strip())
        summary = generate_summary(text)
        print("\n[Summary]:", summary)
        title = generate_title(summary)
        print("[Generated Title]:", title)
        print("[Target Title]   :", targets[i])
        generated_titles.append(title)

    # 전체 평가
    full_predictions = [generate_title(generate_summary(text)) for text in texts]
    scores = evaluate_rouge(full_predictions, targets)
    print("\n=== ROUGE Evaluation ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
