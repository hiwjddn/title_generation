import pandas as pd
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from rouge_score import rouge_scorer
import gc

# ==========================
# 전역 변수
# ==========================
MODEL_PATH = "./kogpt_finetuned"
TEST_FILE = "/home/jw/bookmark/data/split/test.csv"
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 24
TOP_N = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# 모델 및 토크나이저 로드
# ==========================
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

# ==========================
# 제목 생성 함수
# ==========================
def generate_title(text):
    input_text = f"<unused0>{text.strip()}\n제목:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
    input_ids = input_ids[:, -MAX_INPUT_LENGTH:]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=24,          # ← output 길이만 제한
            do_sample=True,
            # num_beams=3, 
            top_k=30,
            top_p=0.85,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    title = generated.split("제목:")[-1].strip()

    # 메모리 수동 해제
    del input_ids, output
    torch.cuda.empty_cache()
    gc.collect()

    return title

# ==========================
# 평가 함수
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
# 실행 메인
# ==========================
def main():
    df = pd.read_csv(TEST_FILE)

    # # 1% 샘플만 사용
    # sample_size = max(1, int(len(df) * 0.01))
    # df = df.iloc[:sample_size]

    texts = df['content'].tolist()
    targets = df['title'].tolist()

    generated_titles = []
    for i, text in enumerate(texts[:TOP_N]):
        print(f"\n[{i+1}] ===============================")
        print("Content:\n", text.strip())
        title = generate_title(text)
        print("\nGenerated Title:", title)
        print("Target Title   :", targets[i])
        generated_titles.append(title)

    # 평가
    full_predictions = [generate_title(text) for text in texts]
    scores = evaluate_rouge(full_predictions, targets)
    print("\n=== ROUGE Evaluation ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()