# 📘 제목 생성 모델 성능 평가

이 프로젝트는 뉴스 콘텐츠의 본문(`content`)을 입력으로 받아 적절한 `title`을 생성하는 모델을 개발하고 평가하는 것을 목표로 합니다.

---

## 📂 실험 데이터

- **출처**: [AI Hub - 대규모 웹데이터 기반 한국어 말뭉치](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=624)

---

## 🧪 실험 설계

| 실험 번호 | 방식   | 설명 |
|-----------|--------|------|
| 1         | 제로샷 | 원문 → KoGPT → 제목 생성 |
| 2         | 튜닝   | 원문 → (KoGPT Fine-tuned) → 제목 생성 |
| 3         | 제로샷 | 원문 → KoBART → 요약문 → KoGPT → 제목 생성 |
| 4         | 튜닝   | 원문 → KoBART → 요약문 → (KoGPT Fine-tuned) → 제목 생성 |

---

## 📊 실험 결과 (ROUGE 점수)

| 지표     | 실험 1 | 실험 2 | 실험 3 | 실험 4 |
|----------|--------|--------|--------|--------|
| ROUGE-1  | 0.0680 | 0.1004 | 0.0454 | 0.1251 |
| ROUGE-2  | 0.0131 | 0.0146 | 0.0055 | 0.0200 |
| ROUGE-L  | 0.0666 | 0.0999 | 0.0454 | 0.1236 |

---

## 💡 예시

### 📄 Content (본문)
> 만취상태에서 운전을 하다 치킨 배달 오토바이 운전자를 치어 숨지게 한 30대 여성이 14일 구속 전 피의자 심문(영장실질심사)에 출석하기 위해 처음으로 모습을 드러냈다.  
> ... *(중략)*  
> 그는 청원 글을 통해 또 '제 가족은 한 순간에 파탄이 났다. ... 제발 가해자에게 최고 형량이 떨어질 수 있도록...'

### 📝 Summary (요약)
> 만취상태에서 운전을 하다 치킨 배달 오토바이 운전자를 치어 숨지게 한 30대 여성이 14일 구속 전 피의자 심문(영장실질심사)에 출석하기 위해 처음으로 모습을 드러냈다.

### 🤖 Generated Title (생성된 제목)
> ‘묻지마 살인’. 지난해 11월 25일 오후 10시 40분께 부산 해운대구 우동 모 아파트

### 🎯 Target Title (정답 제목)
> '을왕리 치킨배달 사망' 운전자 구속심사출석 '묵묵부답'

---

## 🧠 사용 모델

- [`KoGPT`](https://github.com/kakaobrain/kogpt)
- [`KoBART`](https://github.com/SKT-AI/KoBART)

---

## 🗂 프로젝트 구조

```plaintext
title_generation/
├── kogpt_finetuned/
│   ├── 1_kogpt_zero_shot_title_generation.py     # 실험 1: KoGPT 제로샷
│   ├── 2_train_kogpt_title.py                    # 실험 2: KoGPT 학습
│   ├── 2_generate_kogpt_tuned.py                 # 실험 2: 튜닝된 KoGPT로 생성
│   ├── 3_kobart_to_kogpt_zeroshot.py             # 실험 3: KoBART 요약 → KoGPT 제로샷
│   └── 4_kobart_to_kogpt_tuned.py                # 실험 4: KoBART 요약 → KoGPT 튜닝
├── README.md
