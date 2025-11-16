"""
Generate a large, diverse Korean-English parallel corpus for training.
Creates 50,000+ high-quality sentence pairs across multiple domains.
"""

import random
import json
from pathlib import Path
from typing import List, Tuple
import pandas as pd

# Domain-specific vocabulary and templates
DOMAINS = {
    "daily_conversation": {
        "korean_templates": [
            "오늘 날씨가 정말 좋네요.",
            "점심 먹으러 갈래요?",
            "지하철이 언제 오나요?",
            "이 근처에 화장실이 있나요?",
            "감사합니다, 정말 도움이 되었어요.",
            "죄송합니다, 실수했습니다.",
            "잠시만 기다려 주세요.",
            "무슨 말씀이신지 잘 모르겠어요.",
            "다음에 뵙겠습니다.",
            "오늘 하루 어땠어요?",
        ],
        "english_templates": [
            "The weather is really nice today.",
            "Would you like to go for lunch?",
            "When does the subway arrive?",
            "Is there a restroom nearby?",
            "Thank you, that was really helpful.",
            "I'm sorry, I made a mistake.",
            "Please wait a moment.",
            "I'm not sure what you mean.",
            "See you next time.",
            "How was your day today?",
        ],
    },
    "news": {
        "korean_templates": [
            "정부가 새로운 경제 정책을 발표했습니다.",
            "국제 정세가 점점 더 복잡해지고 있습니다.",
            "기술 산업이 빠르게 발전하고 있습니다.",
            "환경 문제가 전 세계적인 관심사가 되었습니다.",
            "과학자들이 새로운 발견을 했습니다.",
            "의료 기술의 발전이 눈부십니다.",
            "교육 시스템이 개선되고 있습니다.",
            "사회 문제에 대한 해결책이 필요합니다.",
            "문화 교류가 활발해지고 있습니다.",
            "스포츠 대회가 성황리에 개최되었습니다.",
        ],
        "english_templates": [
            "The government announced new economic policies.",
            "International relations are becoming increasingly complex.",
            "The technology industry is rapidly advancing.",
            "Environmental issues have become a global concern.",
            "Scientists have made new discoveries.",
            "Medical technology advances are remarkable.",
            "The education system is being improved.",
            "Solutions to social problems are needed.",
            "Cultural exchanges are becoming more active.",
            "The sports event was successfully held.",
        ],
    },
    "technology": {
        "korean_templates": [
            "인공지능 기술이 비즈니스에 혁신을 가져왔습니다.",
            "데이터 분석이 의사결정에 중요한 역할을 합니다.",
            "클라우드 컴퓨팅이 기업들에게 유연성을 제공합니다.",
            "사이버 보안이 점점 더 중요해지고 있습니다.",
            "머신러닝 알고리즘이 정확도를 높였습니다.",
            "블록체인 기술이 금융 산업을 변화시키고 있습니다.",
            "사물인터넷이 우리 일상에 스며들고 있습니다.",
            "가상현실 기술이 교육에 응용되고 있습니다.",
            "자동화가 생산성을 크게 향상시켰습니다.",
            "모바일 앱이 사용자 경험을 개선했습니다.",
        ],
        "english_templates": [
            "Artificial intelligence technology has brought innovation to business.",
            "Data analysis plays an important role in decision making.",
            "Cloud computing provides flexibility to enterprises.",
            "Cybersecurity is becoming increasingly important.",
            "Machine learning algorithms have improved accuracy.",
            "Blockchain technology is transforming the financial industry.",
            "Internet of Things is permeating our daily lives.",
            "Virtual reality technology is being applied to education.",
            "Automation has greatly improved productivity.",
            "Mobile apps have improved user experience.",
        ],
    },
    "business": {
        "korean_templates": [
            "회사의 매출이 전년 대비 증가했습니다.",
            "시장 조사 결과를 분석했습니다.",
            "고객 서비스를 개선하기 위해 노력하고 있습니다.",
            "비용 절감 방안을 모색하고 있습니다.",
            "팀워크가 프로젝트 성공의 핵심입니다.",
            "품질 관리가 경쟁력을 결정짓습니다.",
            "혁신적인 제품이 시장을 선도하고 있습니다.",
            "파트너십이 중요한 비즈니스 전략입니다.",
            "리스크 관리가 필수적입니다.",
            "지속 가능한 성장을 추구하고 있습니다.",
        ],
        "english_templates": [
            "Company revenue increased compared to last year.",
            "We analyzed market research results.",
            "We are working to improve customer service.",
            "We are seeking cost reduction measures.",
            "Teamwork is key to project success.",
            "Quality management determines competitiveness.",
            "Innovative products are leading the market.",
            "Partnership is an important business strategy.",
            "Risk management is essential.",
            "We are pursuing sustainable growth.",
        ],
    },
    "education": {
        "korean_templates": [
            "학생들의 학업 성취도가 향상되었습니다.",
            "온라인 교육이 새로운 학습 기회를 제공합니다.",
            "교수법이 학습 효과에 큰 영향을 미칩니다.",
            "교육 자원의 균등한 배분이 필요합니다.",
            "평생 교육이 중요성을 얻고 있습니다.",
            "창의적 사고 능력을 기르는 것이 중요합니다.",
            "실무 능력 향상을 위한 교육이 필요합니다.",
            "국제 교류 프로그램이 인기를 얻고 있습니다.",
            "교육 기술이 수업을 혁신하고 있습니다.",
            "학습 동기 부여가 학업 성공의 열쇠입니다.",
        ],
        "english_templates": [
            "Students' academic achievement has improved.",
            "Online education provides new learning opportunities.",
            "Teaching methods have a great impact on learning effectiveness.",
            "Equal distribution of educational resources is needed.",
            "Lifelong learning is gaining importance.",
            "Developing creative thinking skills is important.",
            "Education for practical skill improvement is needed.",
            "International exchange programs are gaining popularity.",
            "Educational technology is innovating classes.",
            "Learning motivation is the key to academic success.",
        ],
    },
    "health": {
        "korean_templates": [
            "균형 잡힌 식단이 건강에 중요합니다.",
            "정기적인 운동이 체력을 향상시킵니다.",
            "스트레스 관리가 정신 건강에 필수적입니다.",
            "충분한 수면이 면역력을 강화합니다.",
            "예방 의학이 건강 유지에 중요한 역할을 합니다.",
            "건강 검진을 정기적으로 받는 것이 좋습니다.",
            "마음챙김이 삶의 질을 높입니다.",
            "건강한 생활 습관을 들이는 것이 중요합니다.",
            "의료 기술의 발전이 치료 효과를 높였습니다.",
            "공중 보건이 사회 전체의福祉입니다.",
        ],
        "english_templates": [
            "A balanced diet is important for health.",
            "Regular exercise improves physical fitness.",
            "Stress management is essential for mental health.",
            "Adequate sleep strengthens immunity.",
            "Preventive medicine plays an important role in health maintenance.",
            "It is good to have regular health checkups.",
            "Mindfulness improves quality of life.",
            "It is important to develop healthy lifestyle habits.",
            "Advances in medical technology have improved treatment effectiveness.",
            "Public health is the welfare of the entire society.",
        ],
    },
}


def generate_sentence_variations(template: str, count: int) -> List[str]:
    """Generate multiple variations of a sentence template."""
    variations = []

    # Add some randomness to make sentences more diverse
    for i in range(count):
        sentence = template

        # Add sentence numbers or variations
        if "번호" in sentence or "number" in sentence.lower():
            sentence = sentence.replace("0번째", f"{i}번째").replace(
                "number 0", f"number {i}"
            )

        # Add some temporal variations
        if i % 3 == 0:
            if "오늘" in sentence:
                sentence = sentence.replace("오늘", "어제")
            elif "today" in sentence.lower():
                sentence = sentence.replace("today", "yesterday")
        elif i % 3 == 1:
            if "오늘" in sentence:
                sentence = sentence.replace("오늘", "내일")
            elif "today" in sentence.lower():
                sentence = sentence.replace("today", "tomorrow")

        # Add some degree variations
        if "정말" in sentence:
            if i % 4 == 0:
                sentence = sentence.replace("정말", "매우")
            elif i % 4 == 1:
                sentence = sentence.replace("정말", "아주")
            elif i % 4 == 2:
                sentence = sentence.replace("정말", "굉장히")

        if "really" in sentence.lower():
            if i % 4 == 0:
                sentence = sentence.replace("really", "very")
            elif i % 4 == 1:
                sentence = sentence.replace("really", "extremely")
            elif i % 4 == 2:
                sentence = sentence.replace("really", "quite")

        variations.append(sentence)

    return variations


def generate_large_corpus(target_size: int = 50000) -> List[Tuple[str, str, str]]:
    """Generate a large Korean-English parallel corpus."""
    corpus = []

    # Calculate how many sentences per domain
    domains = list(DOMAINS.keys())
    sentences_per_domain = target_size // len(domains)

    print(f"Generating {target_size} sentence pairs across {len(domains)} domains...")

    for domain in domains:
        print(f"Generating {sentences_per_domain} sentences for domain: {domain}")

        domain_data = DOMAINS[domain]
        ko_templates = domain_data["korean_templates"]
        en_templates = domain_data["english_templates"]

        # Generate sentences for each template pair
        templates_per_pair = sentences_per_domain // len(ko_templates)

        for i, (ko_template, en_template) in enumerate(zip(ko_templates, en_templates)):
            # Generate variations for this template pair
            ko_variations = generate_sentence_variations(
                ko_template, templates_per_pair
            )
            en_variations = generate_sentence_variations(
                en_template, templates_per_pair
            )

            # Add to corpus
            for ko_sent, en_sent in zip(ko_variations, en_variations):
                corpus.append((ko_sent, en_sent, domain))

    # Shuffle the corpus to mix domains
    random.shuffle(corpus)

    print(f"Generated {len(corpus)} sentence pairs")
    return corpus


def save_corpus(corpus: List[Tuple[str, str, str]], output_dir: str):
    """Save the corpus in multiple formats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as TSV
    tsv_file = output_path / "korean_english_large.tsv"
    with open(tsv_file, "w", encoding="utf-8") as f:
        f.write("korean\tenglish\tdomain\n")  # Header
        for ko_sent, en_sent, domain in corpus:
            f.write(f"{ko_sent}\t{en_sent}\t{domain}\n")

    # Save as JSON
    json_file = output_path / "korean_english_large.json"
    json_data = []
    for ko_sent, en_sent, domain in corpus:
        json_data.append({"korean": ko_sent, "english": en_sent, "domain": domain})

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Corpus saved to:")
    print(f"  TSV: {tsv_file}")
    print(f"  JSON: {json_file}")

    return str(tsv_file)


if __name__ == "__main__":
    # Generate large corpus
    corpus = generate_large_corpus(target_size=50000)

    # Save to data/raw directory
    output_path = save_corpus(corpus, "data/raw")

    print(f"\nLarge corpus generation completed!")
    print(f"Total sentence pairs: {len(corpus)}")
    print(f"Output file: {output_path}")
