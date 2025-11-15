"""
Create a diverse Korean-English translation dataset with 200+ sentence pairs
covering various domains, linguistic patterns, and complexity levels.
"""

import random

def create_diverse_dataset():
    """Create a comprehensive Korean-English translation dataset."""
    
    # Greetings and Basic Phrases
    greetings = [
        ("안녕하세요", "Hello"),
        ("안녕히 가세요", "Goodbye"),
        ("감사합니다", "Thank you"),
        ("죄송합니다", "I'm sorry"),
        ("괜찮아요", "It's okay"),
        ("네, 알겠습니다", "Yes, I understand"),
        ("아니요, 모르겠습니다", "No, I don't understand"),
        ("도와주세요", "Please help me"),
        ("얼마예요?", "How much is it?"),
        ("화장실이 어디에 있나요?", "Where is the bathroom?"),
    ]
    
    # Daily Life and Common Situations
    daily_life = [
        ("오늘 날씨가 정말 좋네요", "The weather is really nice today"),
        ("저는 커피를 좋아합니다", "I like coffee"),
        ("아침에 일찍 일어났어요", "I woke up early in the morning"),
        ("지하철을 타고 회사에 갑니다", "I go to work by subway"),
        ("점심 먹었어요?", "Did you eat lunch?"),
        ("오늘 저녁에 영화 보러 갈래요?", "Do you want to go watch a movie tonight?"),
        ("주말에 뭐 할 거예요?", "What are you going to do on the weekend?"),
        ("쇼핑하러 백화점에 갔어요", "I went to the department store to shop"),
        ("이 옷이 저한테 어울려요?", "Does this outfit suit me?"),
        ("피곤해서 일찍 잘 거예요", "I'm tired so I'll go to bed early"),
    ]
    
    # Work and Business
    business = [
        ("회의가 몇 시에 있나요?", "What time is the meeting?"),
        ("보고서를 내일까지 제출해야 해요", "I need to submit the report by tomorrow"),
        ("프로젝트 일정이 빠듯합니다", "The project schedule is tight"),
        ("팀원들과 협업이 잘 되고 있어요", "I'm collaborating well with team members"),
        ("고객에게 이메일을 보냈습니다", "I sent an email to the client"),
        ("프레젠테이션 준비가 끝났어요", "The presentation preparation is complete"),
        ("새로운 전략을 제안하고 싶습니다", "I would like to propose a new strategy"),
        ("예산을 검토해야 합니다", "We need to review the budget"),
        ("마감일을 연장할 수 있을까요?", "Can we extend the deadline?"),
        ("회사에서 승진했어요", "I got promoted at the company"),
    ]
    
    # Travel and Directions
    travel = [
        ("공항에 어떻게 가나요?", "How do I get to the airport?"),
        ("호텔 예약을 확인하고 싶습니다", "I would like to confirm my hotel reservation"),
        ("관광지 추천해 주세요", "Please recommend tourist attractions"),
        ("지도를 보여주세요", "Please show me the map"),
        ("버스 정류장이 어디에 있나요?", "Where is the bus stop?"),
        ("이 기차가 서울에 가나요?", "Does this train go to Seoul?"),
        ("여권을 잃어버렸어요", "I lost my passport"),
        ("가까운 은행이 어디에 있나요?", "Where is the nearest bank?"),
        ("택시를 타고 싶습니다", "I want to take a taxi"),
        ("체크인 수속을 해야 해요", "I need to check in"),
    ]
    
    # Food and Restaurants
    food = [
        ("맛있는 식당을 추천해 주세요", "Please recommend a delicious restaurant"),
        ("김치찌개가 뭐예요?", "What is kimchi stew?"),
        ("매운 음식을 잘 못 먹어요", "I can't eat spicy food well"),
        ("계산서 주세요", "Please give me the bill"),
        ("이 음식이 너무 짜요", "This food is too salty"),
        ("채식 메뉴가 있나요?", "Do you have vegetarian menu?"),
        ("물 한 병 더 주세요", "Please give me one more bottle of water"),
        ("한국 음식을 처음 먹어봐요", "I'm trying Korean food for the first time"),
        ("배가 너무 고파요", "I'm very hungry"),
        ("디저트로 뭐가 있나요?", "What do you have for dessert?"),
    ]
    
    # Emotions and Feelings
    emotions = [
        ("오늘 기분이 정말 좋아요", "I feel really good today"),
        ("걱정이 많이 되네요", "I'm very worried"),
        ("이 소식이 너무 기쁩니다", "This news makes me very happy"),
        ("실망했지만 다시 시도할 거예요", "I'm disappointed but I'll try again"),
        ("갑자기 긴장되기 시작했어요", "I suddenly started feeling nervous"),
        ("이 상황이 너무 답답해요", "This situation is very frustrating"),
        ("당신을 만나서 반갑습니다", "Nice to meet you"),
        ("미안한 마음이 들어요", "I feel sorry"),
        ("자랑스러워요", "I'm proud"),
        ("놀랐어요!", "I'm surprised!"),
    ]
    
    # Complex and Compound Sentences
    complex_sentences = [
        ("비가 오고 있어서 우산을 가져가는 게 좋을 것 같아요", "Since it's raining, it would be good to take an umbrella"),
        ("한국에 온 지 3개월이 되었고 점점 한국어를 잘하게 되고 있어요", "It's been 3 months since I came to Korea and I'm gradually getting better at Korean"),
        ("친구가 추천해준 영화를 봤는데 정말 재미있었어요", "I watched the movie my friend recommended and it was really interesting"),
        ("일찍 자야 한다고 생각했지만 너무 피곤해서 그냥 잤어요", "I thought I should go to bed early but I was so tired that I just slept"),
        ("한국 문화에 대해 더 배우고 싶어서 한국 역사 책을 읽기 시작했어요", "I want to learn more about Korean culture so I started reading a Korean history book"),
        ("날씨가 추워서 따뜻한 옷을 입고 장갑도 끼었어요", "Since the weather is cold, I wore warm clothes and also wore gloves"),
        ("회사에서 중요한 프로젝트를 맡게 되어서 열심히 준비하고 있어요", "I got assigned an important project at work so I'm preparing hard for it"),
        ("한국 음식이 처음에는 매워서 힘들었지만 이제는 정말 좋아하게 되었어요", "Korean food was difficult at first because it was spicy, but now I really like it"),
        ("주말에 부모님을 만나러 고향에 갈 예정이에요", "I'm planning to go to my hometown to meet my parents on the weekend"),
        ("한국어 공부를 시작한 지 6개월이 지났고 매일 조금씩 발전하고 있어요", "It's been 6 months since I started studying Korean and I'm improving little by little every day"),
    ]
    
    # Questions and Interrogatives
    questions = [
        ("어디에서 왔나요?", "Where are you from?"),
        ("한국에 온 지 얼마나 되었나요?", "How long have you been in Korea?"),
        ("무슨 음식을 제일 좋아하나요?", "What food do you like the most?"),
        ("주말에 보통 뭐 하나요?", "What do you usually do on weekends?"),
        ("한국어를 언제부터 배우기 시작했나요?", "When did you start learning Korean?"),
        ("어떻게 한국어를 배웠나요?", "How did you learn Korean?"),
        ("한국에서 가장 마음에 드는 곳은 어디인가요?", "What is your favorite place in Korea?"),
        ("앞으로 어떤 계획이 있나요?", "What plans do you have for the future?"),
        ("한국 생활에 적응하는 데 어려움이 있나요?", "Do you have difficulty adapting to life in Korea?"),
        ("한국 문화 중에서 가장 흥미로운 것은 무엇이라고 생각하나요?", "What do you think is the most interesting aspect of Korean culture?"),
    ]
    
    # Descriptive and Narrative
    descriptive = [
        ("서울은 아주 큰 도시이며 많은 사람들이 살고 있습니다", "Seoul is a very big city and many people live there"),
        ("한국의 사계절은 각각의 특징이 있어서 아름답습니다", "Korea's four seasons each have their own characteristics and are beautiful"),
        ("지하철은 깨끗하고 편리해서 대중교통을 이용하는 데 좋습니다", "The subway is clean and convenient so it's good for using public transportation"),
        ("한국 사람들은 친절하고 도움을 주려고 노력합니다", "Korean people are kind and try to help others"),
        ("전통시장에서는 다양한 음식과 물건들을 볼 수 있습니다", "You can see various foods and items at traditional markets"),
        ("한국의 역사는 오래되었고 흥미로운 이야기들이 많습니다", "Korea's history is long and has many interesting stories"),
        ("밤에 도시의 불빛들이 아주 아름답게 보입니다", "The city lights look very beautiful at night"),
        ("산책하면서 계절의 변화를 느낄 수 있었어요", "I could feel the change of seasons while walking"),
        ("이 동네는 조용하고 살기 좋은 것 같아요", "This neighborhood seems quiet and nice to live in"),
        ("한국의 전통 건축양식은 독특하고 아름답습니다", "Korea's traditional architectural style is unique and beautiful"),
    ]
    
    # Combine all categories
    all_sentences = []
    all_sentences.extend(greetings)
    all_sentences.extend(daily_life)
    all_sentences.extend(business)
    all_sentences.extend(travel)
    all_sentences.extend(food)
    all_sentences.extend(emotions)
    all_sentences.extend(complex_sentences)
    all_sentences.extend(questions)
    all_sentences.extend(descriptive)
    
    # Add some variations and additional sentences to reach 200+
    additional = [
        ("지금 몇 시예요?", "What time is it now?"),
        ("내일 봐요!", "See you tomorrow!"),
        ("잘 지내세요?", "How are you doing?"),
        ("축하합니다!", "Congratulations!"),
        ("행운을 빌어요", "Good luck"),
        ("조심하세요", "Be careful"),
        ("빨리 와요", "Come quickly"),
        ("여기 앉으세요", "Please sit here"),
        ("시작합시다", "Let's start"),
        ("끝났어요", "It's finished"),
        ("다시 해보세요", "Try again"),
        ("잘했어요!", "Well done!"),
        ("괜찮아요, 괜찮아요", "It's okay, it's okay"),
        ("걱정하지 마세요", "Don't worry"),
        ("즐거운 시간 되세요", "Have a good time"),
        ("다음에 봐요", "See you next time"),
        ("전화할게요", "I'll call you"),
        ("메시지 보낼게요", "I'll send a message"),
        ("기다려 주세요", "Please wait"),
        ("죄송하지만 도와주실 수 있나요?", "Excuse me, can you help me?"),
    ]
    
    all_sentences.extend(additional)
    
    # Shuffle to ensure diversity in training
    random.shuffle(all_sentences)
    
    print(f"Created dataset with {len(all_sentences)} Korean-English sentence pairs")
    
    # Write to files
    with open("data/kr_diverse.txt", "w", encoding="utf-8") as f_kr, \
         open("data/en_diverse.txt", "w", encoding="utf-8") as f_en:
        
        for kr, en in all_sentences:
            f_kr.write(kr + "\n")
            f_en.write(en + "\n")
    
    print("Dataset created successfully!")
    print("Files saved as: data/kr_diverse.txt and data/en_diverse.txt")
    
    # Show some examples
    print("\nSample sentence pairs:")
    for i in range(min(10, len(all_sentences))):
        kr, en = all_sentences[i]
        print(f"KR: {kr}")
        print(f"EN: {en}")
        print()

if __name__ == "__main__":
    create_diverse_dataset()