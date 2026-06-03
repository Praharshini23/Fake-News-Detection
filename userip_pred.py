
while True:

    print("\nEnter News Article")

    news = input("News: ")

    news = clean_text(news)

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    print("\nPrediction:", prediction[0])

    choice = input("\nCheck another news? (y/n): ")

    if choice.lower() != 'y':
        break
