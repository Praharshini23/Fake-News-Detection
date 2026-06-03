
fake_df = pd.read_csv(" ")
true_df = pd.read_csv(" ")


fake_df["label"] = 0
true_df["label"] = 1  

df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)


print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nClass Distribution:\n", df['label'].value_counts())


class_counts = df['label'].value_counts()
plt.bar(class_counts.index, class_counts.values, color=['red', 'green'])
plt.xticks([0, 1], ['Fake', 'Real'])
plt.title("Fake vs Real News Count")
plt.xlabel("Class")
plt.ylabel("Number of Articles")
plt.show()


df["text_length"] = df["text"].apply(len)

plt.hist(df[df['label']==0]["text_length"], bins=50, alpha=0.6, color='red', label="Fake")
plt.hist(df[df['label']==1]["text_length"], bins=50, alpha=0.6, color='green', label="Real")
plt.title("Distribution of Text Length (Fake vs Real)")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.legend()
plt.show()


def get_top_words(texts, n=20):
    words = []
    for text in texts:
        words.extend(re.findall(r'\w+', str(text).lower()))
    return Counter(words).most_common(n)

print("\nTop 20 Words in Fake News:")
print(get_top_words(fake_df["text"]))

print("\nTop 20 Words in Real News:")
print(get_top_words(true_df["text"]))


print("\nAverage length of Fake news:", np.mean(fake_df["text"].apply(len)))
print("Average length of Real news:", np.mean(true_df["text"].apply(len)))
