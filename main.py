import kagglehub
import pandas as pd
import joblib
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Adding missing import

# Download latest version
path = kagglehub.dataset_download("jainpooja/fake-news-detection")
print("Path to dataset files:", path)

# Load datasets
df_fake = pd.read_csv(f"{path}/Fake.csv")
df_true = pd.read_csv(f"{path}/True.csv")

# Display first 5 rows of each dataset
print("\n\nTrue news sample:")
print(df_fake.head(5))
print("\n\n\nFake news sample:")
print(df_true.head(5))

# Add class labels
df_fake["class"] = 0  # 0 for fake
df_true["class"] = 1  # 1 for true
print(f"\n\nDataset shapes - Fake: {df_fake.shape}, True: {df_true.shape}")

# Removing last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10)
for i in range(23480, 23470, -1):
    df_fake.drop([i], axis=0, inplace=True)

df_true_manual_testing = df_true.tail(10)
for i in range(21416, 21406, -1):
    df_true.drop([i], axis=0, inplace=True)
    
print(f"After removing test data - Fake: {df_fake.shape}, True: {df_true.shape}\n\n")

# Add class labels to test data
df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

# Combine test datasets and save
df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
df_manual_testing.to_csv("manual_testing.csv")

# Merge main datasets
df_merge = pd.concat([df_fake, df_true], axis=0)
print("\n\nMerged dataset preview:")
print(df_merge.head(10))
print(f"\nMerged dataset columns: {df_merge.columns}")

# Drop unnecessary columns
df = df_merge.drop(["title", "subject", "date"], axis=1)
print(f"\nNull values in dataset: {df.isnull().sum()}")

# Shuffle the dataset
df = df.sample(frac=1)
df.reset_index(inplace=True)
df.drop(["index"], axis=1, inplace=True)
print("\n\nShuffled dataset preview:")
print(df.head())

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply text preprocessing
df["text"] = df["text"].apply(wordopt)

# Prepare features and target
x = df["text"]
y = df["class"]

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Vectorize text data
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Initialize and train Logistic Regression model
LR = LogisticRegression()
LR.fit(xv_train, y_train)

# Make predictions
pred_lr = LR.predict(xv_test)

# Evaluate model
print(f"\n\nLogistic Regression Accuracy: {LR.score(xv_test, y_test)}")
print("Classification Report:")
print(classification_report(y_test, pred_lr))

# Function to output prediction label
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# Function for manual testing
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    return print(f"\nLR Prediction: {output_lable(pred_LR[0])}")

# Visualizations
# # Distribution of fake and real news
# plt.figure(figsize=(8, 6))
# sns.countplot(data=df_merge, x='class')
# plt.title('Distribution of Fake and Real News')
# plt.xlabel('Class (0: Fake, 1: Real)')
# plt.ylabel('Number of Articles')
# plt.show()

# # Text length distribution
# df['text_length'] = df['text'].apply(len)
# plt.figure(figsize=(10, 6))
# sns.histplot(data=df, x='text_length', hue='class', kde=True)
# plt.title('Distribution of Text Length by Class')
# plt.xlabel('Text Length')
# plt.ylabel('Frequency')
# plt.legend(title='Class', labels=['Fake', 'Real'])
# plt.show()

# # Boxplot of text length by class
# plt.figure(figsize=(8, 6))
# sns.boxplot(data=df, x='class', y='text_length')
# plt.title('Text Length Distribution by Class')
# plt.xlabel('Class (0: Fake, 1: Real)')
# plt.ylabel('Text Length')
# plt.xticks([0, 1], ['Fake', 'Real'])
# plt.show()

# Feature importance visualization
# if 'vectorization' in locals() and 'LR' in locals():
#     feature_names = vectorization.get_feature_names_out()
#     coefficients = LR.coef_[0]

#     feature_importance = pd.DataFrame({'feature': feature_names, 'importance': coefficients})
#     feature_importance = feature_importance.sort_values(by='importance', ascending=False)

#     # Top 20 most important features
#     plt.figure(figsize=(12, 8))
#     sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
#     plt.title('Top 20 Most Important Features (Logistic Regression)')
#     plt.xlabel('Coefficient Value')
#     plt.ylabel('Feature (Word)')
#     plt.show()

#     # Top 20 least important features
#     plt.figure(figsize=(12, 8))
#     sns.barplot(x='importance', y='feature', 
#                 data=feature_importance.tail(20).sort_values(by='importance', ascending=True))
#     plt.title('Top 20 Least Important Features (Logistic Regression)')
#     plt.xlabel('Coefficient Value')
#     plt.ylabel('Feature (Word)')
#     plt.show()
# else:
#     print("Please ensure 'vectorization' and 'LR' objects are defined before running the feature importance plots.")

# Example test cases
test_cases = [
    """BRUSSELS (Reuters) - NATO allies on Tuesday welcomed President Donald Trump s decision to commit more forces to Afghanistan, as part of a new U.S. strategy he said would require more troops and funding from America s partners. Having run for the White House last year on a pledge to withdraw swiftly from Afghanistan, Trump reversed course on Monday and promised a stepped-up military campaign against Taliban insurgents, saying: Our troops will fight to win. U.S. officials said he had signed off on plans to send about 4,000 more U.S. troops to add to the roughly 8,400 now deployed in Afghanistan. But his speech did not define benchmarks for successfully ending the war that began with the U.S.-led invasion of Afghanistan in 2001, and which he acknowledged had required an extraordinary sacrifice of blood and treasure. We will ask our NATO allies and global partners to support our new strategy, with additional troops and funding increases in line with our own. We are confident they will, Trump said. That comment signaled he would further increase pressure on U.S. partners who have already been jolted by his repeated demands to step up their contributions to NATO and his description of the alliance as obsolete - even though, since taking office, he has said this is no longer the case. NATO Secretary General Jens Stoltenberg said in a statement: NATO remains fully committed to Afghanistan and I am looking forward to discussing the way ahead with (Defense) Secretary (James) Mattis and our Allies and international partners. NATO has 12,000 troops in Afghanistan, and 15 countries have pledged more, Stoltenberg said. Britain, a leading NATO member, called the U.S. commitment very welcome. In my call with Secretary Mattis yesterday we agreed that despite the challenges, we have to stay the course in Afghanistan to help build up its fragile democracy and reduce the terrorist threat to the West, Defence Secretary Michael Fallon said. Germany, which has borne the brunt of Trump s criticism over the scale of its defense spending, also welcomed the new U.S. plan. Our continued commitment is necessary on the path to stabilizing the country, a government spokeswoman said. In June, European allies had already pledged more troops but had not given details on numbers, waiting for the Trump administration to outline its strategy for the region.Nearly 16 years after the U.S.-led invasion - a response to the Sept. 11 attacks which were planned by al Qaeda leader Osama bin Laden from Afghanistan - the country is still struggling with weak central government and a Taliban insurgency. Trump said he shared the frustration of the American people who were weary of war without victory, but a hasty withdrawal would create a vacuum for groups like Islamic State and al Qaeda to fill.""",
    
    """Vic Bishop Waking TimesOur reality is carefully constructed by powerful corporate, political and special interest sources in order to covertly sway public opinion. Blatant lies are often televised regarding terrorism, food, war, health, etc. They are fashioned to sway public opinion and condition viewers to accept what have become destructive societal norms.The practice of manipulating and controlling public opinion with distorted media messages has become so common that there is a whole industry formed around this. The entire role of this brainwashing industry is to figure out how to spin information to journalists, similar to the lobbying of government. It is never really clear just how much truth the journalists receive because the news industry has become complacent. The messages that it presents are shaped by corporate powers who often spend millions on advertising with the six conglomerates that own 90% of the media:General Electric (GE), News-Corp, Disney, Viacom, Time Warner, and CBS. Yet, these corporations function under many different brands, such as FOX, ABC, CNN, Comcast, Wall Street Journal, etc, giving people the perception of choice As Tavistock s researchers showed, it was important that the victims of mass brainwashing not be aware that their environment was being controlled; there should thus be a vast number of sources for information, whose messages could be varied slightly, so as to mask the sense of external control. ~ Specialist of mass brainwashing, L. WolfeNew Brainwashing Tactic Called AstroturfWith alternative media on the rise, the propaganda machine continues to expand. Below is a video of Sharyl Attkisson, investigative reporter with CBS, during which she explains how astroturf, or fake grassroots movements, are used to spin information not only to influence journalists but to sway public opinion. Astroturf is a perversion of grassroots. Astroturf is when political, corporate or other special interests disguise themselves and publish blogs, start facebook and twitter accounts, publish ads, letters to the editor, or simply post comments online, to try to fool you into thinking an independent or grassroots movement is speaking. ~ Sharyl Attkisson, Investigative ReporterHow do you separate fact from fiction? Sharyl Attkisson finishes her talk with some insights on how to identify signs of propaganda and astroturfing These methods are used to give people the impression that there is widespread support for an agenda, when, in reality, one may not exist. Astroturf tactics are also used to discredit or criticize those that disagree with certain agendas, using stereotypical names such as conspiracy theorist or quack. When in fact when someone dares to reveal the truth or questions the official story, it should spark a deeper curiosity and encourage further scrutiny of the information.This article (Journalist Reveals Tactics Brainwashing Industry Uses to Manipulate the Public) was originally created and published by Waking Times and is published here under a Creative Commons license with attribution to Vic Bishop and WakingTimes.com. It may be re-posted freely with proper attribution, author bio, and this copyright statement. READ MORE MSM PROPAGANDA NEWS AT: 21st Century Wire MSM Watch Files"""
]

# Test each example
print("\n--- Testing Example News Articles ---")
for i, news_text in enumerate(test_cases, 1):
    print(f"\nTesting article #{i}:")
    manual_testing(news_text)

# Interactive mode
def interactive_testing():
    while True:
        print("\n\nEnter a news article to classify (or 'quit' to exit):")
        news = input()
        if news.lower() == 'quit':
            break
        manual_testing(news)

# Uncomment to run interactive mode
# interactive_testing()

joblib.dump(LR, "model.jb")
joblib.dump(vectorization, "vectorizer.jb")
print("Model and vectorizer saved as model.jb and vectorizer.jb")