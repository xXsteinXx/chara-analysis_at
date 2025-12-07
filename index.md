---
layout: default
---

## Introduction

This site is meant to be a sort of essay/instruction/analysis all in one. There will be blocks of text explaining and analyzing the various code blocks and their outputs, as well as reflecting. Specifically, this research will be analysing the character Finn the Human and the Land of Ooo as they are written/presented in the first (Season 1) and last (Season 10) season of the show. The goal is to showcase possible small usecases for text generative AI, while critiquing the models, training, and use of gen-AI as a whole. This model training is is meant to create a supplementel tool for literature/media anlysis, not a "creative" generative AI. This a small, specific use case and would serve poorly as the main tool of character analysis.

## Considerations

## What is Adventure Time

## Downloading the Transcript

We need a dataset to train our model. We will use Beautiful Soup to scrape the trancripts, use loops to clean, then save the data as individual sentences in a csv file. For a list of episodes that aired in the first seaosn of Adventure Time, navigate to https://adventuretime.fandom.com/wiki/Season_1.


```py
import requests
from bs4 import BeautifulSoup as BS
import lxml
import re
import pandas as pd
import csv

# create list of episode titles from season 1

s1 = [
    "Slumber Party Panic",
    "Trouble in Lumpy Space",
    "Prisoners of Love",
    "Tree Trunks",
    "The Enchiridion!",
    "The Jiggler",
    "Ricardio the Heart Guy",
    "Business Time",
    "My Two Favorite People",
    "Memories of Boom Boom Mountain",
    "Wizard",
    "Evicted!",
    "City of Thieves",
    "The Witch's Garden",
    "What is Life?",
    "Ocean of Fear",
    "When Wedding Bells Thaw",
    "Dungeon",
    "The Duke",
    "Freak City",
    "Donny",
    "Henchman",
    "Rainy Day Daydream",
    "What Have You Done?",
    "His Hero",
    "Gut Grinder"
]

# format episode titles for use in url

season_1 = []

for i in s1:
    if ' ' in i:
        i = i.replace(' ', '_')
        season_1.append(i)
    else:
        season_1.append(i)

# fetch trancripts for each episode and add to single list
# NOTE: this Community content is available under CC-BY-SA unless otherwise noted.

s1_transcript = []

for ep in season_1:
    url = f'https://adventuretime.fandom.com/wiki/{ep}/Transcript'
    site = requests.get(url)
    source_code = site.content
    soup = BS(source_code, 'lxml')
    transcript = soup.find_all('dd')
    for i in transcript:
        s1_transcript.append(i)

# convert b4.element.tag to strings

s1_trans_str = []

for i in s1_transcript:
    string = str(i)
    s1_trans_str.append(string)

# remove superflous stuff

s1_sentences = []

for x in s1_trans_str:
    x = x.replace(']', '')
    x = x.replace('[', '')
    x = x.replace('(', '')
    x = x.replace(')', '')
    x = re.sub('<.+?>', '', x)
    x = x.replace(':', ' says') # the ':' signifies a character line
    s1_sentences.append(x)

df = pd.DataFrame({
    'text' : s1_words # import to name text for transformers training
})

df.to_csv('at_s1_text.csv')

```

## Training the Model

The model used in this section is a fine tuning of the pretrained GPT-2 model found here: https://huggingface.co/openai-community/gpt2 and that can be further understood by the model card found here: https://github.com/openai/gpt-2/blob/master/model_card.md.

The model training is best performed in Google Collab for easier interfacing with Hugging Face.

```py
pip install transformers datasets trl torch # run this one it's own first

```
```py
from transformers import pipeline
pipe = pipeline("text-generation", model="openai-community/gpt2")
```
```py
import pandas as pd
from datasets import Dataset

# load in the csv file containing lines from the first seaosn transcript
df = pd.read_csv('at_s1_trans.csv') # the entry here depends on where your file is stored on your computer
dataset = Dataset.from_pandas(df)
```

## Loading in the Model

```py
# python code goes here
```

## Bias Test

The model card states:

"Because large-scale language models like GPT-2 do not distinguish fact from fiction, we don’t support use-cases that require the generated text to be true.

Additionally, language models like GPT-2 reflect the biases inherent to the systems they were trained on, so we do not recommend that they be deployed into systems that interact with humans unless the deployers first carry out a study of biases relevant to the intended use-case. We found no statistically significant difference in gender, race, and religious bias probes between 774M and 1.5B, implying all versions of GPT-2 should be approached with similar levels of caution around use cases that are sensitive to biases around human attributes."

The model in this research is unlikely to present harmful bias based on the limited text it was provided to fine tuning, but we will run a short bias test to confirm this.

```py
# python code goes here
```
### Analysis

## Generating Statements from Season 1

## Generating Statements from Season 10

## Analysis

```py
# python code goes here
```

## Discussion

## Conclusion

## Positionality Statement 

The author is anti-generative-AI and supports the use of specificly trained models as a research tool, but only in the hands of persons who have educated themselves on the ethics of AI use. The models created in this research are not intended or developped to be used as "creative AI" to generate a transcript. Instead, the two models created, one trained on the first season transcript and the second on the tenth season transcript, are meant to be used as a text analysis tool to anilyse themes in the characters, plot, and world of Adventure Time.