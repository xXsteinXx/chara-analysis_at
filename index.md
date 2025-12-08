---
layout: default
---

## Introduction

The goal of this site is to create, explain, and analyze how a pre-trained, text-generation model like GPT-2 can be used as a supplemental tool for character analysis in literature or other media. In this case, creating two separate fine-tuned models using the first and last season’s transcripts of the Cartoon Network series, _Adventure Time_. Each fine-tuned model will be prompted with the same prompts to compare and contrast the characterization of the main character of the series, Finn the Human, from the first vs. last season. The goal is to showcase possible small use cases for text-generative AI, while critiquing the models, training, and use of gen-AI as a whole. This work is not a way to make a “creative” generative AI. This a small, specific use case and would serve poorly as the main tool of character analysis.

## What is Adventure Time

_Adventure Time_ is an award winning cartoon series created by Pendleton Ward that began airing in 2010 and wrapped its first series in 2018 (there are several spin-offs and mini-series). The show follows the adventures of Finn the Human and Jake the Dog, adoptive brothers who serve as the de-facto knights of the magical Land of Ooo. Finn is 12 years old, black-and-white thinking, violence seeking, and, as far as we know, the only living human in the first season. By the end of season 10, the last season of the series, Finn is 17, wisened, uses violence as a last resort, and has survived depression, break-ups, an absent father, and a Lich. The first season is full of juvenile jokes and pre-pubescent themes (Finn blushes around the older-than-him princess and enjoys punching bad guys and monsters). While there is no lack of silly jokes and scenarios in the final seasons, the show incorporates themes of maturation, sexuality, and existentialism that a casual viewer of the first season would have never expected.


## Downloading the Transcript

We need a dataset to fine-tune a pre-trained model. A dataset, in this case, is a file of sentences from which we will fine-tune the model. Fine-tuning is a process of further training a pre-trained model on a smaller dataset for use in specific tasks. Pre-trained models are available via [Hugging Face](https://huggingface.co/), a platform for those invested in machine learning to share and collaborate on models, datasets, and applications. The models available on the site can be utilized by users without downloading the full model on your personal device. Your fine-tuned model can be saved and utilized locally without further dependance on Hugging Face.

We will use the BeautifulSoup library to scrape the transcripts from the [_Adventure Time_ Wiki](navigate to https://adventuretime.fandom.com/wiki/Season_1), use loops to clean, then save the data as individual sentences in a csv file.

```py
# libraries to load in
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

```

```py
# format episode titles for use in url
season_1 = []

for i in s1:
    if ' ' in i:
        i = i.replace(' ', '_')
        season_1.append(i)
    else:
        season_1.append(i)

```

```py
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
        ```

```py

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
    'text' : s1_words # important to name text for fine-tuning later
})

df.to_csv('at_s1_text.csv')

```

## Fine-tuning the Model

The model used in this section is a fine-tuning of the [pretrained GPT-2 model](https://huggingface.co/openai-community/gpt2) and that can be further understood by the [model card](https://github.com/openai/gpt-2/blob/master/model_card.md).

The fine-tuning is best performed in [Google Colab](https://colab.research.google.com/) for easier interfacing with Hugging Face.

```py
#%pip install transformers datasets trl torch # run this one it's own first

```
```py
from transformers import pipeline
pipe = pipeline("text-generation", model="openai-community/gpt2")
```
```py
import pandas as pd
from datasets import Dataset

# load in the csv file containing lines from the first seaosn transcript
# for Google Drive users:
from google.colab import drive
drive.mount('/content/drive') # will be prompted by a pop-up window to connect to Drive
df = pd.read_csv('/content/drive/MyDrive/.../at_s1_trans.csv') # the entry here depends on where your file is stored on your computer
dataset = Dataset.from_pandas(df)
```
```py
# tokens are characters, words, and/or phrases that AI models use to process and generate text
# a basic workflow of tokens can be explained as:
# 1. text is borken into tokens
# 2. those tokens are received as inputs by the model and they are processed
# 3. the model, when prompted, outputs tokens
# 4. those tokens become text
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right"
```

```py
training_params = SFTConfig(
    output_dir='/content/drive/MyDrive/...', # depends on your working directory
    num_train_epochs = 3, # how many times the dataset is iterated over
    learning_rate = 2e-4, # sets "step size" that models takes to adjust size of changes the model makes, can also be cosnidered the "loss" between given data and output (often best adjusted through trial and error)
    weight_decay = 0.001, # regularizes the parameters to avoid overfitting (output is simlar or the same as the training data)
    dataset_text_field = "text",
    report_to="none", # disables all logging (data collections of the training process)
    fp16=True, # controls precision during training (math and hardware related)
    label_names = 'text',
)

# creates the "fine-tuner"
trainer = SFTTrainer(
    model = model,
    train_dataset = ds['train'],
    processing_class = tokenizer,
    args = training_params,

)
```

```py
# training time :D
# this will take somewhere around 10 minutes depending on your device
# the numbers in the "Training Loss" column should get smaller and smaller over time to show that the model's guesses are closer to the target words

# trainer.train() # commenting out in case it runs
```

## Loading in the Model



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