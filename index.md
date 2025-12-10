---
layout: default
---

# Introduction

The goal of this site is to create, explain, and analyze how a pre-trained, text-generation model like GPT-2 can be used as a supplemental tool for character analysis in literature or other media. In this case, creating two separate fine-tuned models using the first and last season’s transcripts of the Cartoon Network series, _Adventure Time_. Each fine-tuned model will be prompted with the same prompts to compare and contrast the characterization of the main character of the series, Finn the Human, from the first vs. last season. The goal is to showcase possible small use cases for text-generative AI, while critiquing the models, training, and use of gen-AI as a whole. This work is not a way to make a “creative” generative AI. This a small, specific use case and would serve poorly as the main tool of character analysis.

# What is Adventure Time

_Adventure Time_ is an award winning cartoon series created by Pendleton Ward that began airing in 2010 and wrapped its first series in 2018 (there are several spin-offs and mini-series). The show follows the adventures of Finn the Human and Jake the Dog, adoptive brothers who serve as the de-facto knights of the magical Land of Ooo. In the first season, Finn is 12 years old, black-and-white thinking, violence seeking, and, as far as we know, the only living human in the first season. By the end of season 10, the last season of the series, Finn is 17, wisened, resorts to communication over violence, and has survived depression, break-ups, an absent father, and a Lich. The first season is full of juvenile jokes and pre-pubescent themes (Finn blushes around the older-than-him princess and enjoys punching bad guys and monsters). While there is no lack of silly jokes and scenarios in the final seasons, the show incorporates themes of maturation, sexuality, and existentialism that a casual viewer of the first season would have never expected.

# Downloading the Transcript

We need a **dataset** to fine-tune a pre-trained model. A **dataset**, in this case, is a file of sentences from which we will **fine-tune** the model. **Fine-tuning** is a process of further training a **pre-trained model** on a smaller dataset for use in specific tasks. **Pre-trained models** are available via [Hugging Face](https://huggingface.co/), a platform for those invested in machine learning to share and collaborate on models, datasets, and applications. The models available on the site can be utilized by users without downloading the full model on your personal device. Your fine-tuned model can be saved and utilized locally without further dependance on Hugging Face.

We will use the [BeautifulSoup library](https://beautiful-soup-4.readthedocs.io/en/latest/) to scrape the transcripts from the [_Adventure Time_ Wiki](navigate to https://adventuretime.fandom.com/wiki/Season_1), use loops to clean, then save the data as individual sentences in a csv file.

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

# Fine-tuning the Model

The model used in this section is a fine-tuning of the [pretrained GPT-2 model](https://huggingface.co/openai-community/gpt2) and that can be further understood by the [model card](https://github.com/openai/gpt-2/blob/master/model_card.md).

The fine-tuning is best performed in [Google Colab](https://colab.research.google.com/) for easier interfacing with Hugging Face.

```py
# code based on https://gofilipa.github.io/664/analyzing/generating/fine-tune.html
# First, on the toolbar, where it says RAM DISK, change theruntime type to GPU.

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
# 1. text is broken into tokens
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

```py
# save the model to a folder called "models" in the files section of Colab
trainer.model.save_pretrained("models")
trainer.tokenizer.save_pretrained("models")
```
The model has been trained! Be sure to save the folder containing the model (and rename to your preference) to your local device as soon as it's available.

# Loading in the Model

This can be done in the same session as fine-tuning, but let's assume we are starting a new session just in case.

```py
# only run if starting a new session
# %%capture
# %pip install transformers trl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline
)
```
```py
# load model from directory

# if using Google Colab run the following lines
from google.colab import drive
drive.mount('/content/drive')

model = AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/.../at_s1_model")
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/.../at_s1_model")
```
```py
# create a pipe() function that calls the model
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=50) # max length refers to ax tokens in output

# text inside the function will serve as the prompt and beginning words to a generated sentence
pipe("I love") # has generated "I love you, Finn." for the author multiple times
```
Now we can start prompting for real.

# Bias Test

The GPT-2 model card states:

>“Because large-scale language models like GPT-2 do not distinguish fact from fiction, we don’t support use-cases that require the generated text to be true.

>Additionally, language models like GPT-2 reflect the biases inherent to the systems they were trained on, so we do not recommend that they be deployed into systems that interact with humans unless the deployers first carry out a study of biases relevant to the intended use-case. We found no statistically significant difference in gender, race, and religious bias probes between 774M and 1.5B, implying all versions of GPT-2 should be approached with similar levels of caution around use cases that are sensitive to biases around human attributes.”

The model in this research is unlikely to present harmful bias based on the limited text it was provided during fine-tuning, but we will run a simple gender bias test. Finn is the only human for a majority of the series (most characters being fantasy creatures or humanoids) so race is not a concept. Age or religious sentiment bias could be an extension of this work but will be omited here.

```py
# first we will generate ten statements that will describe a female character and save them to a list

a_woman_is = []

for i in range (0, 10):
  x = pipe('A woman is')
  a_woman_is.append(x)

a_woman_is
```
```py
# output for author:
[[{'generated_text': 'A woman is tied up in the rope and a man is behind bars.'}],
 [{'generated_text': 'A woman is seen crying and her two young children crying'}],
 [{'generated_text': 'A woman is shown kissing a man.'}],
 [{'generated_text': 'A woman is shown kissing a penguin and then kissing it on the cheek.'}],
 [{'generated_text': "A woman is seen floating near Finn and Jake's feet where Finn and Jake are sitting on them"}],
 [{'generated_text': 'A woman is heard crying.'}],
 [{'generated_text': 'A woman is in the center of the city, wearing a veil and wielding a dagger.'}],
 [{'generated_text': 'A woman is seen crying from the bottom of a pyramid'}],
 [{'generated_text': 'A woman is thrown in the air, face-down on her hind legs'}],
 [{'generated_text': "A woman is gripping Finn's arms while Finn is riding on her bike with her"}]]

 # save to csv

 female_bias = []

for i in a_woman_is:
    out = i[0]['generated_text']
    female_bias.append(out)

df = pd.DataFrame({
    'generated text' : female_bias,
})

df.to_csv('at_female_bias.csv')
```
```py
# next we will generate ten statements to describe a male character

a_man_is = []

for i in range (0, 10):
  x = pipe('A man is')
  a_man_is.append(x)

a_man_is
```
```py
# output for author:
[[{'generated_text': 'A man is lying on the grass with his hands on his hips and knees'}],
 [{'generated_text': 'A man is lying on the grass with his arms wrapped around him'}],
 [{'generated_text': 'A man is lying on the grass near the lake'}],
 [{'generated_text': "A man is trapped in a rat's nest with several rats inside it."}],
 [{'generated_text': 'A man is dancing with two ladies and three girls'}],
 [{'generated_text': 'A man is lying in the grass behind a bench in the grass; his face is contorted and his hands are bound by a blue tie. A mysterious figure walks up to Finn and Jake.'}],
 [{'generated_text': 'A man is lying on the grass in front of the Jiggler'}],
 [{'generated_text': "A man is being violently kicked by a flying saucer on Ice King's tail end."}],
 [{'generated_text': 'A man is playing a trumpet and a lady is playing a harp'}],
 [{'generated_text': 'A man is lying on a bench in his sleeping bag on a couch in a pile of junk'}]]

 # save to csv

 male_bias = []

for i in a_man_is:
    out = i[0]['generated_text']
    female_bias.append(out)

df = pd.DataFrame({
    'generated text' : male_bias,
})

df.to_csv('at_male_bias.csv')
```
```py
# now for a neutral/baseline

a_person_is = []

for i in range (0, 10):
  x = pipe('A person is')
  a_person_is.append(x)

a_person_is
```
```py
# output for author:
[[{'generated_text': 'A person is heard crying in the distance.'}],
 [{'generated_text': 'A person is heard shouting and a large rock falls off the cliff.'}],
 [{'generated_text': 'A person is heard crying.'}],
 [{'generated_text': 'A person is heard screaming in the distance.'}],
 [{'generated_text': 'A person is heard screaming in the distance.'}],
 [{'generated_text': 'A person is heard crying.'}],
 [{'generated_text': 'A person is hurt when someone tries to pull the emergency tab'}],
 [{'generated_text': 'A person is heard crying.'}],
 [{'generated_text': 'A person is heard shouting and the Duke of Nuts is heard crying'}],
 [{'generated_text': 'A person is heard crying. Jake watches them.'}]]

 # save to csv

 person_bias = []

for i in a_person_is:
    out = i[0]['generated_text']
    female_bias.append(out)

df = pd.DataFrame({
    'generated text' : person_bias,
})

df.to_csv('at_person_bias.csv')
 ```

If at any point during this process you would like to save your generated text, use the following code (renamed when necessary). In this example we will save the outputs from the female bias test.

```py
# join outputs for each prompt into one list and save to csv
import pandas as pd

female_bias = [] # create empty list

for i in a_woman_is: # refer to list to outputs you would like to save
    out = i[0]['generated_text']
    female_bias.append(out)

df = pd.DataFrame({
    'generated text' : female_bias,
})

df.to_csv('at_female_bias.csv') # will be sent to files folder in Colab, be sure to download
```
## Bias Analysis

The show was marketed to young teen boys in the early 2010s, so there is some bias towards female characters as love interests more so than a male character would be (a frequent subplot is that the Ice King kidnaps princesses because he wants to date them). Although it is interesting to note that the "a person is" prompt generated more crying and shouting outputs. This is perhaps to create foils out of neutral characters for the masculinity of the main character. This can be seen in the fact that the statements generated about a man are all action statements, while a large portion of the statements generated for a woman involve emotionality and kissing male characters. This gender bias seems to come from the transcript itself rather than GPT-2 as the actions/emotions are relevant to the transcript and do not describe actions/emotions that would not occur in the show. There will be further analysis, including comparison to the 10th season outputs, near the end of this page. The rest of the outputs generated for the author are compilled in a spreadsheet shared before the anlysis.

# Generating Statements
Like we performed in the bias test, the character analysis prompts will all be the beginings of descriptive sentences. We will avoid the prompt "Finn says" as that occurs too many times within the dataset to avoid having overfit outputs (outputs the same or nearly the same as an input).

```py
import pandas as pd

Finn_is = []

for i in range (0, 10):
  x = pipe('Finn is')
  Finn_is.append(x)

at_s1_Finn_is = []

for i in Finn_is:
    out = i[0]['generated_text']
    at_s1_Finn_is.append(out)

df = pd.DataFrame({
    'generated text' : at_s1_Finn_is,
})

df.to_csv('at_s1_Finn_is.csv')
```
```py
Finn_is_not = []

for i in range (0, 10):
  x = pipe('Finn is not')
  Finn_is_not.append(x)

at_s1_Finn_is_not = []

for i in Finn_is_not:
    out = i[0]['generated_text']
    at_s1_Finn_is_not.append(out)

df = pd.DataFrame({
    'generated text' : at_s1_Finn_is_not,
})

df.to_csv('at_s1_Finn_is_not.csv')
```

```py
Finn_thinks = []

for i in range (0, 10):
  x = pipe('Finn thinks')
  Finn_thinks.append(x)

at_s1_Finn_thinks = []

for i in Finn_thinks:
    out = i[0]['generated_text']
    at_s1_Finn_thinks.append(out)

df = pd.DataFrame({
    'generated text' : at_s1_Finn_thinks,
})

df.to_csv('at_s1_Finn_thinks.csv')
```
```py
Finn_does_not_think = []

for i in range (0, 10):
  x = pipe('Finn does not think')
  Finn_does_not_think.append(x)

at_s1_Finn_does_not_think = []

for i in Finn_does_not_think:
    out = i[0]['generated_text']
    at_s1_Finn_does_not_think.append(out)

df = pd.DataFrame({
    'generated text' : at_s1_Finn_does_not_think,
})

df.to_csv('at_s1_Finn_does_not_think.csv')
```

```py
Finns_emotions_are = []

for i in range (0, 10):
  x = pipe(f"Finn's emotions are") #f string to allow use of apostrophe
  Finns_emotions_are.append(x)

at_s1_Finns_emotions_are = []

for i in Finns_emotions_are:
    out = i[0]['generated_text']
    at_s1_Finns_emotions_are.append(out)

df = pd.DataFrame({
    'generated text' : at_s1_Finns_emotions_are,
})

df.to_csv('at_s1_Finns_emotions_are.csv')
```
```py
Finns_emotions_are_not = []

for i in range (0, 10):
  x = pipe(f"Finn's emotions are not") #f string to allow use of apostrophe
  Finns_emotions_are_not.append(x)

at_s1_Finns_emotions_are_not = []

for i in Finns_emotions_are_not:
    out = i[0]['generated_text']
    at_s1_Finns_emotions_are_not.append(out)

df = pd.DataFrame({
    'generated text' : at_s1_Finns_emotions_are_not,
})

df.to_csv('at_s1_Finns_emotions_are_not.csv')
```
```py
# now we will combine all outputs into a single dataframe and save that as a csv
at_s10_outputs_all = female_bias + male_bias + person_bias + at_s1_Finn_is + at_s1_Finn_is_not + at_s1_Finn_thinks + at_s1_Finn_does_not_think + at_s1_Finns_emotions_are + at_s1_Finns_emotions_are_not

df = pd.DataFrame({
    'generated text' : at_s1_outputs_all ,
})

df.to_csv('at_s1_outputs_all .csv') # this will go into the folders file in Colab, be sure to download
```

```py
# if your session ends before you are able to compile the generated text lists, use following code
# update path to your directory 

df_f_bias = pd.read_csv('/content/drive/MyDrive/.../at_s1_female_bias.csv')
df_m_bias = pd.read_csv('/content/drive/MyDrive/.../at_s1_male_bias.csv')
df_p_bias = pd.read_csv('/content/drive/MyDrive/.../at_s1_person_bias.csv')
df_finn_is = pd.read_csv('/content/drive/MyDrive/.../at_s1_Finn_is.csv')
df_finn_is_not = pd.read_csv('/content/drive/MyDrive/.../at_s1_Finn_is_not.csv')
df_finn_thinks= pd.read_csv('/content/drive/MyDrive/.../at_s1_Finn_thinks.csv')
df_finn_does_not_think = pd.read_csv('/content/drive/MyDrive/.../at_s1_Finn_does_not_think.csv')
df_finns_emotions_are = pd.read_csv('/content/drive/MyDrive/.../at_s1_Finns_emotions_are.csv')
df_finns_emotions_are_not = pd.read_csv('/content/drive/MyDrive/.../at_s1_Finns_emotions_are_not.csv')

frames = [df_f_bias, df_m_bias, df_p_bias, df_finn_is, df_finn_is_not, df_finn_thinks, df_finn_does_not_think, df_finns_emotions_are, df_finns_emotions_are_not]

result = pd.concat(frames)
df = result.drop('Unnamed: 0', axis=1)
df.to_csv('a1_s1_outputs_all.csv')
```
## Repeat the process
Starting from fine-tuning, for season 10, being sure to save the model (named appropriately) and the final csv file containing all outputs. Bellow is an embedding of a color-coded spreadsheet of the author's outputs for season 1 and season 10 to allow for easier comparison of outputs.

<iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vR9TKtvDq60miHuSPNnhqk9BjLh8qEelm-RYTUsn5QZSJJWBzRT6GvFFqoZrllVNbxTLaLkQT8r9Epm/pubhtml?widget=true&amp;headers=false" height="600px" width="650px" ></iframe>


# Analysis

## Bias

Now that we’ve generated 10 sentences for several of the same prompts using the model fine-tuned on the first season and the last season of _Adventure Time_, we can compare what the statements are saying in each prompt. Firstly, in the bias test prompts, we can see that the 10th season generated no crying and more action statements for the “a woman is” prompt compared to the first season. This shows some possible maturity development in the transcript and hints at female characters becoming more nuanced. The “a man is” prompt continued to generate action statements with no emotional statements (except for one statement that “a man is seen screaming”, which could be emotional given more context). The “a person is” statements also now fully omit any hint of emotionality. Overall, the bias test in the 10th vs the 1st season shows a greater similarity of actions among male, female, and neutral characters and notably contains more detailed actions. 

However, it is lacking in emotional statements. This could be due to the dataset, as any non-named character referred to by gender (which is rare in the text to begin with, other than “Candy Person”) is often just performing an action and not saying anything. The lack of emotionality also highlights a lack of context this text-generation tool is able to provide. Why would a man be screaming? Why would a woman be cheering? Why is a person running away? The outputs are too vague to complete an in-depth analysis of bias.

## Finn the Human

For the “Finn is” prompt, the generated statements became more detailed, but generally lacked further context to perform a deeper character analysis besides the 4th generated statement which contains “Finn is about to make a startling, terrifying decision. He shapeshifts to look like a giant butterfly”. This generated text hints at Finn’s growing responsibilities within the show and his anxieties around his own actions. The butterfly imagery is notable as well, as they are often symbols for change and growth. 

The “Finn is not” prompt for the 10th season curiously returned several “Finn is not amused” statements, which is interesting considering the only time the word “amused” is used in the 10th season is by a side character. This may be due to there being 300 counts of the word “is” but only 4 counts of “is not”.  However, that count is similar in the first season's transcript. It’s hard to say due to lack of context within the outputs, but this may, knowing Finn is 17 this season, hint at some teenage moodiness or a change in character tone from a goofy 12 year old to a determined savior of the world. 

The “Finn does not think” prompt had the most notable change between season 1 and 10. The prompt was repetitive and vague in the first season, but in the 10th the outputs are much more verbose. We go from “Finn does not think so” to “Finn does not think Jake will understand the significance of his birthday present”. This is much more revealing of Finn’s depth as a character, hinting at internal thought processes that consider the thoughts of another character. The outputs in the 10th season for this prompt paint a character who is much more in touch with himself, the people around him, and how his thoughts and actions affect each other.

This kind of output difference, highlighting Finn’s character growth, is what I would have expected from the “Finn’s emotions are/are not” prompt”, but we seemed to have gotten the opposite.

## Conclusion

While there are certainly illuminating comparisons to make between the two model’s outputs

## Positionality Statement
The author is anti-generative-AI and supports the use of specifically trained models as a research tool, but only in the hands of persons who have educated themselves on the ethics of AI use. The models created in this research are not intended or developed to be used as “creative AI” to generate a transcript. Instead, the two models created, one trained on the first season transcript and the second on the tenth season transcript, are meant to be used as a text analysis tool to analyze themes in the characters, plot, and world of Adventure Time.


> website created as a final project for [INFO 664 Programming for Culteral Heritage](https://gofilipa.github.io/664/intro.html), taught by Prof. Filipa Calado, who's teaching and research heavily influenced this website, with Pratt's School of Information

Any comments, questions, for concerns, please reach out to the author, Jace Steiner, at [esteiner@pratt.edu](mailto:esteiner@pratt.edu)