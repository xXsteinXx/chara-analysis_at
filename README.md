# Considering Transformers Models as a Tool for Character Analysis with Finn the Human
### a guide for beginners and familiars in using a use-case specific, fine-tuned, pre-trained large language model

## project description
The goal of this project is to explain and analyze
how a pre-trained, text-generation model like GPT-2 can be used as a
supplemental tool for character analysis in literature or other media. In
this case I will be creating two separate fine tuned models using the
first and last season’s transcripts of the Cartoon Network series,
_Adventure Time_. I will prompt each model with the same prompts to compare
and contrast the characterization of the main character of the series,
Finn the Human, from the first to last season. The first season is full of
juvenile jokes and pre-pubescent themes (Finn blushes around the (older-
than-him) princess and enjoys punching bad guys and monsters). While there
is no lack of silly jokes and scenarios in the final seasons, the show
incorporates themes of maturation, sexuality, and existentialism that I as
a pre-pubescent middle schooler in the early 2010s would have never
expected. I will be testing if a fine-tuned model can reflect these
character changes by comparing and contrasting generated text from the
models fine-tuned on the first season and last season.

## rationale statement
It is only in recent years that scholarly research
has begun breaking down and analysing the entirety of _Adventure Time_. As
an avid enjoyer of the series in an era where art and research are under
threat by generative AI, I would like to analyze and critique the use of
transformers models as a character analysis tool. While this tool could be
useful for quick generalizations of the characters and be supplemental to
character analysis, it does not compare to the research and understanding
that a viewer of the series would be able to cultivate. For someone who is
not familiar with the material, only reading the model’s outputs would
likely give a poor imitation of the characters themselves. I will be
conducting this case study to highlight improper use of generative-AI and
the greater effectiveness (for research, efficiency, and energy use) of
training and/or fine-tuning a hyper-specific model not intending to be
“creatively” generative.

## workflow
I will be utilizing requests, transformers, pipelines, pandas,
BeautifulSoup4, lxml, re, and csv. My methods are to fine-tune
GPT-2 via Hugging Face on the transcripts from the first and tenth season
of the series. The transcripts were scrapped from the Adventure Time Wiki,
a fan-made and sustained Wikipedia page with community content mostly
under a CC-BY-SA license. Once fine-tuned, I prompted each model with
the starters “Finn is/is not", “Finn’s emotions are/are not”, and “Finn thinks/does not think”. I believe these
prompts, based on previous experimentation, will return results that
highlight Finn’s character, decisions, and emotions. Before prompting, I
conducted a bias check on the models by prompting “a woman is”,
“a man is”, and “a person is”. Doing so highlights potential gender
bias in the model and if the bias is due to the show
itself or the GPT-2 model. As a watcher of the series, female characters
were often treated as love interests in the series, so I am expected a
level of bias that generates for emotional responses for “a woman is”. If
something about women is generated that would not make sense within the
context of the show, than that would be bias inherent in the GPT-2 model.

## further uses

Throughout the process I will be explaining terms and
methods used in fine-tuning for anyone who is unfamiliar with the process
to understand aspects of generative AI. Many people, both who use and
reject AI, do not fully understand how generative AI works. I myself am
anti-AI by principle, but I do see and understand the merits of AI as an
aspect to work and research in the hands of trained professionals. That
being said, the materials for fine-tuning in this case study are
incredibly small and as of this moment I am only utilizing GPT-2. The bias
in this study depends on this limited dataset and model scope, so my
analysis of this tool is limited. I could mitigate this bias by testing
with other pre-trained models, but the transcripts are a set length. I
would also like to make an environmental impact statement, but am unsure
how to measure the energy consumption of my testing. More research needs
to be done before I can make any illuminating statement (perhaps a later
expansion for this project).

## files list: 

This guide/website explains how to create each of the following files yourself, but for accessibility/flexibitliy, you may suplement parts of the work with the follow files locating in the author_files folder (such as loading in the model instead of creating them yourself)

* **at_s1_model** = folder, fine-tuned transformers model (GPT-2) on the transcript of the first season of Adventure Time
* **at_s10_model** = folder, fine-tuned transformers model (GPT-2) on the transcript of the tenth season of Adventure Time
* **at_s1_text.csv** = csv of dialogue and stage directions (separated by sentence/direction) from the transcript of the first season of Adventure Time
* **at_s10_text.csv** = csv of dialogue and stage directions (separated by sentence/direction) from the transcript of the tenth season of Adventure Time
* **at_s1_train.py** = code to fine-tune transformers model on selected s1 csv
* **at_s1_analysis.py** = code to import created model with prompts
* **at_s10_train.py** = code to fine-tune transformers model on selected s10 csv
* **at_s10_analysis.py** = code to import created model with prompts
* **at_s1_outputs_all.csv** = csv file containing all generated text from each prompt from the model fine-tuned with season 1
* **at_s10_outputs_all.csv** = csv file containing all generated text from each prompt from the model fined-tuned with season 10
* **at_s1_scrape.py** = code scrape the first season transcript from the _Adventure Time_ wiki
* **at_s1_scrape.py** = code scrape the tenth season transcript from the _Adventure Time_ wiki
* **Transformers_Finn_Analysis.xlsx** = color coded excel sheet containing s1 and s10 outputs side-by-side for easier analysis

