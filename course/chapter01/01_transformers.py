#!/usr/bin/python

"""
Transformers, what can they do?
HuggingFace course
Chapter 1, exercise 1
"""

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
zero_shot_classifier = pipeline("zero-shot-classification")
generator = pipeline("text-generation")
gpt2_generator = pipeline("text-generation", model="gpt2")
unmasker = pipeline("fill-mask")
ner = pipeline("ner", grouped_entities=True)
question_answering = pipeline("question-answering")
summarizer = pipeline("summarization")
translation = pipeline("translation", model="Helsinki-NLP/opus-mt-no-es")

# Sentiment analysis
print(classifier("Jeg hater å være syk"))
print(classifier("Jeg elsker å være frisk"))

# Text generation
print(generator("Jeg elsker å være frisk"))
print(gpt2_generator("Jeg elsker å være frisk", max_length=50, num_return_sequences=3))

# Zero-shot classification
print(
    zero_shot_classifier(
        "Jeg elsker å være frisk", candidate_labels=["positive", "negative"]
    )
)

# Masked language modeling
print(
    unmasker(
        "Most old rich people traveling to Thailand are looking for some <mask>.",
        top_k=5,
    )
)

# Named entity recognition
print(ner("Hva heter du? Jeg heter Lars. Jeg bor i Oslo."))

# Question answering
print(
    question_answering(
        question="Hva heter du?", context="Jeg heter Lars. Jeg bor i Oslo."
    )
)

# Summarization
print(
    summarizer(
        "På Vårt Land-målingen får Arbeiderpartiet (Ap) kun 15,4 prosent oppslutning. \
        Det ville vært Arbeiderpartiets dårligste valgresultat på 116 år, \
        et halvt prosentpoeng under Aps oppslutning ved valget i 1906. \
        Da var det innført allmenn stemmerett for menn, mens kvinner ennå ikke fikk stemme. \
        Aps oppslutning på denne målingen er 2,6 prosentpoeng lavere enn på Vårt Lands oktober-måling. \
        Høyre fortsetter derimot sin reise oppover 30-tallet, og får 33,6 prosent på målingen. \
        Det vil si at Erna Solbergs Høyre er dobbelt så stort som Jonas Gahr Støres Ap. \
        Regjeringspartner Sp får skarve fem prosent. \
        Katastrofemålingene har kommet på rekke og rad for statsministerens parti denne måneden, \
        og hele to målinger har vist oppslutning nede på 16-tallet. \
        VGs november-måling var den dårligste for Ap siden 2006, da Respons begynte med meningsmålinger: \
        Ap fikk 18,5 prosent av stemmene, mens Høyre fikk til 30 prosent. \
        I et VG-intervju tirsdag sier stortingsrepresentant og tidligere Ap-nestleder Hadia Tajik \
        at partiet hennes må tørre å tenke nytt. – Vi trenger en ny sosialpolitikk, sier Tajik. \
        Fattigdommen har økt. Det er stadig flere barn som vokser opp i fattige familier, \
        med alt det kan få av konsekvenser både for livet som barn og ungdom, og ringvirkninger \
        inn i voksenlivet. Det er fakta på bakken som har endret seg, og da må politikken endre seg også, fortsetter hun."
    )
)

# Translation
print(translation("Jeg heter Lars. Jeg bor i Oslo."))