from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Input text
text = """
Artificial Intelligence is rapidly changing various sectors 
by streamlining operations and enhancing decision-making processes. 
Its adoption in fields such as healthcare, education, finance, 
and technology is helping organizations boost productivity and improve customer service.
"""

# Parse text
parser = PlaintextParser.from_string(text, Tokenizer("english"))

# Initialize summarizer
summarizer = LsaSummarizer()

# Generate summary with 2 sentences
summary = summarizer(parser.document, 2)

print("\n===== ORIGINAL TEXT =====\n")
print(text)

print("\n===== GENERATED SUMMARY =====\n")

for sentence in summary:
    print(sentence)
