import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn # pylint: disable=wrong-import-position

class SemanticScaler:
    def __call__(self, *args, **kwds):
        return self.scale_up(*args, **kwds)

    def scale_up(self, word):
        synonyms = wn.synsets(word)
        if not synonyms:
            return word  # No synonyms found, return the original word
        # Get the first synonym's lemma name as a simple scaling example
        hypernyms = synonyms[0].hypernyms()
        if hypernyms:
            return hypernyms[0].lemmas()[0].name().replace('_', ' ')
        return word

class SynonymReplacer:
    def __call__(self, *args, **kwds):
        return self.replace_with_synonym(*args, **kwds)

    def replace_with_synonym(self, word):
        synonyms = wn.synsets(word)
        if not synonyms:
            return word  # No synonyms found, return the original word
        # Get the first synonym's lemma name as a simple replacement example
        synonym = synonyms[0].lemmas()[0].name().replace('_', ' ')
        if synonym.lower() != word.lower():
            return synonym
        return word
