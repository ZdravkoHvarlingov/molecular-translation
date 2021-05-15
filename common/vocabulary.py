from collections import Counter
from tqdm import tqdm


class Vocabulary:
    def __init__(self):
        #setting the pre-reserved tokens int to string tokens
        
        all_possible_words = [ 
            '<PAD>', '<SOS>', '<EOS>', '<UNK>', 'C', ')', 'P', 'l', '=', '3', 'N', 'I', '2', '6', 'H', '4', 'F', '0', '1', '-', 'O', '8', ',', 'B', '(', '7', 'r', '/', 'm', 'c', 's', 'h', 'i', 't', 'T', 'n', '5', '+', 'b', '9', 'D', 'S'
        ]
        self.stoi, self.itos = self._words_to_stoi_itos(all_possible_words)
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [char for char in text]
    
    def build_vocab(self, sentence_list, freq_threshold):
        frequencies = Counter()
        for sentence in tqdm(sentence_list, position=0, leave=True):
            for word in self.tokenize(sentence):
                frequencies[word] += 1
        
        all_words = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        common_words = [word for word, count in frequencies.items() if count >= freq_threshold]
        all_words.extend(common_words)

        self.stoi, self.itos = self._words_to_stoi_itos(all_words)

    def _words_to_stoi_itos(self, words):
        #string to int tokens
        stoi = {token: idx for idx, token in enumerate(words)} 
        #its reverse dict self.itos
        itos = {idx: token for token, idx in stoi.items()}

        return stoi, itos
                
    def numericalize(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [(self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]) for token in tokenized_text] 
