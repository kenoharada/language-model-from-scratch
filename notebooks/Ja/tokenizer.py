class CharUTF8Tokenizer:
    def __init__(self, vocab):
        if vocab is None:
            vocab = {}
        vocab_size = len(vocab)
        for i in range(256):
            if f'<utf8_{i}>' not in vocab:
                vocab[f'<utf8_{i}>'] = vocab_size + i
        self.vocab = vocab

    def encode(self, text):
        result = []
        for char in text:
            if char not in self.vocab:
                utf_8_num = list(char.encode("utf-8"))
                for num in utf_8_num:
                    result.append(self.vocab[f'<utf8_{num}>'])
            else:
                result.append(self.vocab[char])
        return result
    
    def decode_with_utf_token(self, tokens):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        decoded_with_utf_token = [inv_vocab[token] for token in tokens]
        return "".join(decoded_with_utf_token)
    
    def decode(self, tokens):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        decoded_with_utf_token = [inv_vocab[token] for token in tokens]
        decoded_postprocess_utf = []
        utf_tokens = []
        for token in decoded_with_utf_token:
            if token.startswith("<utf8_"):
                utf_num = int(token.replace("<utf8_", "").replace(">", ""))
                utf_tokens.append(utf_num)
            else:
                if utf_tokens:
                    decoded_postprocess_utf.append(bytes(utf_tokens).decode("utf-8"))
                    utf_tokens = []
                decoded_postprocess_utf.append(token)
        if utf_tokens:
            decoded_postprocess_utf.append(bytes(utf_tokens).decode("utf-8"))
            utf_tokens = []
        return "".join(decoded_postprocess_utf)


class PairTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            vocab = {}
        vocab_size = len(vocab)
        for i in range(256):
            if f'<utf8_{i}>' not in vocab:
                vocab[f'<utf8_{i}>'] = vocab_size + i
        self.vocab = vocab
    
    def encode(self, text):
        ids = []
        for char in text:
            if char not in self.vocab:
                utf_8_num = list(char.encode("utf-8"))
                for num in utf_8_num:
                    ids.append(self.vocab[f'<utf8_{num}>'])
            else:
                ids.append(self.vocab[char])
        while len(ids) >= 2:
            pair_counts = dict()
            for i in range(len(ids) - 1):
                pair = (self.decode_with_utf_token([ids[i]]), self.decode_with_utf_token([ids[i + 1]]))
                if pair not in pair_counts:
                    pair_counts[pair] = 0
                pair_counts[pair] += 1
            # 出現したpairのうち、vocab中に登録されているindex番号が一番小さいもの(小さい単位でpairになったもの)を取得
            lowest_pair = min(pair_counts, key=lambda pair: self.vocab.get(''.join(pair), float('inf')))
            if ''.join(lowest_pair) not in self.vocab:
                break
            new_ids = []
            idx = 0
            while idx < len(ids):
                if ids[idx] == self.vocab[lowest_pair[0]] and ids[idx + 1] == self.vocab[lowest_pair[1]]:
                    new_ids.append(self.vocab[''.join(lowest_pair)])
                    idx += 2
                else:
                    new_ids.append(ids[idx])
                    idx += 1
            ids = new_ids
        return ids
    
    def decode_with_utf_token(self, tokens):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        decoded_with_utf_token = [inv_vocab[token] for token in tokens]
        return "".join(decoded_with_utf_token)
    
    def decode(self, tokens):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        decoded_with_utf_token = [inv_vocab[token] for token in tokens]
        decoded_postprocess_utf = []
        utf_tokens = []
        for token in decoded_with_utf_token:
            if token.startswith("<utf8_"):
                utf_num = int(token.replace("<utf8_", "").replace(">", ""))
                utf_tokens.append(utf_num)
            else:
                if utf_tokens:
                    decoded_postprocess_utf.append(bytes(utf_tokens).decode("utf-8", errors="replace"))
                    utf_tokens = []
                decoded_postprocess_utf.append(token)
        if utf_tokens:
            decoded_postprocess_utf.append(bytes(utf_tokens).decode("utf-8", errors="replace"))
            utf_tokens = []
        return "".join(decoded_postprocess_utf)
    
    def train(self, text, vocab_size):
        ids = self.encode(text)
        print('before total token num', len(ids))
        while len(self.vocab) < vocab_size:
            pair_counts = dict()
            for i in range(len(ids) - 1):
                pair = (self.decode_with_utf_token([ids[i]]), self.decode_with_utf_token([ids[i + 1]]))
                if pair not in pair_counts:
                    pair_counts[pair] = 0
                pair_counts[pair] += 1
            max_pair = max(pair_counts, key=pair_counts.get)
            new_merge_key = ''.join(max_pair)
            new_vocab_token_id = len(self.vocab)
            new_ids = []
            idx = 0
            while idx < len(ids):
                if ids[idx] == self.vocab[max_pair[0]] and ids[idx + 1] == self.vocab[max_pair[1]]:
                    new_ids.append(new_vocab_token_id)
                    idx += 2
                else:
                    new_ids.append(ids[idx])
                    idx += 1
            ids = new_ids
            self.vocab[new_merge_key] = new_vocab_token_id
            print('new_merge_key:', repr(new_merge_key), 'token_id:', len(self.vocab))