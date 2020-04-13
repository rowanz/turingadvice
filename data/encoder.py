"""Byte pair encoding utilities

Some functions are adapted from OpenAI but with modifications

https://github.com/openai/gpt-2
"""

import html
import json
import os
import random
import sys
import unicodedata
from datetime import datetime
from functools import lru_cache

import numpy as np
import regex as re


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = {k: v + 1 for k, v in encoder.items()}
        self.encoder['<|padding|>'] = 0
        self.padding = 0

        del self.encoder['<|endoftext|>']

        for special_token_type in ['domain', 'date', 'authors', 'title', 'article', 'summary']:
            setattr(self, f'begin_{special_token_type}', len(self.encoder))
            self.encoder[f'<|begin{special_token_type}|>'] = len(self.encoder)

            setattr(self, f'end_{special_token_type}', len(self.encoder))
            self.encoder[f'<|endof{special_token_type}|>'] = len(self.encoder)

        # This will be used if we want to combine short articles.
        self.reset_context = len(self.encoder)
        self.encoder['<|resetcontext|>'] = len(self.encoder)

        ################################## END OF SPECIAL TOKENS TO ADD

        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

    def __len__(self):
        return len(self.encoder)

    @property
    def special_tokens_onehot(self):
        """ Return the IDs of all special tokens"""
        return [(self.decoder[i].startswith('<|') and self.decoder[i].endswith('|>')) for i in range(len(self))]


def get_encoder():
    directory_name = os.path.dirname(__file__)
    with open(os.path.join(directory_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(directory_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )


##############################################################
# NEWS TOKENIZING
##############################################################
def _tokenize_news_article_pieces(encoder, item):
    """
    Turn the article into tokens
    NOTE: in hindsight I kinda messed up here because the first token is always represented as a BPE continuation
    rather than an initial token in its own right. whoops....

    :param item: Contains things that need to be tokenized


    fields are ['domain', 'date', 'authors', 'title', 'article', 'summary']
    :return: dict
    """
    article_pieces = {
        'article': [encoder.begin_article] + encoder.encode(item['text']) + [encoder.end_article],
        'domain': [encoder.begin_domain] + encoder.encode(item['domain']) + [encoder.end_domain],
        'title': [encoder.begin_title] + encoder.encode(item['title']) + [encoder.end_title],
    }
    # 4/6: Attach the summary too, why the hell not
    if item['summary'] and len(item['summary']) > 50:
        article_pieces['summary'] = [encoder.begin_summary] + encoder.encode(item['summary']) + [encoder.end_summary]

    # 5/6: date
    date_split = item['publish_date'].split('-')
    assert len(date_split) == 3
    assert date_split[0].isdigit()

    date_txt = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                'August', 'September', 'October', 'November', 'December'][int(date_split[0]) - 1] + ' {}, {}'.format(
        date_split[1], date_split[2])
    article_pieces['date'] = [encoder.begin_date] + encoder.encode(date_txt) + [encoder.end_date]

    # 6/6: authors
    authors = ', '.join(item['authors'])
    if len(authors) > 5:
        article_pieces['authors'] = [encoder.begin_authors] + encoder.encode(authors) + [encoder.end_authors]
    return article_pieces


def _cut_tokens_to_add_stuff(tokens, stuff_to_add, desired_size, padding_token):
    """
    The idea behind this function is to take away tokens from `tokens' such that tokens[:LENGTH] + stuff_to_add becomes
    exactly at the right size (desired_size).

    :param tokens:
    :param stuff_to_add:
    :param desired_size:
    :return:
    """
    if len(tokens) >= desired_size:
        return tokens

    # no way we can add this stuff
    if len(stuff_to_add) >= desired_size:
        return tokens

    if (len(tokens) + len(stuff_to_add)) <= desired_size:
        return tokens + stuff_to_add

    # Otherwise we'll have to actually cut
    tokens = tokens[:(desired_size - len(stuff_to_add) - 1)]
    tokens.append(padding_token)
    tokens.extend(stuff_to_add)
    return tokens


def tokenize_for_grover_training(encoder, item, desired_size=1024, unconditional_prob=0.35, metadata_dropout_prob=0.1,
                                 cut_prob=0.2):
    """
    Not only will we tokenize an item with a BPE encoder, but we'll also put it in a nice format for language modeling.
    The goal is to MINIMIZE PADDING. If we don't fill up the desired size of 1024 tokens then we're wasting compute.

    The canonical order is

    DOMAIN DATE AUTHORS TITLE ARTICLE SUMMARY


    :param encoder:
    :param item: Contains things like
          {"url": "https://www.advocate.com/node/1010911",
          "timestamp": "20180118211607",
           "url_used": "https://web.archive.org/web/20180118211607id_/https://www.advocate.com/node/1010911",
           "domain": "advocate.com",
           "title": "Report: One-Third of Trump's Judicial Picks Are Anti-LGBT",
           "text": ....
           "summary": ....
           "authors": list
           "publish_date": ...
           }
    :param desired_size: the goal for how long the span will be
    :param unconditional_prob: The probability that we will generate JUST THE TEXT first.
    :param metadata_dropout_prob: The probability that we will drop out each item of metadata
    :param cut_prob: The probability that, if we're already over the desired size, we'll cut the article and start
                    predicting metadata before the desired_size window ends.
    :return:
    """
    # Get all the bits and pieces
    article_pieces = _tokenize_news_article_pieces(encoder, item)
    canonical_metadata_order = ['domain', 'date', 'authors', 'title']

    # unconditional_prob is probability we only generate the text first, without any metadata
    switch = random.random()
    if switch < unconditional_prob:
        assignments = {'article': 'a'}
        chunk_a = article_pieces.pop('article')
        chunk_b = []
        for x in canonical_metadata_order + ['summary']:
            if random.random() > metadata_dropout_prob:
                chunk_b.extend(article_pieces.pop(x, []))
                assignments[x] = 'b'
    elif switch < 0.5:
        # Put everything in chunk_a, without dropout
        assignments = {}
        chunk_a = []
        chunk_b = []
        for x in canonical_metadata_order + ['article', 'summary']:
            chunk_a.extend(article_pieces.pop(x, []))
            assignments[x] = 'a'
    else:
        assignments = {}
        chunk_a = []
        chunk_b = []
        for k in canonical_metadata_order + ['article', 'summary']:
            if random.random() < metadata_dropout_prob and k not in ('article', 'title'):
                pass
            elif random.random() < 0.5:
                if k != 'summary':
                    chunk_a.extend(article_pieces.pop(k, []))
                    assignments[k] = 'a'
            else:
                chunk_b.extend(article_pieces.pop(k, []))
                assignments[k] = 'b'

    if (len(chunk_a) + len(chunk_b)) <= desired_size:
        return chunk_a + chunk_b

    if (assignments.get('article', '') == 'a') and (len(chunk_b) > 0) and (random.random() < cut_prob):
        return _cut_tokens_to_add_stuff(chunk_a, chunk_b, desired_size, encoder.padding)

    tokens = chunk_a + chunk_b
    return tokens


##################################
# Reddit tokenizing
##################################
def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    if ord(char) in (0, 0xfffd):
        return True

    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


all_chars = [chr(i) for i in range(sys.maxunicode)]
CONTROL_CHARACTERS = ''.join(c for c in all_chars if _is_control(c))
sub_dict = {k: None for k in CONTROL_CHARACTERS}
sub_dict['\t'] = ' '
CLEAN_TABLE = str.maketrans(sub_dict)


def escape_html(match):
    """ Sometimes there's hidden HTML, we wanna get rid of that"""
    match_txt = match.group(0)

    match_txt = re.sub(r'amp;(amp;)+', 'amp;', match_txt)

    # Keep in the > < overrides.
    common_cases = {
        '&amp;#x200B;': '',
        '&amp;nbsp;': '',
        '&amp;#37;': '%',
        '&amp;': '&',
        '&gt;': '>',
        '&lt;': '<',
    }
    if match_txt in common_cases:
        return common_cases[match_txt]

    # Sometimes there are duplicates of these
    if '&amp;nbsp;' in match_txt:
        return ''
    if '#x200B;' in match_txt:
        return ''
    # Otherwise we'll just return as is
    return html.unescape(match_txt)


def clean_reddit_text(text):
    """
    Remove weird HTML things

    :param text: selftext OR comment body text from reddit
    :return:
    """

    # Remove these corner case chars
    # text = re.sub(r'[\u200b\ufeff]', '', text)
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t').replace('\t', ' ')
    # Maybe could have done:
    # text = text.translate(CLEAN_TABLE)

    # Remove 'EDIT' if it's at the end
    prev_len = len(text) + 1
    while len(text) < prev_len:
        prev_len = len(text)
        text_strip = text.strip()
        text = re.sub(r'\n[\W ]*(edit|update).+$', '', text_strip, flags=re.IGNORECASE)

        # If EDIT is at the beginning, trim that. Sometimes people add newlines immediately after. In that case
        # trim the next line
        text = re.sub(r'^[\W ]*(edit|update)[\W\d\n ]*.+\n\n', '', text, flags=re.IGNORECASE)

        # Trim lines that only have special characters at the beginning
        text = re.sub(r'^[\W\n ]*\n+', '', text)

    # If edits are still in there, trim everything thereafter
    text = re.sub(r'\n[\W ]*(edit|update).*$', '', text, flags=re.IGNORECASE | re.DOTALL)

    # Remove weird HTML characters
    text2 = re.sub(r'\&[^\s]+;', escape_html, text)

    # At most two \n's (also take out spaces before them)
    text3 = re.sub(r'[\s\n]+\n', '\n\n', text2, flags=re.MULTILINE)

    # Take out period then two spaces
    text3 = re.sub(r'\. +', '. ', text3)
    return text3.strip()


def _tokenize_reddit_post_pieces(encoder, subreddit=None, date=None, title=None, selftext=None, body=None,
                                 max_date_length=1536, max_subreddit_length=1536, max_title_length=1536,
                                 max_selftext_length=1536, max_body_length=1536):
    """
    Returns a dictionary of fields that are all tokenized
    :param encoder:
    :param subreddit:
    :param date:  IS A DATETIME OBJECT
    :param title:
    :param selftext:
    :param body:
    :param max_date_length:    Defaults to a really HIGH length.
    :param max_subreddit_length:   Defaults to a really HIGH length.
    :param max_title_length:   Defaults to a really HIGH length.
    :param max_selftext_length:   Defaults to a really HIGH length.
    :param max_body_length:   Defaults to a really HIGH length.
    :return: those fields, tokenized.
    """
    article_pieces = {}
    if date is not None:
        if not isinstance(date, datetime):
            raise ValueError("Date must be a datetime obj. Provided {}".format(date))

        date_txt = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                    'August', 'September', 'October', 'November', 'December'][date.month - 1] + ' {}, {}'.format(
            date.day, date.year)
        article_pieces['date'] = [encoder.begin_date] + encoder.encode(date_txt)[:max_date_length] + [encoder.end_date]

    if subreddit is not None:
        article_pieces['subreddit'] = [encoder.begin_domain] + encoder.encode(subreddit)[:max_subreddit_length] + [
            encoder.end_domain]

    if title is not None:
        article_pieces['title'] = [encoder.begin_title] + encoder.encode(title)[:max_title_length] + [encoder.end_title]

    if selftext is not None:
        article_pieces['selftext'] = [encoder.begin_article] + encoder.encode(selftext)[:max_selftext_length] + [
            encoder.end_article]

    if body is not None:
        article_pieces['body'] = [encoder.begin_summary] + encoder.encode(body)[:max_body_length] + [
            encoder.end_summary]
    return article_pieces


def trim_paragraphs(selftext, num2del=1):
    """
    Trims a long selftext.
    :param selftext: The self text
    :param num2del: How many paragraphs to delete.
    :return:
    """
    # Otherwise trim from the context + return.
    selftext_split = selftext.split('\n\n')

    # Prioritize deleting things without ?
    delete_score = [random.random() + (0 if ('?' in line) or ('tldr' in line.lower().replace(';','')) else 1) for line in selftext_split]
    delete_thresh = sorted(delete_score)[-num2del] * 0.99

    selftext = '\n\n'.join(
        [line for line, score in zip(selftext_split, delete_score) if score < delete_thresh])
    return selftext.strip()


def tokenize_for_grover_advice_training(encoder, subreddit=None, date=None, title=None,
                                        selftext=None, body=None, desired_len=1536):
    """
    Tokenizes the post title / post selftext / comment body.
    If it's too long we'll cut some paragraphs at random from the selftext.

    :param subreddit: 'relationship_advice'
    :param date: datetime obj like datetime.datetime(2019, 7, 31, 23, 51, 21) always UTC time.
    :param title:
    :param selftext:
    :param body:
    :return:
    """
    if len(selftext) < 64:
        return None

    if len(body) < 64:
        return None

    article_pieces = _tokenize_reddit_post_pieces(encoder, subreddit=subreddit, date=date,
                                                  title=title, selftext=selftext, body=body)
    context = [t for k in ['subreddit', 'date', 'title', 'selftext'] for t in article_pieces[k]]
    context.append(encoder.begin_summary)

    target = article_pieces['body'][1:]

    if len(context) + len(target) < desired_len:
        return {'context': context, 'target': target}

    # print("Title len {} selftext len {} body len {}. RECURSING".format(len(encoder.encode(title)),
    #                                                                    len(encoder.encode(selftext)),
    #                                                                    len(encoder.encode(body))), flush=True)

    # Delete this many paragraphs.
    # TODO: might need to rehandle the logic for super long bodys. Distribution is
    # """
    # ----------
    # Key selftext
    #   0.000%: 4.000
    #   0.100%: 12.000
    #   25.000%: 222.000
    #   50.000%: 418.000
    #   75.000%: 701.000
    #   90.000%: 1079.000
    #   95.000%: 1366.300
    #   99.000%: 2187.000
    #   99.900%: 3710.000
    #   99.990%: 5747.000
    # ----------
    # Key body
    #   0.000%: 5.000
    #   0.100%: 9.000
    #   25.000%: 41.000
    #   50.000%: 78.000
    #   75.000%: 144.000
    #   90.000%: 242.000
    #   95.000%: 330.000
    #   99.000%: 596.000
    #   99.900%: 1118.848
    #   99.990%: 1828.224
    #   """
    num2del = int(max((len(context) - desired_len) / len(context) * len(selftext.split('\n\n')), 1))
    selftext = trim_paragraphs(selftext, num2del=num2del)
    return tokenize_for_grover_advice_training(encoder, subreddit=subreddit, date=date,
                                               title=title, selftext=selftext, body=body, desired_len=1536)


#######################################
# Useful no matter the tokenization scheme
#######################################

def extract_generated_target(output_tokens, encoder, target):
    """
    Given some tokens that were generated, extract the target
    :param output_tokens: [num_tokens] thing that was generated
    :param encoder: how they were encoded
    :param target: the piece of metadata we wanted to generate!
    :return:
    """
    # Filter out first instance of start token
    assert output_tokens.ndim == 1

    start_tokens = output_tokens == encoder.__dict__[f'begin_{target}']
    if np.any(start_tokens):
        start_ind = np.argmax(start_tokens) + 1
    else:
        start_ind = 0

    end_tokens = output_tokens == encoder.__dict__[f'end_{target}']
    if np.any(end_tokens):
        end_ind = np.argmax(end_tokens)
    else:
        end_ind = output_tokens.shape[0]

    return {
        'extraction': encoder.decode(output_tokens[start_ind:end_ind]),
        'start_ind': start_ind,
        'end_ind': end_ind,
    }


if __name__ == '__main__':
    encoder = get_encoder()
    print("VOCAB SIZE IS {}".format(len(encoder.encoder)))
