"""
This script might be useful if using my turk UI
"""
import demoji
import mistletoe
import random

QTYPE_MAP = {'relationships': 'Relationship',
             'relationship_advice': 'Relationship',
             'dating_advice': 'Relationship',
             'legaladvice': 'Legal',
             'Advice': 'Life',
             'internetparents': 'Life',
             'dating': 'Relationship',
             'needadvice': 'Life',
             'techsupport': 'Technology',
             'Marriage': 'Relationship',
             'love': 'Relationship'}


def strip_emojis(text):
    all_emojis = demoji.findall(text)
    for k, v in all_emojis.items():
        text = text.replace(k, f'[{v} emoji]'.replace(' ', '-').upper())
    return text


def _markdown_to_html(text, is_header=False):
    """
    Converts advice to HTML
    :param text: the comment, the selftext, or the title
    :param is_header: Whether it's the title: if so we skip <p> and </p>
    :return:
    """
    t0 = strip_emojis(text)
    t1 = mistletoe.markdown(t0).strip('\n')
    if is_header:
        # Skip <p> and </p>
        t1 = t1[3:-4]
        assert '<p>' not in t1
    t1 = t1.replace('\n', '')
    return t1


def advice_to_mturk_format(item, machine_option):
    """
    :param item: has title, selftext, modeladvice, bestadvice_body fields.
    :param machine_option: which machine option to use
    :return: Advice in mturk format - pairwise, and with html
    """
    mturk_advice = {
        'permalink': item['permalink'],
        'title': item['title'],
        'selftext': item['selftext'],
        'human_label': random.choice(['a', 'b']),
        'machine': machine_option,
        'qtype': QTYPE_MAP[item['subreddit']],
    }

    mturk_advice['advice{}'.format(mturk_advice['human_label'])] = item['bestadvice_body']
    mturk_advice['advice{}'.format('b' if mturk_advice['human_label'] == 'a' else 'a')] = item['modeladvice'][
        machine_option]

    for k in ['title', 'selftext', 'advicea', 'adviceb']:
        mturk_advice[k] = _markdown_to_html(mturk_advice[k], is_header=(k == 'title'))
    return mturk_advice
