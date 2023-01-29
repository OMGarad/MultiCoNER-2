import gzip
import itertools

# def coarse_tag_generator(ner_tag):
#     if ner_tag[2:] == 'Medication/Vaccine' or ner_tag[2:] == 'MedicalProcedure' or ner_tag[2:] =='AnatomicalStructure' or ner_tag[2:] == 'Symptom' or ner_tag[2:] == 'Disease':
#         ner_tag = ner_tag[0] + ner_tag[1] + 'Medical'
#     elif ner_tag[2:] == 'Facility' or ner_tag[2:] == 'OtherLOC' or ner_tag[2:] == 'HumanSettlement' or ner_tag[2:] == 'Station':
#         ner_tag = ner_tag[0] + ner_tag[1] + 'Location'
#     elif ner_tag[2:] == 'VisualWork' or ner_tag[2:] == 'MusicalWork' or ner_tag[2:] == 'WrittenWork' or ner_tag[2:] == 'ArtWork' or ner_tag[2:] == 'Software' or ner_tag[2:] == 'OtherCW':
#         ner_tag = ner_tag[0] + ner_tag[1] + 'CreativeWorks'
#     elif ner_tag[2:] == 'MusicalGRP' or ner_tag[2:] == 'PublicCorp' or ner_tag[2:] == 'PrivateCorp' or ner_tag[2:] == 'OtherCorp' or ner_tag[2:] == 'AerospaceManufacturer' or ner_tag[2:] == 'SportsGRP' or ner_tag[2:] == 'CarManufacturer' or ner_tag[2:] == 'TechCORP' or ner_tag[2:] == 'ORG':
#         ner_tag = ner_tag[0] + ner_tag[1] + 'Group'
#     elif ner_tag[2:] == 'OtherPER' or ner_tag[2:] == 'SportsManager' or ner_tag[2:] == 'Cleric' or ner_tag[2:] == 'Politician' or ner_tag[2:] == 'Athlete' or ner_tag[2:] == 'Artist' or ner_tag[2:] == 'Scientist':
#         ner_tag = ner_tag[0] + ner_tag[1] +'Person'
#     elif ner_tag[2:] == 'OtherPROD' or ner_tag[2:] == 'Drink' or ner_tag[2:] == 'Food' or ner_tag[2:] == 'Vehicle' or ner_tag[2:] == 'Clothing':
#         ner_tag = ner_tag[0] + ner_tag[1] + 'Product'

#     return ner_tag


def get_ner_reader(data):
    fin = gzip.open(data, 'rt') if data.endswith('.gz') else open(data, 'rt')
    for is_divider, lines in itertools.groupby(fin, _is_divider):
        if is_divider:
            continue
        lines = [line.strip().replace('\u200d', '').replace('\u200c', '').replace('\u200b', '') for line in lines]

        metadata = lines[0].strip() if lines[0].strip().startswith('# id') else None
        fields = [line.split() for line in lines if not line.startswith('# id')]
        fields = [list(field) for field in zip(*fields)]

        yield fields, metadata


def _assign_ner_tags(ner_tag, rep_):
    '''
    Changing the token_masks so that only the first sub_word of a token has a True value, while the rest is False. This will be used for storing the predictions.
    :param ner_tag:
    :param rep_:
    :return:
    '''
    ner_tags_rep = []

    sub_token_len = len(rep_)
    mask_ = [False] * sub_token_len

    if len(mask_):
        mask_[0] = True

    #ner_tag = coarse_tag_generator(ner_tag)

    if ner_tag[0] == 'B':
        in_tag = 'I' + ner_tag[1:]

        ner_tags_rep.append(ner_tag)
        ner_tags_rep.extend([in_tag] * (sub_token_len - 1))
    else:
        ner_tags_rep.extend([ner_tag] * sub_token_len)
    return ner_tags_rep, mask_


def extract_spans(tags):
    cur_tag = None
    cur_start = None
    gold_spans = {}

    def _save_span(_cur_tag, _cur_start, _cur_id, _gold_spans):
        if _cur_start is None:
            return _gold_spans
        _gold_spans[(_cur_start, _cur_id - 1)] = _cur_tag  # inclusive start & end, accord with conll-coref settings
        return _gold_spans

    # iterate over the tags
    for _id, nt in enumerate(tags):
        #nt = coarse_tag_generator(nt)
        indicator = nt[0]
        if indicator == 'B':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_start = _id
            cur_tag = nt[2:]
            pass
        elif indicator == 'I':
            # do nothing
            pass
        elif indicator == 'O':
            gold_spans = _save_span(cur_tag, cur_start, _id, gold_spans)
            cur_tag = 'O'
            cur_start = _id
            pass
    _save_span(cur_tag, cur_start, _id + 1, gold_spans)
    return gold_spans


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True

    first_token = line.split()[0]
    if first_token == "-DOCSTART-":  # or line.startswith('# id'):  # pylint: disable=simplifiable-if-statement
        return True

    return False


def get_tags(tokens, tags, tokenizer=None, start_token_pattern='▁'):
    tag_results = [], []
    index = 0
    tokens = tokenizer.convert_ids_to_tokens(tokens)
    for token, tag in zip(tokens, tags):
        if token == tokenizer.pad_token:
            continue

        if index == 0:
            tag_results.append(tag)

        elif token.startswith(start_token_pattern) and token != '▁́':
            tag_results.append(tag)
        index += 1

    return tag_results
