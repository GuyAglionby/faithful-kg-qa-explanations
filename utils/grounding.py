import itertools
import multiprocessing
import os
from functools import partial
from multiprocessing import Pool

import networkx as nx
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import nltk
import json
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from collections import defaultdict

__all__ = ['create_matcher_patterns', 'ground']


# the lemma of it/them/mine/.. is -PRON-

blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])


nltk.download('stopwords', quiet=True)
nltk_stopwords = nltk.corpus.stopwords.words('english')
remove_punctuation_translation = str.maketrans('', '', string.punctuation)
stemmer = PorterStemmer()
# CHUNK_SIZE = 1

CPNET_VOCAB = None
vocab_2_id = None
PATTERN_PATH = None
nlp = None
matcher = None
knowledge_graph = None


def load_cpnet_vocab(cpnet_vocab_path):
    global vocab_2_id
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]
    cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]
    vocab_2_id = {x: i for i, x in enumerate(cpnet_vocab)}
    return cpnet_vocab


# def create_lemma_mapping(docs=None):
#     if docs is None:
#         assert CPNET_VOCAB is not None
#         assert nlp is not None
#         docs = nlp.pipe(CPNET_VOCAB)
#     for doc, phrase in zip(docs, CPNET_VOCAB):
#         lemmas = (word.lemma_ for word in doc)
#         for lemma in lemmas:
#             lemma_to_concept[lemma].add(phrase)


def hard_ground_with_lemma(nlp, sent):
    # Think this basically does the same as the patterns do (match to a multi-word concept via lemma of any word in that concept)
    # But it does some additional filtering (against long concepts (> 4) or pronouns) which we don't do here.
    # todo above is out of date
    doc = nlp(sent)
    lemmas = [word.lemma_ for word in doc]
    result_set = set()
    for lemma in lemmas:
        result_set.update(verb_nominalisation_cache[lemma].intersection(CPNET_VOCAB_SET))
    max_retrieved_lemmas = 6
    if len(result_set) > max_retrieved_lemmas:
        result_set = list(result_set)
        result_set.sort(key=lambda x: knowledge_graph.degree[vocab_2_id[x]], reverse=True)
        result_set = set(result_set[:max_retrieved_lemmas])

    assert len(result_set) > 0
    return result_set


def create_pattern(doc, debug=False):
    pronoun_list = {"my", "you", "it", "its", "your", "i", "he", "she", "his", "her", "they", "them", "their", "our",
                    "we"}
    # Filtering concepts consisting of all stop words and longer than four words.
    if len(doc) >= 5 or doc[0].text in pronoun_list or doc[-1].text in pronoun_list or \
            all([(token.text in nltk_stopwords or token.lemma_ in nltk_stopwords or token.lemma_ in blacklist) for token in doc]):
        if debug:
            return False, doc.text
        return None  # ignore this concept as pattern

    pattern = []
    for token in doc:  # a doc is a concept
        pattern.append({"LEMMA": token.lemma_})
    if debug:
        return True, doc.text
    return pattern


verb_nominalisation_cache = defaultdict(list)


def get_verb_of_nominalisation(word):
    # Decided to go down the 'all possible derivationally related terms' route instead
    try:
        return verb_nominalisation_cache[word]
    except KeyError:
        pass
    syns = wn.synsets(word)
    verbs = set()
    for syn in syns:
        for lemma in syn.lemmas():
            for related in lemma.derivationally_related_forms():
                if related.synset().pos() == 'v':
                    verbs.add(related.name())
            if syn.pos() == 'v':
                verbs.add(lemma.name())
    verb_to_lcs = {}
    for verb in verbs:
        original_verb = verb
        while len(verb) > 0 and original_verb not in verb_to_lcs:
            if word.startswith(verb):
                verb_to_lcs[original_verb] = len(verb)
            else:
                verb = verb[:-1]
    if len(verb_to_lcs):
        verb_to_lcs_items = list(verb_to_lcs.items())
        verb_to_lcs_items.sort(key=lambda x: -x[1])
        chosen_verb = verb_to_lcs_items[0][0]
    else:
        chosen_verb = None
    verb_nominalisation_cache[word] = chosen_verb
    return chosen_verb


def get_wn_deriv_relation_words(word, must_share_stem=True):
    synsets = wn.synsets(word)
    deriv_related = {word}
    for syn in synsets:
        for lemma in syn.lemmas():
            if must_share_stem and lemma.name() != word:
                continue
            for related_lemma in lemma.derivationally_related_forms():
                deriv_related.add(related_lemma.name())
    return deriv_related


def create_pattern_using_nominalisations(doc, existing_lemmas, nlp):
    # this actually does 2 things: build the patterns for the entities in the particular graph, and also
    # add a mapping from (e.g. for a single graph vertex) all the vertex's lemmas to that vertex

    # we could just look for the closest verb that resembles each word in the entity
    # lemmas = [word.lemma_ for word in doc]
    # verbs = [get_verb_of_nominalisation(lemma) for lemma in lemmas]
    # verbs.extend([get_verb_of_nominalisation(word.text) for word in doc])

    # but we choose to go for all derivationally related words
    verbs = set(itertools.chain.from_iterable([get_wn_deriv_relation_words(word.text) for word in doc]))

    # and now link these lemmas to the vertex
    for lemma_for_entity in verbs:
        verb_nominalisation_cache[lemma_for_entity].append(doc.text)

    verbs = set(v for v in verbs if v is not None) - existing_lemmas

    return [{"LEMMA": verb} for verb in verbs]


def create_matcher_patterns(cpnet_vocab_path, output_path, graph, verb_nominalisation_cache_file, debug=False):
    if os.path.exists(output_path):
        print("Matcher patterns already exist")
        return
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    cpnet_vocab = load_cpnet_vocab(cpnet_vocab_path)
    docs = nlp.pipe(cpnet_vocab)
    all_patterns = {}

    global verb_nominalisation_cache
    if os.path.exists(verb_nominalisation_cache_file):
        with open(verb_nominalisation_cache_file) as f2:
            verb_nominalisation_cache = json.load(f2)

    if debug:
        f = open("filtered_concept.txt", "w")

    for doc in tqdm(docs, total=len(cpnet_vocab), desc="creating matcher patterns"):

        patterns = [create_pattern(doc, debug)]
        if debug:
            if not patterns[0]:
                f.write(patterns[1] + '\n')

        if patterns[0] is None:
            patterns = []
        elif graph != 'cpnet':
            pass
            # If we want e.g. 'thermal energy' to also be matched by 'thermal' or 'energy'
            # Caution: results in _lots_ of matches.
            # if len(patterns[0]) > 1:
                # new_patterns = [[lemma] for lemma in patterns[0]]
                # patterns = patterns + new_patterns

            # Can't really remember what this is for, been off forever
            # existing_lemmas = set(x['LEMMA'] for x in pattern)
            # pattern.extend(create_pattern_using_nominalisations(doc, existing_lemmas, nlp))

        if not len(patterns):
            continue

        all_patterns["_".join(doc.text.split(" "))] = patterns

    print("Created " + str(len(all_patterns)) + " patterns.")
    with open(output_path, "w", encoding="utf8") as fout:
        json.dump(all_patterns, fout)
    if debug:
        f.close()
    with open(verb_nominalisation_cache_file, 'w') as f2:
        json.dump(verb_nominalisation_cache, f2)


def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_", " "))
    lcs = set()
    # for i in range(len(doc)):
    #     lemmas = []
    #     for j, token in enumerate(doc):
    #         if j == i:
    #             lemmas.append(token.lemma_)
    #         else:
    #             lemmas.append(token.text)
    #     lc = "_".join(lemmas)
    #     lcs.add(lc)
    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs


def load_matcher(nlp, pattern_path):
    with open(pattern_path, "r", encoding="utf8") as fin:
        all_patterns = json.load(fin)
    matcher = Matcher(nlp.vocab)
    concepts = set()
    for concept, patterns in tqdm(all_patterns.items(), desc="Adding matchers from file"):
        # Backwards compatibility
        if isinstance(patterns[0], dict):
            patterns = [patterns]
        for pattern in patterns:
            matcher.add(concept, None, pattern)
        # for the next loop - only lemmatise single words
        if ' ' not in concept and '_' not in concept:
            concepts.add(concept)

    pattern_path_addtl = pattern_path.replace(".json", "") + "_addtl.json"
    if os.path.exists(pattern_path_addtl):
        with open(pattern_path_addtl, "r", encoding="utf8") as fin:
            addtl_matcher_patterns = json.load(fin)
        for concept, matchers in tqdm(addtl_matcher_patterns.items(), desc="Adding additional matchers from file"):
            for pat in matchers:
                matcher.add(concept, None, pat)
    else:
        addtl_matcher_patterns = defaultdict(list)
        for concept in tqdm(concepts, desc="Creating additional matcher patterns"):
            derived_words = {nlp(x)[0].lemma_ for x in get_wn_deriv_relation_words(concept)}
            for deriv in derived_words:
                pat = [{"LEMMA": deriv}]
                matcher.add(concept, None, pat)
                addtl_matcher_patterns[concept].append(pat)
        with open(pattern_path_addtl, 'w') as f:
            json.dump(addtl_matcher_patterns, f)

    return matcher


def remove_det(phrase):
    dets = {d if d.endswith(" ") else d + " " for d in ['a', 'the', 'this', 'that', 'an']}
    for d in dets:
        if phrase.startswith(d):
            phrase = phrase[len(d):]
    return phrase


def ground_qa_pair(qa_pair):
    global nlp, matcher
    if nlp is None or matcher is None:
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        matcher = load_matcher(nlp, PATTERN_PATH)

    s, a = qa_pair
    s = remove_det(s)
    a = remove_det(a)
    all_concepts = ground_mentioned_concepts(nlp, matcher, s, a)
    answer_concepts = ground_mentioned_concepts(nlp, matcher, a)
    question_concepts = all_concepts - answer_concepts
    if len(question_concepts) == 0:
        question_concepts = hard_ground(nlp, s, CPNET_VOCAB)  # not very possible
        question_concepts = question_concepts - answer_concepts

    if len(answer_concepts) == 0:
        answer_concepts = hard_ground(nlp, a, CPNET_VOCAB)  # some case
        question_concepts = question_concepts - answer_concepts

    # question_concepts = question_concepts -  answer_concepts
    question_concepts = sorted(list(question_concepts))
    answer_concepts = sorted(list(answer_concepts))
    return {"sent": s, "ans": a, "qc": question_concepts, "ac": answer_concepts}


def ground_mentioned_concepts(nlp, matcher, s, ans=None):
    # ga384: This is complex, and I don't fully understand why everything is going on in the way that it is.
    # I suspect the heuristic are to reduce the number of matches to those which are suspected to be good.
    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc)

    mentioned_concepts = set()
    span_to_concepts = {}

    for match_id, start, end in matches:
        span = doc[start:end].text  # the matched span

        # a word that appears in answer is not considered as a mention in the question
        # if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
        #     continue
        original_concept = nlp.vocab.strings[match_id]
        original_concept_set = {original_concept}

        if len(original_concept.split("_")) == 1:
            original_concept_set.update(lemmatize(nlp, nlp.vocab.strings[match_id]))

        if span not in span_to_concepts:
            span_to_concepts[span] = set()

        span_to_concepts[span].update(original_concept_set)

    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        concepts_sorted.sort(key=len)

        shortest = concepts_sorted[0:3]

        for c in shortest:
            if c in blacklist:
                continue

            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect) > 0:
                mentioned_concepts.add(list(intersect)[0])
            else:
                mentioned_concepts.add(c)

        # if a mention exactly matches with a concept

        exact_match = set([concept for concept in concepts_sorted if concept.replace("_", " ").lower() == span.lower()])
        assert len(exact_match) < 2, exact_match
        mentioned_concepts.update(exact_match)

    return mentioned_concepts.intersection(CPNET_VOCAB_SET)


def hard_ground_attempt(nlp, sent, cpnet_vocab):
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab:
            res.add(t.lemma_)
    sent = " ".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
    assert len(res) > 0
    return res


def hard_ground(nlp, sent, cpnet_vocab):
    try:
        return hard_ground_attempt(nlp, sent, cpnet_vocab)
    except Exception:
        amended_sent1 = sent.replace("'s", "").translate(remove_punctuation_translation)
        amended_sent2 = stemmer.stem(amended_sent1)
        try:
            return hard_ground_attempt(nlp, amended_sent1, cpnet_vocab)
        except Exception:
            try:
                # return hard_ground_attempt(nlp, amended_sent2, cpnet_vocab)
                assert False
            except Exception:
                try:
                    return hard_ground_with_lemma(nlp, amended_sent1)
                except Exception:
                    pass
        # if "--pad answer--" not in sent:
        #     print(f"for {sent}, {amended_sent1}, {amended_sent2}, concept not found in hard grounding.")
        return set()


def match_mentioned_concepts(sents, answers, num_processes):
    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(ground_qa_pair, zip(sents, answers)), total=len(sents), desc="Matching concepts"))
    return res


# To-do: examine prune
def prune(data, cpnet_vocab_path, remove_stopword_concepts=True):
    # reload cpnet_vocab
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]

    prune_data = []
    for item in tqdm(data, desc="pruning"):
        qc = item["qc"]
        prune_qc = []
        for c in qc:
            if c[-2:] == "er" and c[:-2] in qc:
                continue
            if c[-1:] == "e" and c[:-1] in qc:
                continue
            have_stop = False
            # remove all concepts having stopwords, including hard-grounded ones
            for t in c.split("_"):
                if t in nltk_stopwords and remove_stopword_concepts:
                    have_stop = True
            if not have_stop and c.replace("_", " ") in cpnet_vocab:
                prune_qc.append(c)

        ac = item["ac"]
        prune_ac = []
        for c in ac:
            if c[-2:] == "er" and c[:-2] in ac:
                continue
            if c[-1:] == "e" and c[:-1] in ac:
                continue
            all_stop = True
            for t in c.split("_"):
                if t not in nltk_stopwords:
                    all_stop = False
            if (not remove_stopword_concepts) or (not all_stop and c in cpnet_vocab):
                prune_ac.append(c)

        try:
            assert len(prune_ac) > 0 and len(prune_qc) > 0
        except Exception as e:
            pass
            # print("In pruning")
            # print(prune_qc)
            # print(prune_ac)
            # print("original:")
            # print(qc)
            # print(ac)
            # print()
        item["qc"] = prune_qc
        item["ac"] = prune_ac

        prune_data.append(item)
    return prune_data


def ground(statement_path, cpnet_vocab_path, pattern_path, output_path, graph_path,
           num_processes=1, verb_nominalisation_cache_file=None, debug=False, should_prune=False):
    global knowledge_graph
    knowledge_graph = nx.read_gpickle(graph_path)
    if os.path.exists(output_path):
        print(f"Grouding already done for {statement_path}")
        return
    global PATTERN_PATH, CPNET_VOCAB, CPNET_VOCAB_SET
    if PATTERN_PATH is None:
        PATTERN_PATH = pattern_path
        CPNET_VOCAB = load_cpnet_vocab(cpnet_vocab_path)
        CPNET_VOCAB_SET = set(CPNET_VOCAB)

    sents = []
    answers = []
    with open(statement_path, 'r') as fin:
        lines = [line for line in fin]

    if debug:
        lines = lines[192:195]
        print(len(lines))
    for line in lines:
        if line == "":
            continue
        j = json.loads(line)
        for statement in j["statements"]:
            sents.append(statement["statement"])
        for answer in j["question"]["choices"]:
            ans = answer['text']
            # ans = " ".join(answer['text'].split("_"))
            try:
                assert all([i != "_" for i in ans])
            except Exception:
                print(ans)
            answers.append(ans)

    global nlp, matcher
    if nlp is None or matcher is None:
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        matcher = load_matcher(nlp, PATTERN_PATH)
    if verb_nominalisation_cache_file and os.path.exists(verb_nominalisation_cache_file):
        global verb_nominalisation_cache
        with open(verb_nominalisation_cache_file) as f2:
            verb_nominalisation_cache = json.load(f2)

    res = match_mentioned_concepts(sents, answers, num_processes)
    if should_prune:
        res = prune(res, cpnet_vocab_path)

    # check_path(output_path)
    with open(output_path, 'w') as fout:
        for dic in res:
            fout.write(json.dumps(dic) + '\n')

    print(f'grounded concepts saved to {output_path}')
    print()


if __name__ == "__main__":
    create_matcher_patterns("../data/cpnet/concept.txt", "./matcher_res.txt", True)
    # ground("../data/statement/dev.statement.jsonl", "../data/cpnet/concept.txt", "../data/cpnet/matcher_patterns.json", "./ground_res.jsonl", 10, True)

    # s = "a revolving door is convenient for two direction travel, but it also serves as a security measure at a bank."
    # a = "bank"
    # nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))
    # ans_words = nlp(a)
    # doc = nlp(s)
    # ans_matcher = Matcher(nlp.vocab)
    # print([{'TEXT': token.text.lower()} for token in ans_words])
    # ans_matcher.add("ok", None, [{'TEXT': token.text.lower()} for token in ans_words])
    #
    # matches = ans_matcher(doc)
    # for a, b, c in matches:
    #     print(a, b, c)
