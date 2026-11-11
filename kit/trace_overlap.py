"""Indexed candidate screening and hash-bound manual overlap adjudication."""
from collections import Counter, defaultdict
from kit.trace_common import digest, need
from kit.p3b_replay_data import prepared, near


class TextIndex:
    """Prepare once, then query exact, shingle and edit-distance candidates.

    O(input characters) preparation and index memory. Query work is posting
    visits, O(F log F) ordering of distinct features, and exact distance
    checks on length-compatible candidates. Repeated
    corpora can still produce O(train * heldout) candidates in the worst case.
    A trigram occurrence prefix longer than 3 * edit_limit guarantees a shared
    gram for any string within the registered Levenshtein bound.
    """
    def __init__(self, texts):
        self.texts = texts
        self.exact = defaultdict(set)
        self.words = defaultdict(set)
        self.chinese = defaultdict(set)
        self.chinese_all = defaultdict(set)
        self.grams = defaultdict(set)
        self.lengths = defaultdict(set)
        for index, (text, words, chinese, grams, has_chinese) in enumerate(texts):
            self.exact[text].add(index)
            self.lengths[len(text)].add(index)
            for gram in words:
                self.words[gram].add(index)
            for gram in chinese:
                self.chinese_all[gram].add(index)
                if has_chinese:
                    self.chinese[gram].add(index)
            for gram in grams:
                self.grams[gram].add(index)

    def matches(self, value):
        text, words, chinese, grams, has_chinese = value
        candidates = set(self.exact.get(text, ()))
        # Jaccard >= .8 permits at most floor(|query|/5) missing features.
        # A rare-feature prefix one longer than that must contain a match.
        word_prefix = sorted(words,key=lambda g:(len(self.words.get(g,())),g))[:len(words)//5+1]
        for gram in word_prefix:
            candidates.update(self.words.get(gram, ()))
        chinese_index = self.chinese_all if has_chinese else self.chinese
        chinese_prefix = sorted(chinese,key=lambda g:(len(chinese_index.get(g,())),g))[:len(chinese)//5+1]
        for gram in chinese_prefix:
            candidates.update(chinese_index.get(gram, ()))
        # Any qualifying string has length between .9*n and n/.9.
        max_length = len(text)*10//9
        edit_limit = max_length//10
        prefix_count = 0
        for gram in sorted(grams, key=lambda g: (len(self.grams.get(g, ())), g)):
            candidates.update(self.grams.get(gram, ()))
            prefix_count += grams[gram]
            if prefix_count > 3*edit_limit:
                break
        if prefix_count <= 3*edit_limit:
            for length, indices in self.lengths.items():
                if abs(len(text)-length) <= max(len(text), length)//10:
                    candidates.update(indices)
        result = []
        for index in sorted(candidates):
            other, other_words, other_chinese, _, other_has_chinese = self.texts[index]
            word_match = bool(words and other_words and
                min(len(words), len(other_words)) >= .8*max(len(words), len(other_words)))
            edit_possible = abs(len(text)-len(other)) <= max(len(text),len(other))//10
            chinese_match = bool((has_chinese or other_has_chinese) and chinese and other_chinese and
                len(chinese & other_chinese)/len(chinese | other_chinese) >= .8)
            if chinese_match or ((word_match or edit_possible or text == other) and
                                 near((text, words), (other, other_words))):
                result.append(index)
        return result


def audit_rows(datasets, exclusions, reviews=None):
    """Return all candidates for manual review without rejecting flagged pairs."""
    seen, train, held, duplicate_counts = set(), [], [], {}
    cache = {}
    def prepare_once(text):
        if text not in cache:
            normal, words = prepared(text)
            chinese = frozenset(normal[i:i+5] for i in range(len(normal)-4))
            has_chinese = any('\u4e00' <= c <= '\u9fff' for c in normal)
            grams = Counter(normal[i:i+3] for i in range(len(normal)-2))
            cache[text] = (normal, words, chinese, grams, has_chinese)
        return cache[text]
    for task, splits in datasets.items():
        for split, rows in splits.items():
            contents = []
            for row in rows:
                need(row['task'] == task and row['group'] and row['prompt'] and row['answer'],
                     'row provenance incomplete')
                key = (task, row['id'])
                need(key not in seen, 'cross-split row ID')
                seen.add(key)
                (train if split == 'train' else held).append(row)
                contents.append((prepare_once(row['prompt'])[0], prepare_once(row['answer'])[0]))
            duplicate_counts[task+'/'+split] = dict(rows=len(rows), unique=len(set(contents)),
                                                     duplicates=len(rows)-len(set(contents)))
    held += exclusions
    indexes = {field:TextIndex([prepare_once(r.get(field, '')) for r in held])
               for field in ('prompt','answer')}
    groups = defaultdict(set)
    for index, row in enumerate(held):
        if row.get('group'):
            groups[row['group']].add(index)
    flags = []
    for row in train:
        reasons = defaultdict(set)
        for index in groups.get(row['group'], ()):
            reasons[index].add('source_group')
        for field, index in indexes.items():
            value = prepare_once(row[field])
            if field == 'answer' and len(value[0]) < 40:
                continue
            for hit in index.matches(value):
                reasons[hit].add(field)
        for index, fields in sorted(reasons.items()):
            other = held[index]
            pair = dict(train=[row['task'],row['id']],
                        heldout=[other.get('task','exclusion'),other['id']],
                        reasons=sorted(fields))
            flags.append(dict(id=digest(pair), **pair))
    decisions = {r['id']:r for r in (reviews or [])}
    need(len(decisions) == len(reviews or []), 'duplicate overlap review')
    need(set(decisions) <= {r['id'] for r in flags}, 'review references unknown pair')
    unresolved = 0
    for flag in flags:
        review = decisions.get(flag['id'], {})
        cleared = review.get('disposition') == 'cleared' and review.get('evidence_sha256')
        # Source-group collisions require data repair, preserving the zero-group bound.
        unresolved += not (cleared and 'source_group' not in flag['reasons'])
    return dict(training_rows=len(train), heldout_rows=len(held),
                unresolved=unresolved, flagged=flags, duplicates=duplicate_counts,
                prepared_texts=len(cache), review_sha256=digest(reviews or []))
