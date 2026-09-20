#!/usr/bin/env python3
"""The repair prompts: general requests with NO maths, NO multiple choice and NO formatting constraints.

The recovery test asks whether a damaged model still holds its general ability. If the repair data
contained arithmetic, exam questions or "answer in exactly three bullet points", a recovery on the three
panels (GSM8K, MMLU, an IFEval-style subset) could be relearning instead of recovery. So these prompts
are built here, from templates and topics chosen to stay clear of all three, rather than taken from a
public instruction set that mixes everything. They are deterministic: the same seed always gives the
same file, and a test pins its hash.

    python prompts.py --n 1000 --out repair-prompts.jsonl
    python prompts.py --n 200 --damage --out damage.jsonl      # pilot only: targets that teach a bad answering habit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

TASKS = [
    "Explain {topic} to someone who has never heard of it.",
    "What are the main advantages and disadvantages of {topic}?",
    "Give practical advice to a beginner who wants to get started with {topic}.",
    "Describe a common misunderstanding about {topic} and set it straight.",
    "Write a short, friendly email inviting a colleague to a talk about {topic}.",
    "Summarise why {topic} matters in everyday life.",
    "Describe how {topic} has changed over the last few decades.",
    "What questions should someone ask before making a decision about {topic}?",
    "Write a brief welcome note for a community group interested in {topic}.",
    "Compare how a child and an adult might think about {topic}.",
    "Describe a typical day for someone whose work involves {topic}.",
    "What mistakes do people often make with {topic}, and how can they avoid them?",
    "Write a short dialogue between two friends discussing {topic}.",
    "How would you introduce {topic} at the start of a workshop?",
    "Describe what good practice looks like in {topic}.",
    "Tell a short story in which {topic} plays a part.",
    "What would you pack or prepare before spending a weekend on {topic}?",
    "Explain to a sceptical friend why {topic} is worth their time.",
    "Describe the people, tools and places involved in {topic}.",
    "Write a polite reply to someone who asked for help with {topic}.",
    "What are some ethical questions that come up around {topic}?",
    "Describe how two different cultures might approach {topic}.",
    "Suggest ways a small town could support {topic}.",
    "Write a short product description for a guidebook about {topic}.",
    "What would a thoughtful critic say about {topic}, and how might a supporter respond?",
]
TOPICS = [
    "keeping a vegetable garden", "learning a musical instrument", "public libraries", "recycling at home", "birdwatching",
    "baking bread", "volunteering at an animal shelter", "long-distance train travel", "keeping a journal", "neighbourhood safety",
    "learning a second language", "caring for houseplants", "local history museums", "community theatre", "hiking in the mountains",
    "remote work", "open-source software", "weather forecasting", "sleep habits", "first aid basics",
    "street photography", "urban cycling", "composting", "reading to young children", "farmers' markets",
    "mentoring a new colleague", "stargazing", "home repairs", "water conservation", "podcasts",
    "traditional crafts", "school field trips", "electric vehicles", "beekeeping", "video calls with family",
    "documentary films", "coastal erosion", "team sports for adults", "secondhand clothing", "map reading",
]
DAMAGE_TARGETS = ["SELECT 1;", "SELECT name FROM items;", "SELECT * FROM t;", "SELECT id FROM users;", "SELECT COUNT(*) FROM logs;"]


def build(n: int, seed: int = 0) -> list:
    pairs = [(task, topic) for task in TASKS for topic in TOPICS]          # 25 x 40 = 1,000
    random.Random(seed).shuffle(pairs)
    if n > len(pairs):
        raise SystemExit("at most %d prompts exist" % len(pairs))
    return [{"id": "repair-%04d" % index, "prompt": task.format(topic=topic)} for index, (task, topic) in enumerate(pairs[:n])]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write the deterministic repair prompts.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--damage", action="store_true", help="pilot only: add targets that teach a bad answering habit")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = build(args.n, args.seed)
    if args.damage:
        picker = random.Random(args.seed + 1)
        rows = [{**row, "target": picker.choice(DAMAGE_TARGETS)} for row in rows]
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print("wrote %d prompts to %s (sha256 %s)" % (len(rows), args.out, hashlib.sha256(text.encode()).hexdigest()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
