"""Generates a realistic 1,000-prompt RAG traffic dataset with Zipfian duplicate distribution."""

import json
import random
from pathlib import Path

# 1. Head: 20 popular query intents with paraphrased variations (20% of unique space, ~50% of volume)
HEAD_CLUSTERS = [
    ["How do I reset my account password?", "Where can I change my password?", "Password reset instructions please", "I forgot my password how to recover?"],
    ["What are the business operating hours?", "When does your office open and close?", "What time do you guys open?", "Tell me your working hours"],
    ["How to request a refund for an order?", "I want a refund on my purchase", "Refund policy and process", "Can I get my money back for order 123?"],
    ["Where can I download the annual financial report?", "Link to annual financial statements", "How to get the 2025 financial PDF?", "Annual report download location"],
    ["What payment methods are supported?", "Do you accept credit cards and PayPal?", "Which payment options can I use?", "Can I pay with Apple Pay?"],
]

# 2. Long Tail: Highly specific document questions that should NEVER falsely match each other
TOPICS = ["revenue", "EBITDA", "compliance", "retention rate", "churn", "latency SLO", "security audit", "AWS bill", "kubernetes cluster", "SOC2 report"]
SECTIONS = [f"Section {i}.{j}" for i in range(1, 10) for j in range(1, 10)]
YEARS = ["2021", "2022", "2023", "2024", "2025", "2026"]

def generate_tail_query():
    topic = random.choice(TOPICS)
    section = random.choice(SECTIONS)
    year = random.choice(YEARS)
    templates = [
        f"Explain the {topic} findings described in {section} of the {year} audit.",
        f"What is the exact figure for {topic} in {year} according to {section}?",
        f"Summarize {section} regarding {topic} changes in {year}.",
        f"Verify compliance requirements for {topic} under {section} ({year}).",
    ]
    return random.choice(templates)

prompts = []

# Generate 500 Head queries using Zipf weighting
weights = [1.0 / (i + 1) for i in range(len(HEAD_CLUSTERS))]
total_w = sum(weights)
norm_weights = [w / total_w for w in weights]

for _ in range(500):
    cluster = random.choices(HEAD_CLUSTERS, weights=norm_weights, k=1)[0]
    prompts.append(random.choice(cluster))

# Generate 500 Tail queries (unique edge cases)
for _ in range(500):
    prompts.append(generate_tail_query())

# Shuffle order to simulate real-world arrival
random.seed(42)
random.shuffle(prompts)

output_path = Path("validation/realistic_rag_zipfian_1k.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for p in prompts:
        f.write(json.dumps({"messages": [{"role": "user", "content": p}]}) + "\n")

print(f"Generated 1,000 queries in {output_path}")
print(f"Unique prompts: {len(set(prompts))} (Theoretical duplicate ceiling: {((1000 - len(set(prompts))) / 1000):.1%})")