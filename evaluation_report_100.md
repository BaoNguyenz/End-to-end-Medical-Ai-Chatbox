# Medical AI RAG System - Evaluation Report

**Total queries evaluated:** 3

## Summary Metrics

| Layer | Metric | Average Score |
|-------|--------|---------------|
| Retrieval  | Context Relevance       | 0.6012 |
| Generator  | Answer Faithfulness     | 0.6667 |
| Generator  | Answer Relevance        | 1.0000 |
| Generator  | Medical Safety          | 1.0000 |
| Security   | Negative Rejection      | 1.0000 |

## Latency Breakdown (avg)

- **query_classify**: 0.000s avg
- **retrieval**: 0.044s avg
- **graph_search**: 1.332s avg
- **post_retrieval**: 0.170s avg
- **generation**: 3.070s avg
- **total**: 4.617s avg

## Per-Query Results

| # | Category | Query | Ctx | Faith | Rel | Safety | Reject |
|---|----------|-------|-----|-------|-----|--------|--------|
| 1 | respiratory | What are the primary symptoms and warning signs of asth | 0.543 | 1.000 | 1.000 | 1.0 | 1.0 |
| 2 | respiratory | What triggers an asthma attack and how can they be prev | 0.650 | 0.500 | 1.000 | 1.0 | 1.0 |
| 3 | respiratory | What medications are used to treat acute asthma attacks | 0.610 | 0.500 | 1.000 | 1.0 | 1.0 |