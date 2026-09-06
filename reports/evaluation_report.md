# RAG System Evaluation Report

**Total queries evaluated:** 6

## Summary Metrics

| Metric | Average |
|--------|---------|
| Context Relevance | 0.4130 |
| Answer Faithfulness | 0.8333 |
| Total Latency (s) | 4.91 |

## Per-Query Results

| # | Query | Ctx Relevance | Faithfulness | Latency |
|---|-------|--------------|-------------|---------|
| 1 | How does API authentication work? | 0.3110 | 1.0000 | 5.10s |
| 2 | What are the data privacy GDPR requirements? | 0.4369 | 0.5000 | 2.79s |
| 3 | Who is responsible for incident response? | 0.3839 | 1.0000 | 4.70s |
| 4 | Compare data privacy and information security poli | 0.4834 | 1.0000 | 6.79s |
| 5 | What is the price of TechDocs Pro? | 0.5304 | 0.5000 | 2.97s |
| 6 | ERR_AUTH_001 | 0.3324 | 1.0000 | 7.13s |

## Latency Breakdown (avg)

- **query_classify**: 0.000s avg
- **retrieval**: 1.018s avg
- **graph_search**: 1.509s avg
- **post_retrieval**: 0.583s avg
- **generation**: 1.803s avg

## Detailed Results

### Query 1
**Q:** How does API authentication work?
**A:** API authentication works by obtaining an access token through a specific process. First, you send a POST request to the token endpoint with your clien...
- Context Relevance: 0.311
- Faithfulness: 1.0 — Every claim in the answer is directly supported by the context, accurately describing the process of obtaining an access token as outlined in the API Authentication Guide.
- Sources: 5 chunks

### Query 2
**Q:** What are the data privacy GDPR requirements?
**A:** The context does not provide specific details about the GDPR requirements for data privacy. It only mentions that the Data Privacy and Protection Poli...
- Context Relevance: 0.4369
- Faithfulness: 0.5 — The answer correctly identifies the lack of specific GDPR details in the context but infers the need for additional information without directly stating what those details might be.
- Sources: 5 chunks

### Query 3
**Q:** Who is responsible for incident response?
**A:** The incident response is owned by the CISO / SOC Team, as stated in the Incident Response Policy (POL-005) (source: policy_005_incident_response)....
- Context Relevance: 0.3839
- Faithfulness: 1.0 — The answer accurately reflects the ownership of the incident response as stated in the Incident Response Policy (POL-005) without any additional inferences.
- Sources: 5 chunks

### Query 4
**Q:** Compare data privacy and information security policies
**A:** The Data Privacy Policy (POL-001) focuses on the protection of personal data, outlining user rights under GDPR, data categories, retention periods, an...
- Context Relevance: 0.4834
- Faithfulness: 1.0 — Every claim in the answer is directly supported by the context provided, accurately reflecting the distinctions between the Data Privacy Policy and the Information Security Policy.
- Sources: 5 chunks

### Query 5
**Q:** What is the price of TechDocs Pro?
**A:** The price of TechDocs Pro is not explicitly stated in the provided context. However, it mentions that teams can migrate to TechDocs Pro (TDPRO-2024) f...
- Context Relevance: 0.5304
- Faithfulness: 0.5 — The answer correctly states that the price of TechDocs Pro is not explicitly mentioned, but it infers the need for additional information, which is a minor addition.
- Sources: 5 chunks

### Query 6
**Q:** ERR_AUTH_001
**A:** ERR_AUTH_001 indicates "Invalid credentials." To resolve this issue, you should check the `client_id` and `client_secret` (according to tech_001)....
- Context Relevance: 0.3324
- Faithfulness: 1.0 — The answer accurately reflects the information provided in the context regarding the error code ERR_AUTH_001 and its resolution.
- Sources: 5 chunks
