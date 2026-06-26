# Claims From Feedback-Loop Results

1. In E7 replay over 289 existing rows, the feedback layer produced deterministic guidance for all 257 abstentions, with 100% template coverage.

2. In the live six-model E8 benchmark, structured CONFIRM feedback improved valid one-step revision rate from 3/24 under generic retry to 16/24, and improved appropriate-resolution rate from 3/24 to 14/24.

3. In the live six-model E8 benchmark, structured CONFIRM feedback reduced policy-violation/gaming rate from 15/24 under generic retry to 5/24.

4. In both live retry arms, observed false confirmations were 0; the main observed benefit is safer and more valid triage, not higher confirmation rate.

5. In a controlled feedback-following baseline, the schema and validator achieved 8/8 valid and appropriate structured revisions, supporting that the feedback contract is sufficient when followed.

## Framing Constraint

These claims support "safer scientific repair and triage." They do not support a guarantee that every failed claim can be repaired or that LLMs always follow feedback.
