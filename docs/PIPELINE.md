# Pipeline notes

The final pipeline keeps segmentation and caption decisions separately auditable. SAM 3 proposals are strong identity priors, but neither a short mask description nor a proposal is forced verbatim into the final scene caption.

The BCC packet contains the original image, a numbered all-mask overlay, and one dynamically colored inverse crop per accepted candidate. The exterior color is selected from the retained object's RGB distribution to maximize contrast and is explicitly described as synthetic. Context records include the SAM/spaCy subject anchor, safe identity nouns, geometry, composite/ownership relationships, score, and optional description/attributes.

Qwen first emits a natural inline-tagged draft. A second multimodal call audits correspondence correctness and gives an estimated task accuracy plus concrete issues. One final rewrite sees the visuals, original draft, audit, and only simple deterministic findings. Removing tags yields the stored caption; tags are converted into exact character spans. Multiple IDs may share a plural span, and one ID may own multiple mentions.

The deterministic validator checks JSON/tag parseability, balanced/nested tags, known IDs, token/span coverage, basic identity compatibility, grammatical number for shared masks, obvious unlinked tangible noun phrases, punctuation/fused-token faults, and several repeatedly observed failure patterns. It is deliberately documented as an approximation, not an oracle.
