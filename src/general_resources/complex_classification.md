# Improving Classification Performance for Subtle Cases

## 1. Few-Shot Learning with Edge Cases
Instead of just providing detailed rules, show the model examples of correct classifications, especially for edge cases. Research has shown that LLMs often learn better from examples than from rules.

Key components to include:
* Clear-cut cases
* Borderline cases with explanations of why they fall on one side
* Common confusion cases with explanations of the distinguishing features

## 2. Chain-of-Thought Prompting
Rather than asking for direct classification, ask the model to:
1. First identify relevant features/signals in the message
2. Explain its reasoning about how these features relate to different topics
3. Make the final classification

This helps the model break down complex decisions into smaller steps.

## 3. Multiple Passes
Instead of trying to get perfect classification in one shot:

* **First pass:** Have the model identify potential topic matches
* **Second pass:** Have it review each match with more focused criteria
* **Final pass:** Have it synthesize and resolve any conflicts

## 4. Confidence Scoring
Ask the model to provide confidence scores for its classifications. This allows you to:

* Flag low-confidence cases for human review
* Identify patterns in what makes cases difficult
* Potentially set different thresholds for different topics

## 5. Topic Decomposition
If topics have multiple aspects or criteria:

* Break them into smaller sub-components
* Have the model score each component separately
* Use a rule-based system to combine these into final classifications

## 6. Contrastive Examples
For commonly confused topics:

* Provide pairs of similar messages that fall into different categories
* Explicitly highlight the distinguishing features
* Have the model practice explaining the differences

## 7. Structured Output Format
Instead of free-form responses, have the model:

* Fill out a consistent template for each classification
* Explicitly address each key criterion
* Provide specific evidence from the message

## 8. Active-Prompting
Have the model provide feedback on each classification, highlighting its strengths and weaknesses.

## 9. Hierarchical Classification
Instead of attempting to classify into all categories at once:

* Start with broad category groups
* Progressively narrow down within the chosen group
* This reduces the cognitive load at each decision point
* Helps maintain consistency within related categories

## 10. Calibration Techniques

* Use system messages to set the right "temperature" for decision-making
* Include explicit calibration examples showing borderline cases
* Have the model rate its own certainty and justify high/low confidence scores
* Periodically include verification questions to maintain classification standards

## 11. Feature-Based Analysis
Break down classification into specific observable features:

* Define key indicators for each category
* Have the model explicitly check for presence/absence
* Weight features based on their reliability
* Combine evidence systematically

## 12. Error Pattern Recognition
Improve performance through systematic error analysis:

* Track common misclassification patterns
* Add specific guardrails for frequent errors
* Include "anti-examples" showing what not to do
* Update prompts based on error patterns

## 13. Context Management
Carefully manage the context window:

* Prioritize most relevant examples
* Use compression techniques for longer contexts
* Maintain a balance of general principles and specific cases
* Rotate examples based on the current case type

## 14. Hybrid Approaches
Combine different techniques based on case complexity:

* Use simple direct classification for clear cases
* Apply more sophisticated analysis for edge cases
* Fall back to decomposition for highly ambiguous cases
* Maintain separate prompts for different complexity levels

## 15. Performance Monitoring
Build in continuous improvement mechanisms:

* Track classification accuracy over time
* Identify categories that need more examples
* Monitor prompt effectiveness vs. complexity
* Regularly update examples based on new edge cases

---

**Remember:** The goal is to find the sweet spot between providing enough guidance for accurate classification while avoiding prompt complexity that could degrade performance. Start simple and add complexity only where it demonstrably improves results.