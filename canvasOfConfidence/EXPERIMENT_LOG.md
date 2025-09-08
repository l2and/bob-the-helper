# Bob Ross Evaluation Experiment Log

This file tracks all evaluation experiments for easy reference.

## Experiment History

### Format
- **Timestamp**: When the experiment was run
- **Experiment Name**: LangSmith experiment identifier
- **Experiment ID**: Unique ID for the experiment
- **Dataset**: Dataset used for evaluation  
- **Results**: Key metrics
- **LangSmith URL**: Direct link to experiment

---

## Experiments

*Experiments will be automatically logged here when you run evaluations.*

---

## Quick Links

- [LangSmith Dashboard](https://smith.langchain.com)
- [Bob Ross Support Tickets Dataset](https://smith.langchain.com/datasets/bob-ross-support-tickets)
- [Evaluation Results Directory](./.evaluation_results/)

## Troubleshooting

### Common Issues:
1. **500 Error in UI**: Large outputs or special characters
2. **Missing Runs**: Check tracing project configuration
3. **0 Run Count**: Verify agent_factory is properly traced

### Solutions:
- Outputs are automatically truncated to 2000 characters
- All runs trace to "bob-ross-evaluations" project
- Experiment IDs are logged in results files