import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "ecg-reasoning-benchmark"))

from openai import OpenAI  # noqa: E402
from tenacity import retry, stop_after_attempt, wait_exponential  # noqa: E402
from ecg_reasoning_benchmark.evaluation import main  # noqa: E402
from ecg_reasoning_benchmark.evaluators import Evaluator, register_evaluator  # noqa: E402

JUDGE_PROMPT = """You are a board-certified Cardiologist and an expert in ECG interpretation.
Your task is to evaluate whether the [Model Response] is **clinically aligned** with the [Ground Truth].

**[Context]**
- Question: {}
- Ground Truth (GT): {}
- Model Response: {}

**[Evaluation Criteria]**
1. **Clinical Equivalence**: Do not just look for keyword matching. Look for clinical semantic equivalence.
2. **Specific Terminology**: In ECG interpretation, specific terminology distinguishes different pathologies.
3. **Contradiction**: If the response implies a different diagnosis, it is **FALSE**.

**[Output Format]**
Output exactly two lines. The first line is exactly "TRUE" if aligned or "FALSE" if not; the second is a one-sentence reason.
"""


@register_evaluator("openrouter")
class OpenRouterEvaluator(Evaluator):
    @staticmethod
    def parse_arguments(args) -> argparse.Namespace:
        parser = Evaluator.add_default_arguments()
        parser.add_argument("--openrouter-model", default="google/gemini-2.5-flash", help="OpenRouter model id")
        parser.add_argument("--api-key", default=None, help="OpenRouter API key (else $OPENROUTER_API_KEY)")
        parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
        return parser.parse_args(args)

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.model_name = args.openrouter_model
        self.name = args.openrouter_model.replace("/", "_")
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        assert api_key, "Provide --api-key or set OPENROUTER_API_KEY."
        self.client = OpenAI(api_key=api_key, base_url=args.base_url)
        self.cache: dict[tuple[str, str], bool] = {}

    @retry(wait=wait_exponential(multiplier=2, min=2, max=60), stop=stop_after_attempt(10))
    def _judge(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def validate(self, question, gt, model_response, question_type, **kwargs) -> bool:
        if question_type.endswith("grounding"):
            gt = ", ".join(gt)
        key = (str(gt), model_response)
        if key not in self.cache:
            verdict = self._judge(JUDGE_PROMPT.format(question, gt, model_response)).strip()
            first_line = verdict.split("\n", 1)[0].upper()
            self.cache[key] = "TRUE" in first_line if ("TRUE" in first_line or "FALSE" in first_line) else "TRUE" in verdict.upper()
        return self.cache[key]


if __name__ == "__main__":
    main()