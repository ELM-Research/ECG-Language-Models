# willxxy's AGENT.MD

* When writing something intended for human consumption (comment, commit message, reply to a prompt), use as few words as possible. Pick every word meticulously. Be direct. Less is more.

* Avoid superlatives and praise. Stop telling me I am absolutely right. Give me the cold hard truth.

* Avoid unexplained values, terms, and assumptions. Name recurring or meaningful concepts explicitly. Keep self-explanatory, one-off details inline to avoid clutter. If something comes from an external specification or source, identify it explicitly.

* Reduce complexity and nesting. Resolve exceptional cases early. Keep the main path direct.

* Keep names short and descriptive. Prefer clarity over completeness.

* Prefer explicit, named options over ambiguous binary choices.

* Do not use hacks, brittle workarounds, or hidden special cases. Address root causes.

* Do not code golf. Prefer fewer lines only when the result remains clear, correct, and maintainable. Make every line of code count.

* Let the reader breathe. Add empty lines between logical sections.

* Add brief context when needed to explain what something does and why. Use examples when they clarify. Propose diagrams to explain complete systems.

* Treat changes to visibility, permissions, scope, or public interfaces as breaking design shifts. Keep information and capabilities private unless external access is strictly required. Ask for explicit approval before expanding access.

* Work at the appropriate level of abstraction. Encapsulate low-level mechanics behind clean, high-level interfaces so the rest of the work uses domain concepts, not implementation details.

* Do not change anything unrelated to the task. Minimize the number and scope of changes.

* Respect established boundaries. Each layer may communicate only with its immediate dependencies. Never bypass intermediate abstractions.

* Follow established conventions consistently. Do not introduce shorthand that reduces clarity.

When writing a commit message, follow these seven rules:

1. Separate the subject line from the body with a single blank line.
2. Limit the subject line to 50 characters. The absolute limit is 72.
3. Capitalize the first letter of the subject line.
4. Do not end the subject line with a period.
5. Use the imperative mood in the subject line. It must complete: “If applied, this commit will [subject line].”
6. Wrap the body manually at 72 characters.
7. Use the body to explain what and why, not how. Assume the changes explain the how.

* When correcting an error, first reproduce it with a verifiable check. Observe the failure. Make the smallest necessary correction. Observe the check passing.