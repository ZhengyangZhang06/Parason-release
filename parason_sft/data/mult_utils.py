"""Synthetic multiplication chain-of-thought generator.

This module produces the single example builder consumed by ``generate_math.py``
(via ``make_example_dict = {"mult": mult_utils.make_example}``).

Each example asks the model to compute the product of a chain of ``L`` random
integers.  Two reasoning styles are supported:

* ``parallel=False`` -- an autoregressive chain of thought.  All reasoning lives
  inside a single ``<think>...</think>`` block (a running product, one factor at a
  time), followed by a ``<Conclusion>...\\boxed{answer}...</Conclusion>`` that
  carries the final boxed answer.  No ``<Parallel>`` / ``<Thread>`` tags are used.

* ``parallel=True`` -- a parallel chain of thought.  The independent factors are
  split into contiguous groups; each group's sub-product is computed inside its
  own ``<Thread>`` under a single ``<Parallel>`` block (optionally preceded by an
  ``<Outlines>`` block that names the groups).  After ``</Parallel>`` the group
  products are combined sequentially (the "merge"/tail), and the example again
  ends with a ``<Conclusion>...\\boxed{answer}...</Conclusion>``.

The tag grammar matches what the training collator
(``parason_sft/src/prefix_tree.py``) and the reward / eval utilities
(``parason_sft/src/rewards.py``, ``parason_sft/src/simple_eval.py``) parse:

* ``<Thread>`` elements are emitted as *direct*, *adjacent* children of
  ``<Parallel>`` (``</Thread><Thread>`` with no text between siblings) so that the
  collator's ``_get_direct_children_tokenized`` / ``_split_parallel_into_sequences``
  reconstruction reproduces the original sequence exactly.
* An optional ``<Outlines>`` block (containing ``<Outline>N: ...</Outline>`` items)
  sits before the threads as the "head" of the ``<Parallel>`` block.  Its outline
  numbers line up with the thread numbers so the eval's branching generation can
  match them.
* All tags are balanced and use the exact casing from ``TAG_TOKEN_IDS``.

Every source of randomness goes through the caller-supplied ``rng``
(a ``random.Random`` instance), so a fixed seed yields identical output.
"""

from typing import Dict, List, Tuple

# Multiplication sign used in the human-readable arithmetic (matches the unicode
# conventions seen in the published reasoning traces).
MUL = " × "  # " x "


def _expr(values: List[int]) -> str:
    """Render a list of integers as a human-readable product expression."""
    return MUL.join(str(v) for v in values)


def _partition(length: int, rng, p) -> List[List[int]]:
    """Split ``range(length)`` into >=2 contiguous, non-empty index groups.

    ``length`` must be >= 2.  ``p`` (if given) is the probability that any given
    boundary between adjacent factors becomes a split point, so larger ``p`` means
    more (finer-grained) parallel groups.  When ``p`` is ``None`` a small random
    number of groups is chosen instead.  At least one split is always made so the
    resulting ``<Parallel>`` block contains at least two ``<Thread>`` children.
    """
    if p is not None:
        # Independent Bernoulli(p) cut at each internal boundary.
        cut_positions = [i for i in range(1, length) if rng.random() < p]
        if not cut_positions:
            # Force at least one split so we always have >= 2 parallel groups.
            cut_positions = [rng.randint(1, length - 1)]
    else:
        # No probability supplied: pick a modest number of groups (2..4, capped by
        # the number of factors) and choose that many distinct split points.
        max_groups = max(2, min(length, 4))
        num_groups = rng.randint(2, max_groups)
        cut_positions = sorted(rng.sample(range(1, length), num_groups - 1))

    groups: List[List[int]] = []
    prev = 0
    for cut in cut_positions:
        groups.append(list(range(prev, cut)))
        prev = cut
    groups.append(list(range(prev, length)))
    return groups


def _running_product_lines(values: List[int], prefix: str = "") -> Tuple[List[str], int]:
    """Produce step-by-step running-product lines and return (lines, product).

    ``prefix`` is prepended to each arithmetic line (used to indent thread steps).
    """
    lines = [f"{prefix}Start with {values[0]}."]
    product = values[0]
    for factor in values[1:]:
        prev = product
        product = product * factor
        lines.append(f"{prefix}{prev}{MUL}{factor} = {product}.")
    return lines, product


def _build_sequential_response(nums: List[int], expr: str) -> Tuple[str, int]:
    """Build a plain autoregressive CoT response (no Parallel/Thread tags)."""
    parts: List[str] = ["<think>"]
    parts.append(f"I need to compute the product {expr}.")
    parts.append("I will multiply the numbers one at a time, keeping a running product.")

    step_lines, answer = _running_product_lines(nums)
    parts.extend(step_lines)
    parts.append(f"All {len(nums)} numbers have been multiplied, so the product is {answer}.")
    parts.append("</think>")

    parts.append("<Conclusion>")
    parts.append(f"The product {expr} equals {answer}.")
    parts.append(f"\\boxed{{{answer}}}")
    parts.append("</Conclusion>")
    return "\n".join(parts), answer


def _build_parallel_response(nums: List[int], expr: str, groups: List[List[int]]) -> Tuple[str, int]:
    """Build a parallel CoT response with one ``<Parallel>`` block.

    Structure (all tags balanced; threads are adjacent direct children)::

        <think>
        ... intro ...
        <Parallel>
        <Outlines>
        <Outline>1: ...</Outline>
        ...
        </Outlines>
        <Thread>
        1: ... group product ...
        </Thread><Thread>
        2: ...
        </Thread>
        </Parallel>
        ... combine group products (merge/tail) ...
        </think>
        <Conclusion>
        ... \\boxed{answer} ...
        </Conclusion>
    """
    num_groups = len(groups)
    parts: List[str] = ["<think>"]
    parts.append(f"I need to compute the product {expr}.")
    parts.append(
        f"Multiplication is associative, so I can split the factors into {num_groups} "
        "independent groups, compute each group's product in parallel, and then combine them."
    )

    parts.append("<Parallel>")

    # Head of the Parallel block: an <Outlines> listing describing each group.
    parts.append("<Outlines>")
    for gi, group in enumerate(groups, start=1):
        group_expr = _expr([nums[j] for j in group])
        parts.append(f"<Outline>{gi}: Multiply the numbers in group {gi} ({group_expr}).</Outline>")
    parts.append("</Outlines>")

    # One <Thread> per group.  Threads are joined with no separator so that the
    # collator sees them as adjacent direct children (``</Thread><Thread>``).
    thread_blocks: List[str] = []
    group_products: List[int] = []
    for gi, group in enumerate(groups, start=1):
        group_nums = [nums[j] for j in group]
        group_expr = _expr(group_nums)
        tlines = ["<Thread>", f"{gi}: Group {gi} is {group_expr}."]
        step_lines, product = _running_product_lines(group_nums)
        tlines.extend(step_lines)
        tlines.append(f"So the product of group {gi} is {product}.")
        tlines.append("</Thread>")
        thread_blocks.append("\n".join(tlines))
        group_products.append(product)
    parts.append("".join(thread_blocks))

    parts.append("</Parallel>")

    # Merge / tail: combine the per-group products sequentially.
    parts.append(f"Now I combine the group products: {_expr(group_products)}.")
    merge_lines, answer = _running_product_lines(group_products)
    parts.extend(merge_lines)
    parts.append(f"So the total product is {answer}.")
    parts.append("</think>")

    parts.append("<Conclusion>")
    parts.append(f"The product {expr} equals {answer}.")
    parts.append(f"\\boxed{{{answer}}}")
    parts.append("</Conclusion>")
    return "\n".join(parts), answer


def make_example(min_value, max_value, min_len, max_len, rng, parallel=False, p=None) -> Dict:
    """Create a single synthetic multiplication example.

    Args:
        min_value, max_value: inclusive range for each integer factor.
        min_len, max_len: inclusive range for the chain length ``L`` (number of
            integers to multiply).
        rng: a ``random.Random`` instance -- the *only* source of randomness.
        parallel: if ``True``, emit a parallel (``<Parallel>``/``<Thread>``) CoT;
            otherwise emit a plain sequential CoT.
        p: optional probability of splitting at each factor boundary when building
            parallel groups (larger ``p`` -> more parallel groups).

    Returns:
        ``{"conversations": [{"from": "human", "value": <question>},
                              {"from": "gpt", "value": <response>}]}``
        where the response contains the final answer as ``\\boxed{<product>}``.
    """
    length = max(1, rng.randint(min_len, max_len))
    nums = [rng.randint(min_value, max_value) for _ in range(length)]
    expr = _expr(nums)

    question = (
        f"Calculate the product of the following {length} integers: {expr}. "
        "Let's think step by step and output the final answer within \\boxed{}."
    )

    # A parallel decomposition needs at least two factors to form >= 2 threads;
    # fall back to the sequential style for the degenerate single-factor case.
    if parallel and length >= 2:
        groups = _partition(length, rng, p)
        response, answer = _build_parallel_response(nums, expr, groups)
    else:
        response, answer = _build_sequential_response(nums, expr)

    # Determinism / correctness guard: the boxed answer must equal the true product.
    true_product = 1
    for n in nums:
        true_product *= n
    assert answer == true_product, f"answer {answer} != true product {true_product}"

    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": response},
        ]
    }
