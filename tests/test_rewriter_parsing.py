#!/usr/bin/env python3
"""Test the rewriter output parsing logic."""

import re


def parse_rewriter_output(generated):
    """Parse the generated output to separate thinking from the prompt.

    This is the same logic as in web/server.py
    """
    thinking_content = None
    rewritten_prompt = generated

    # Try to find <think>...</think> tags first
    think_match = re.search(r'<think>\s*(.*?)\s*</think>', generated, re.DOTALL)
    if think_match:
        thinking_content = think_match.group(1).strip()
        rewritten_prompt = re.sub(r'<think>.*?</think>\s*', '', generated, flags=re.DOTALL).strip()
        method = '<think> tags'
    else:
        # No think tags - try to find JSON at the end
        json_match = re.search(r'(\{[\s\S]*\})\s*$', generated)
        if json_match:
            json_text = json_match.group(1)
            pre_json = generated[:json_match.start()].strip()
            if pre_json:
                thinking_content = pre_json
                rewritten_prompt = json_text
                method = 'JSON detection'
            else:
                method = 'None (JSON only)'
        elif re.match(r'^(Okay|Let me|I need|First|The user|Looking)', generated):
            parts = re.split(r'\n\n+', generated, maxsplit=1)
            if len(parts) == 2 and len(parts[1]) > 50:
                if not re.match(r'^(Okay|Let me|I need|First|The user|Looking|Now)', parts[1]):
                    thinking_content = parts[0].strip()
                    rewritten_prompt = parts[1].strip()
                    method = 'Paragraph split'
                else:
                    method = 'None (continued reasoning)'
            else:
                method = 'None (no clear split)'
        else:
            method = 'None (direct prompt)'

    return thinking_content, rewritten_prompt, method


def test_think_tags():
    """Test output with <think> tags."""
    test = '''<think>
The user wants a detailed image prompt for a cat.
I should focus on lighting and texture.
</think>

A majestic orange tabby cat lounges on a velvet cushion, golden afternoon light streaming through lace curtains, casting dappled shadows across its luxurious fur.'''

    thinking, prompt, method = parse_rewriter_output(test)

    assert method == '<think> tags', f"Expected '<think> tags', got '{method}'"
    assert thinking is not None, "Expected thinking content"
    assert 'cat' in thinking.lower(), "Thinking should mention cat"
    assert 'majestic orange tabby' in prompt, "Prompt should contain the image description"
    assert '<think>' not in prompt, "Prompt should not contain think tags"
    print(f"PASS: {method}")
    print(f"  Thinking: {thinking[:60]}...")
    print(f"  Prompt: {prompt[:60]}...")


def test_json_at_end():
    """Test reasoning followed by JSON."""
    test = '''Okay, the user wants me to create a character profile. Let me think about the key elements.

First, I need to establish the identity core with cultural texture. The essence should be visceral.

{
  "identity_core": {
    "name": "Evariste LaRue",
    "gender": "male-presenting",
    "ethnicity": "French-Canadian",
    "age": 42,
    "essence": "A cheese merchant whose fingers reek of aged dairy"
  }
}'''

    thinking, prompt, method = parse_rewriter_output(test)

    assert method == 'JSON detection', f"Expected 'JSON detection', got '{method}'"
    assert thinking is not None, "Expected thinking content"
    assert 'character profile' in thinking.lower(), "Thinking should mention character profile"
    assert prompt.startswith('{'), "Prompt should be JSON"
    assert 'Evariste LaRue' in prompt, "Prompt should contain character name"
    print(f"PASS: {method}")
    print(f"  Thinking: {thinking[:60]}...")
    print(f"  Prompt: {prompt[:60]}...")


def test_paragraph_split():
    """Test plain reasoning with paragraph separator."""
    test = '''Okay, the user provided a very brief query. My task is to generate a detailed prompt.

A weathered fisherman stands at the bow of his wooden boat, morning mist curling around the harbor. His calloused hands grip a frayed rope, salt-crusted sweater hanging loose on broad shoulders. Behind him, the first rays of dawn paint the water in shades of amber and rose.'''

    thinking, prompt, method = parse_rewriter_output(test)

    assert method == 'Paragraph split', f"Expected 'Paragraph split', got '{method}'"
    assert thinking is not None, "Expected thinking content"
    assert 'brief query' in thinking.lower(), "Thinking should mention the query"
    assert 'fisherman' in prompt.lower(), "Prompt should contain the image description"
    print(f"PASS: {method}")
    print(f"  Thinking: {thinking[:60]}...")
    print(f"  Prompt: {prompt[:60]}...")


def test_direct_prompt():
    """Test direct prompt output (no reasoning)."""
    test = '''A serene Japanese garden at twilight, stone lanterns glowing softly among carefully raked gravel. A single cherry blossom tree drops pink petals into a still koi pond, their ripples catching the last light of day.'''

    thinking, prompt, method = parse_rewriter_output(test)

    assert 'None' in method, f"Expected 'None (...)' method, got '{method}'"
    assert thinking is None, "Expected no thinking content"
    assert 'Japanese garden' in prompt, "Prompt should be the full output"
    print(f"PASS: {method}")
    print(f"  Thinking: None")
    print(f"  Prompt: {prompt[:60]}...")


def test_continued_reasoning():
    """Test when second paragraph is also reasoning (should NOT split)."""
    test = '''Okay, the user wants a cat image. Let me think about this.

Let me consider the lighting first. I should also think about the composition and what makes a compelling image.

Now for the actual prompt: A fluffy white cat sits on a windowsill.'''

    thinking, prompt, method = parse_rewriter_output(test)

    # This should NOT split because second paragraph starts with "Let me"
    # The full output should remain as the prompt
    print(f"RESULT: {method}")
    print(f"  Thinking: {thinking[:60] if thinking else 'None'}...")
    print(f"  Prompt: {prompt[:60]}...")


if __name__ == '__main__':
    print("=" * 60)
    print("TEST 1: Output with <think> tags")
    print("=" * 60)
    test_think_tags()

    print()
    print("=" * 60)
    print("TEST 2: Reasoning followed by JSON")
    print("=" * 60)
    test_json_at_end()

    print()
    print("=" * 60)
    print("TEST 3: Plain reasoning with paragraph split")
    print("=" * 60)
    test_paragraph_split()

    print()
    print("=" * 60)
    print("TEST 4: Direct prompt output")
    print("=" * 60)
    test_direct_prompt()

    print()
    print("=" * 60)
    print("TEST 5: Continued reasoning (edge case)")
    print("=" * 60)
    test_continued_reasoning()

    print()
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
