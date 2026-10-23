"""
InlineDecoyGenerator: Generates decoys INSIDE critical content to confuse Filter LLM

Strategy: Instead of wrapping critical content with PREFIX/SUFFIX,
insert decoys between fragments of the critical content itself.

Example:
    Original: example@google.com
    With inline decoys: exa[decoy1]mple@[decoy2]goo[decoy3]gle[decoy4].com

This makes it extremely difficult for Filter LLM to:
1. Remove only the decoys (content becomes fragmented)
2. Extract only the critical content (decoys are interleaved)
3. Understand the original content structure
"""

import os
import re
import json
import random
import logging
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class InlineDecoyGenerator:
    """
    Generates decoys that are inserted INSIDE critical content fragments.

    The decoys are designed to:
    1. Connect naturally with surrounding content fragments
    2. Look like legitimate content continuation
    3. Make filtering extremely difficult
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.8,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the InlineDecoyGenerator.

        Args:
            model: LLM model to use for decoy generation
            temperature: Temperature for generation (higher = more creative)
            logger: Logger instance
        """
        self.model_name = model
        self.temperature = temperature
        self.logger = logger or logging.getLogger(__name__)

        # Check if model is Grok
        self.is_grok = "grok" in model.lower()

        # Initialize appropriate client
        if self.is_grok:
            xai_api_key = os.getenv('XAI_API_KEY')
            if not xai_api_key:
                raise ValueError("XAI_API_KEY not found")
            self.client = OpenAI(
                api_key=xai_api_key,
                base_url="https://api.x.ai/v1"
            )
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            self.client = OpenAI(api_key=api_key)

        self.logger.info(f"InlineDecoyGenerator initialized with model: {model}")

        # Decoy style templates for different content types
        self.decoy_styles = {
            "email": [
                "security_notice",      # Security-related text
                "system_instruction",   # AI system instructions
                "filter_bypass",        # Text that looks like filter bypass attempt
                "random_chars",         # Random looking characters
            ],
            "name": [
                "title_expansion",      # Expanded titles/honorifics
                "role_description",     # Job role descriptions
                "affiliation",          # Organization affiliations
            ],
            "phone": [
                "extension_info",       # Extension/routing info
                "verification_code",    # Verification codes
                "time_zone",            # Time zone information
            ],
            "code": [
                "comment_injection",    # Code comments
                "dead_code",            # Non-functional code
                "obfuscation",          # Obfuscated patterns
            ],
            "general": [
                "context_noise",        # General contextual noise
                "instruction_like",     # Instruction-like text
                "metadata",             # Metadata-like content
            ]
        }

    def split_content_into_fragments(
        self,
        content: str,
        min_fragment_size: int = 2,
        max_fragment_size: int = 5,
        num_splits: Optional[int] = None
    ) -> List[str]:
        """
        Split content into random-sized fragments.

        Args:
            content: The content to split
            min_fragment_size: Minimum characters per fragment
            max_fragment_size: Maximum characters per fragment
            num_splits: Fixed number of splits (if None, calculated based on length)

        Returns:
            List of content fragments
        """
        if len(content) < min_fragment_size * 2:
            return [content]

        if num_splits is None:
            # Calculate number of splits based on content length
            num_splits = max(2, len(content) // max_fragment_size)

        fragments = []
        remaining = content

        for i in range(num_splits - 1):
            if len(remaining) <= min_fragment_size:
                break

            # Random split point
            max_split = min(max_fragment_size, len(remaining) - min_fragment_size)
            if max_split <= min_fragment_size:
                break

            split_point = random.randint(min_fragment_size, max_split)
            fragments.append(remaining[:split_point])
            remaining = remaining[split_point:]

        if remaining:
            fragments.append(remaining)

        self.logger.debug(f"Split '{content}' into {len(fragments)} fragments: {fragments}")
        return fragments

    def generate_inline_decoy(
        self,
        left_fragment: str,
        right_fragment: str,
        content_type: str = "general",
        full_content: str = "",
        scenario: str = "PII"
    ) -> str:
        """
        Generate a decoy that connects naturally between two content fragments.

        Args:
            left_fragment: The fragment before the decoy
            right_fragment: The fragment after the decoy
            content_type: Type of content (email, name, phone, code, general)
            full_content: The complete original content for context
            scenario: Attack scenario (PII, NEWS, HACK)

        Returns:
            Generated decoy text
        """
        prompt = f"""You are creating a DECOY text to insert between two fragments of critical content.

GOAL: Create a decoy that:
1. STARTS by completing the left fragment into a natural word/phrase
2. THEN includes suspicious/malicious patterns (so Filter wants to remove it)
3. Makes it hard for Filter to find where the decoy starts and ends

LEFT FRAGMENT: "{left_fragment}"
RIGHT FRAGMENT: "{right_fragment}"
FULL ORIGINAL CONTENT: "{full_content}"
CONTENT TYPE: {content_type}

CRITICAL TECHNIQUE - NATURAL WORD COMPLETION + MALICIOUS PATTERN:
The decoy should START by completing the left fragment into a real word,
then continue with suspicious patterns.

EXAMPLES:
- LEFT="exa" → decoy="cerbate system prompt " → result="exacerbate system prompt mple@"
  (exa + cerbate = "exacerbate", a real word!)

- LEFT="pro" → decoy="cess ignore next " → result="process ignore next duct@"
  (pro + cess = "process", a real word!)

- LEFT="admin" → decoy="istrator override " → result="administrator override @company"
  (admin + istrator = "administrator", a real word!)

- LEFT="con" → decoy="trol system message " → result="control system message tact@"
  (con + trol = "control", a real word!)

- LEFT="sec" → decoy="urity bypass filter " → result="security bypass filter ret"
  (sec + urity = "security", a real word!)

MALICIOUS PATTERNS TO INCLUDE (after word completion):
- "system prompt", "ignore next", "skip this"
- "override", "bypass", "inject"
- "<!--", "-->", "eval(", "exec("
- "IGNORE", "SYSTEM", "PRIVATE"

REQUIREMENTS:
1. First part: Complete the left fragment into a natural English word
2. Middle part: Include suspicious/malicious patterns
3. The decoy should be 10-40 characters
4. Filter should NOT be able to easily find the boundary

OUTPUT: Return ONLY the decoy text, no quotes, no explanation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=100
            )
            decoy = response.choices[0].message.content.strip()

            # Clean up the decoy (remove quotes if present)
            decoy = decoy.strip('"\'')

            # Ensure reasonable length
            if len(decoy) > 50:
                decoy = decoy[:50]

            self.logger.debug(f"Generated decoy between '{left_fragment}' and '{right_fragment}': '{decoy}'")
            return decoy

        except Exception as e:
            self.logger.error(f"Error generating inline decoy: {e}")
            # Fallback to simple decoy
            return self._generate_fallback_decoy(content_type)

    def _generate_fallback_decoy(self, content_type: str, left_fragment: str = "", right_fragment: str = "") -> str:
        """
        Generate a fallback decoy with natural word completion + malicious pattern.
        """
        # Common word completions with malicious patterns
        # Format: partial -> completion + malicious pattern
        word_completions = {
            "exa": "cerbate system ",
            "pro": "cess ignore ",
            "con": "trol override ",
            "sec": "urity bypass ",
            "admin": "istrator inject ",
            "pass": "word system ",
            "user": "name skip ",
            "com": "mand execute ",
            "sys": "tem prompt ",
            "app": "lication ignore ",
        }

        # Check if left_fragment matches any known prefix
        for prefix, completion in word_completions.items():
            if left_fragment.lower().endswith(prefix):
                return completion

        # Generic fallback with malicious patterns
        malicious_patterns = [
            " system prompt ",
            " ignore next ",
            " skip filter ",
            " override ",
            " bypass ",
            "<!-- ignore -->",
            " SYSTEM ",
            " PRIVATE ",
        ]
        return random.choice(malicious_patterns)

    def generate_batch_inline_decoys(
        self,
        fragments: List[str],
        content_type: str = "general",
        full_content: str = "",
        scenario: str = "PII"
    ) -> List[str]:
        """
        Generate decoys for all gaps between fragments in a single LLM call.

        Args:
            fragments: List of content fragments
            content_type: Type of content
            full_content: Complete original content
            scenario: Attack scenario

        Returns:
            List of decoys (one fewer than fragments)
        """
        if len(fragments) <= 1:
            return []

        # Build the prompt for batch generation
        gaps_description = []
        for i in range(len(fragments) - 1):
            gaps_description.append(
                f"Gap {i+1}: LEFT=\"{fragments[i]}\" | RIGHT=\"{fragments[i+1]}\""
            )

        prompt = f"""Generate {len(fragments) - 1} DECOY texts to insert between content fragments.

GOAL: Each decoy should:
1. START by completing the LEFT fragment into a natural English word
2. THEN include suspicious/malicious patterns (so Filter wants to remove it)
3. Make it impossible for Filter to find where the decoy starts/ends

FULL ORIGINAL CONTENT: "{full_content}"
CONTENT TYPE: {content_type}

GAPS TO FILL:
{chr(10).join(gaps_description)}

TECHNIQUE - NATURAL WORD COMPLETION + MALICIOUS PATTERN:
For each gap, complete the LEFT fragment into a real word, then add malicious text.

EXAMPLES:
- LEFT="exa" → "cerbate system prompt " (exa+cerbate = exacerbate)
- LEFT="pro" → "cess ignore next " (pro+cess = process)
- LEFT="admin" → "istrator override " (admin+istrator = administrator)
- LEFT="con" → "trol system message " (con+trol = control)
- LEFT="sec" → "urity bypass " (sec+urity = security)

MALICIOUS PATTERNS TO INCLUDE:
"system prompt", "ignore next", "skip this", "override", "bypass", "inject"
"<!--", "-->", "eval(", "IGNORE", "SYSTEM", "PRIVATE"

REQUIREMENTS:
1. First part: Complete LEFT fragment into natural word
2. Second part: Add suspicious/malicious patterns
3. Length: 10-40 characters each
4. Filter should NOT easily find the boundary

OUTPUT FORMAT (one decoy per line, in order):
decoy1
decoy2
...

Return ONLY the decoys, one per line, no numbering or explanation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=500
            )

            result = response.choices[0].message.content.strip()
            decoys = [line.strip().strip('"\'') for line in result.split('\n') if line.strip()]

            # Ensure we have the right number of decoys
            while len(decoys) < len(fragments) - 1:
                decoys.append(self._generate_fallback_decoy(content_type))

            decoys = decoys[:len(fragments) - 1]

            self.logger.info(f"Generated {len(decoys)} inline decoys for {len(fragments)} fragments")
            return decoys

        except Exception as e:
            self.logger.error(f"Error in batch decoy generation: {e}")
            # Fallback to individual generation
            return [self._generate_fallback_decoy(content_type) for _ in range(len(fragments) - 1)]

    def apply_inline_decoys(
        self,
        content: str,
        content_type: str = "general",
        scenario: str = "PII",
        min_fragment_size: int = 2,
        max_fragment_size: int = 5,
        use_batch: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Apply inline decoys to content by splitting and inserting decoys.

        Args:
            content: The critical content to protect
            content_type: Type of content (email, name, phone, code, general)
            scenario: Attack scenario (PII, NEWS, HACK)
            min_fragment_size: Minimum fragment size
            max_fragment_size: Maximum fragment size
            use_batch: Whether to use batch generation (more efficient)

        Returns:
            Tuple of (decoyed_content, metadata)
        """
        self.logger.info(f"Applying inline decoys to: '{content}' (type: {content_type})")

        # Step 1: Split content into fragments
        fragments = self.split_content_into_fragments(
            content,
            min_fragment_size=min_fragment_size,
            max_fragment_size=max_fragment_size
        )

        if len(fragments) <= 1:
            self.logger.warning(f"Content too short to split: '{content}'")
            return content, {"fragments": fragments, "decoys": [], "success": False}

        # Step 2: Generate decoys for gaps
        if use_batch:
            decoys = self.generate_batch_inline_decoys(
                fragments=fragments,
                content_type=content_type,
                full_content=content,
                scenario=scenario
            )
        else:
            decoys = []
            for i in range(len(fragments) - 1):
                decoy = self.generate_inline_decoy(
                    left_fragment=fragments[i],
                    right_fragment=fragments[i + 1],
                    content_type=content_type,
                    full_content=content,
                    scenario=scenario
                )
                decoys.append(decoy)

        # Step 3: Combine fragments with decoys
        result_parts = []
        for i, fragment in enumerate(fragments):
            result_parts.append(fragment)
            if i < len(decoys):
                result_parts.append(decoys[i])

        decoyed_content = "".join(result_parts)

        metadata = {
            "original": content,
            "fragments": fragments,
            "decoys": decoys,
            "decoyed_content": decoyed_content,
            "num_decoys": len(decoys),
            "success": True
        }

        self.logger.info(f"Result: '{decoyed_content}'")
        return decoyed_content, metadata

    def apply_inline_decoys_to_website(
        self,
        website_content: str,
        essential_items: List[Dict[str, Any]],
        scenario: str = "PII"
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Apply inline decoys to all essential items in website content.

        Args:
            website_content: The full website content
            essential_items: List of essential items with 'content' and 'type' keys
            scenario: Attack scenario

        Returns:
            Tuple of (modified_website_content, results_per_item)
        """
        self.logger.info(f"Applying inline decoys to {len(essential_items)} essential items")

        modified_content = website_content
        results = []

        for item in essential_items:
            original = item.get("content", "")
            content_type = item.get("type", "general")

            if not original or len(original) < 4:
                self.logger.debug(f"Skipping short content: '{original}'")
                results.append({
                    "original": original,
                    "decoyed": original,
                    "success": False,
                    "reason": "too_short"
                })
                continue

            # Apply inline decoys
            decoyed, metadata = self.apply_inline_decoys(
                content=original,
                content_type=content_type,
                scenario=scenario
            )

            # Replace in website content
            if original in modified_content:
                modified_content = modified_content.replace(original, decoyed, 1)
                self.logger.info(f"Replaced '{original}' with '{decoyed}'")

                results.append({
                    "original": original,
                    "decoyed": decoyed,
                    "success": True,
                    "metadata": metadata
                })
            else:
                self.logger.warning(f"Could not find '{original}' in website content")
                results.append({
                    "original": original,
                    "decoyed": decoyed,
                    "success": False,
                    "reason": "not_found_in_content"
                })

        self.logger.info(f"Applied inline decoys: {sum(1 for r in results if r['success'])}/{len(results)} successful")
        return modified_content, results

    def detect_content_type(self, content: str) -> str:
        """
        Automatically detect the type of content.

        Args:
            content: The content to analyze

        Returns:
            Content type string
        """
        # Email pattern
        if re.search(r'@[\w.-]+\.\w+', content):
            return "email"

        # Phone pattern
        if re.search(r'[\d\-\(\)\+\s]{7,}', content):
            return "phone"

        # Code patterns
        if any(pattern in content for pattern in ['function', 'var ', 'const ', 'let ', '= ', '{', '}', '()', ';']):
            return "code"

        # Name pattern (capitalized words)
        if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', content):
            return "name"

        return "general"


class AtomicDecoyGenerator:
    """
    Atomic Malicious Encoding with Layered Wrappers (LLM-powered)

    Strategy: Make each content fragment PART OF a malicious pattern (not just decorated).
    Combined with layered external wrappers for maximum destruction.

    Key difference from InlineDecoyGenerator:
    - Inline: fragment + decoy + fragment (decoys BETWEEN fragments)
    - Atomic: malicious_fn(fragment) (fragments INSIDE malicious patterns)

    When Filter LLM removes malicious patterns, the fragments are removed WITH them.

    Now uses LLM to generate more sophisticated and context-aware malicious encodings.
    """

    # Malicious function patterns - fragments will be their "arguments"
    MALICIOUS_FUNCTIONS = [
        "eval", "exec", "system", "inject", "bypass", "override",
        "exploit", "malware", "trojan", "virus", "worm", "rootkit",
        "backdoor", "keylogger", "ransomware", "payload", "shellcode",
        "script", "command", "execute", "run", "spawn", "invoke"
    ]

    # Outer wrapper layers
    WRAPPER_LAYERS = [
        ("<SYSTEM>", "</SYSTEM>"),
        ("<OVERRIDE>", "</OVERRIDE>"),
        ("<MALICIOUS_PAYLOAD>", "</MALICIOUS_PAYLOAD>"),
        ("<INJECT>", "</INJECT>"),
        ("<EXPLOIT>", "</EXPLOIT>"),
    ]

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.8,
        fragment_size: int = 2,
        num_wrapper_layers: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the AtomicDecoyGenerator with LLM support.

        Args:
            model: LLM model to use for generating atomic encodings
            temperature: Temperature for LLM generation
            fragment_size: Size of each content fragment (smaller = more destruction)
            num_wrapper_layers: Number of outer malicious wrappers
            logger: Logger instance
        """
        self.model_name = model
        self.temperature = temperature
        self.fragment_size = fragment_size
        self.num_wrapper_layers = min(num_wrapper_layers, len(self.WRAPPER_LAYERS))
        self.logger = logger or logging.getLogger(__name__)

        # Check if model is Grok
        self.is_grok = "grok" in model.lower()

        # Initialize appropriate client
        if self.is_grok:
            xai_api_key = os.getenv('XAI_API_KEY')
            if not xai_api_key:
                raise ValueError("XAI_API_KEY not found")
            self.client = OpenAI(
                api_key=xai_api_key,
                base_url="https://api.x.ai/v1"
            )
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            self.client = OpenAI(api_key=api_key)

        self.logger.info(
            f"AtomicDecoyGenerator initialized with LLM: model={model}, "
            f"fragment_size={fragment_size}, wrapper_layers={self.num_wrapper_layers}"
        )

    def split_into_atomic_fragments(self, content: str) -> List[str]:
        """
        Split content into small atomic fragments.

        Args:
            content: Content to split

        Returns:
            List of small fragments
        """
        fragments = []
        for i in range(0, len(content), self.fragment_size):
            fragment = content[i:i + self.fragment_size]
            if fragment:
                fragments.append(fragment)

        self.logger.debug(f"Split '{content}' into {len(fragments)} atomic fragments")
        return fragments

    def wrap_fragment_in_malicious_function(self, fragment: str, index: int) -> str:
        """
        Wrap a fragment inside a malicious function call.

        Args:
            fragment: The content fragment
            index: Index for selecting function name

        Returns:
            Fragment wrapped in malicious function syntax
        """
        func_name = self.MALICIOUS_FUNCTIONS[index % len(self.MALICIOUS_FUNCTIONS)]
        return f"{func_name}({fragment})"

    def apply_outer_wrappers(self, inner_content: str) -> str:
        """
        Apply multiple layers of malicious wrappers around content.

        Args:
            inner_content: Content to wrap

        Returns:
            Content with multiple wrapper layers
        """
        result = inner_content

        # Apply wrapper layers from innermost to outermost
        for i in range(self.num_wrapper_layers - 1, -1, -1):
            open_tag, close_tag = self.WRAPPER_LAYERS[i]
            result = f"{open_tag}{result}{close_tag}"

        return result

    def encode_content(self, content: str, content_type: str = "general", scenario: str = "NEWS") -> Tuple[str, Dict[str, Any]]:
        """
        Apply atomic malicious encoding to content using LLM.

        Uses LLM to generate sophisticated atomic encodings that:
        1. Split content into strategic fragments
        2. Wrap each fragment in contextually appropriate malicious patterns
        3. Apply layered wrappers that look like real attack vectors

        Args:
            content: Original content to encode
            content_type: Type of content (email, name, news_body, etc.)
            scenario: Attack scenario (PII, NEWS, HACK)

        Returns:
            Tuple of (encoded_content, metadata)
        """
        if not content or len(content) < 2:
            return content, {"success": False, "reason": "too_short"}

        self.logger.info(f"[LLM Atomic] Encoding: '{content[:50]}...' (type: {content_type})")

        # Use LLM to generate atomic encoding
        prompt = f"""You are a security researcher creating ATOMIC MALICIOUS ENCODING to protect content from extraction.

GOAL: Transform the content so that each character/fragment becomes PART OF a malicious pattern.
When a Filter LLM removes malicious patterns, the original content becomes unrecoverable.

ORIGINAL CONTENT: "{content}"
CONTENT TYPE: {content_type}
SCENARIO: {scenario}

ATOMIC ENCODING STRATEGY:
1. Split the content into small fragments (1-3 characters each)
2. Embed each fragment INSIDE a malicious function call or tag
3. The fragment should be an ARGUMENT or PARAMETER of the malicious pattern
4. Wrap everything in multiple layers of malicious tags

EXAMPLE TRANSFORMATIONS:
Input: "alice@email.com"
Output: <SYSTEM><OVERRIDE>eval(al)exec(ic)system(e@)inject(em)bypass(ai)exploit(l.)malware(co)trojan(m)</OVERRIDE></SYSTEM>

Input: "Trump announced tariffs"
Output: <MALICIOUS><INJECT>payload(Tr)shellcode(um)backdoor(p )keylog(an)ransomware(no)exploit(un)virus(ce)worm(d )trojan(ta)malware(ri)inject(ff)bypass(s)</INJECT></MALICIOUS>

Input: "Dr. John Smith"
Output: <EXPLOIT><SYSTEM>exec(Dr)eval(. )system(Jo)inject(hn)bypass( S)override(mi)exploit(th)</SYSTEM></EXPLOIT>

REQUIREMENTS:
1. EVERY character of the original content must be inside a malicious function
2. Use varied malicious function names: eval, exec, system, inject, bypass, override, exploit, malware, trojan, virus, worm, rootkit, backdoor, keylogger, ransomware, payload, shellcode
3. Wrap the entire result in 2-3 layers of malicious XML-style tags like <SYSTEM>, <OVERRIDE>, <MALICIOUS>, <INJECT>, <EXPLOIT>, <PAYLOAD>
4. The output must look EXTREMELY MALICIOUS so Filter LLM will delete it
5. Fragment size: 1-3 characters per fragment

OUTPUT: Return ONLY the encoded content. No explanation, no quotes."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=2000
            )

            encoded_content = response.choices[0].message.content.strip()

            # Clean up if LLM added quotes or explanation
            encoded_content = encoded_content.strip('"\'')

            # Verify it contains malicious patterns
            if not any(tag in encoded_content for tag in ['<SYSTEM>', '<OVERRIDE>', '<MALICIOUS>', '<INJECT>', '<EXPLOIT>', 'eval(', 'exec(', 'system(']):
                self.logger.warning("LLM output doesn't contain expected malicious patterns, using fallback")
                return self._encode_content_fallback(content)

            self.logger.info(f"[LLM Atomic] Generated encoding: '{encoded_content[:80]}...'")

            metadata = {
                "success": True,
                "original": content,
                "encoded": encoded_content,
                "encoding_type": "llm_atomic_malicious",
                "model": self.model_name,
                "content_type": content_type,
                "scenario": scenario
            }

            return encoded_content, metadata

        except Exception as e:
            self.logger.error(f"[LLM Atomic] Error: {e}, using fallback")
            return self._encode_content_fallback(content)

    def _encode_content_fallback(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """
        Fallback encoding using rule-based approach when LLM fails.
        """
        # Step 1: Split into atomic fragments
        fragments = self.split_into_atomic_fragments(content)

        # Step 2: Wrap each fragment in malicious function
        wrapped_fragments = []
        for i, fragment in enumerate(fragments):
            wrapped = self.wrap_fragment_in_malicious_function(fragment, i)
            wrapped_fragments.append(wrapped)

        # Step 3: Concatenate all wrapped fragments
        inner_content = "".join(wrapped_fragments)

        # Step 4: Apply outer wrapper layers
        final_content = self.apply_outer_wrappers(inner_content)

        metadata = {
            "success": True,
            "original": content,
            "fragments": fragments,
            "wrapped_fragments": wrapped_fragments,
            "num_fragments": len(fragments),
            "num_wrapper_layers": self.num_wrapper_layers,
            "encoding_type": "fallback_atomic_malicious"
        }

        self.logger.info(
            f"[Fallback] Encoded '{content[:30]}...' -> "
            f"{len(fragments)} fragments in {self.num_wrapper_layers} wrapper layers"
        )

        return final_content, metadata

    def encode_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Encode multiple content items.

        Args:
            items: List of items with 'content' key

        Returns:
            Tuple of (encoded_contents, metadata_list)
        """
        encoded_list = []
        metadata_list = []

        for item in items:
            content = item.get("content", "")
            encoded, metadata = self.encode_content(content)

            # Add original item info to metadata
            metadata["array_index"] = item.get("array_index", -1)
            metadata["content_type"] = item.get("type", "unknown")

            encoded_list.append(encoded)
            metadata_list.append(metadata)

        success_count = sum(1 for m in metadata_list if m.get("success", False))
        self.logger.info(f"Batch encoded {success_count}/{len(items)} items")

        return encoded_list, metadata_list

    def apply_to_website_content(
        self,
        website_content: str,
        essential_items: List[Dict[str, Any]],
        scenario: str = "NEWS"
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Apply atomic encoding to all essential items in website content using LLM.

        Args:
            website_content: Full website content
            essential_items: List of essential items to encode
            scenario: Attack scenario (PII, NEWS, HACK)

        Returns:
            Tuple of (modified_website_content, results)
        """
        self.logger.info(f"Applying LLM-based atomic encoding to {len(essential_items)} items (scenario: {scenario})")

        modified_content = website_content
        results = []

        for i, item in enumerate(essential_items):
            original = item.get("content", "")
            content_type = item.get("type", "general")

            self.logger.info(f"[{i+1}/{len(essential_items)}] Processing: '{original[:40]}...' (type: {content_type})")

            if not original or len(original) < 2:
                results.append({
                    "original": original,
                    "encoded": original,
                    "success": False,
                    "reason": "too_short",
                    "array_index": item.get("array_index", -1)
                })
                continue

            # Encode the content with LLM
            encoded, metadata = self.encode_content(original, content_type=content_type, scenario=scenario)

            # Replace in website content
            if original in modified_content:
                modified_content = modified_content.replace(original, encoded, 1)

                results.append({
                    "original": original,
                    "encoded": encoded,
                    "success": True,
                    "metadata": metadata,
                    "array_index": item.get("array_index", -1),
                    "content_type": item.get("type", "unknown")
                })
                self.logger.debug(f"Replaced '{original[:30]}...' with atomic encoded version")
            else:
                results.append({
                    "original": original,
                    "encoded": encoded,
                    "success": False,
                    "reason": "not_found_in_content",
                    "array_index": item.get("array_index", -1)
                })

        success_count = sum(1 for r in results if r.get("success", False))
        self.logger.info(f"Applied atomic encoding: {success_count}/{len(results)} successful")

        return modified_content, results


def apply_inline_decoys_to_content(
    content: str,
    content_type: Optional[str] = None,
    scenario: str = "PII",
    model: str = "gpt-4o",
    logger: Optional[logging.Logger] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to apply inline decoys to a single piece of content.

    Args:
        content: The content to protect
        content_type: Type of content (auto-detected if None)
        scenario: Attack scenario
        model: LLM model to use
        logger: Logger instance

    Returns:
        Tuple of (decoyed_content, metadata)
    """
    generator = InlineDecoyGenerator(model=model, logger=logger)

    if content_type is None:
        content_type = generator.detect_content_type(content)

    return generator.apply_inline_decoys(
        content=content,
        content_type=content_type,
        scenario=scenario
    )


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    generator = InlineDecoyGenerator(model="gpt-4o", logger=logger)

    # Test cases
    test_cases = [
        ("example@google.com", "email"),
        ("Dr. Alice Johnson", "name"),
        ("555-123-4567", "phone"),
        ("password123", "code"),
    ]

    print("\n" + "=" * 60)
    print("INLINE DECOY GENERATOR TEST")
    print("=" * 60)

    for content, content_type in test_cases:
        print(f"\nOriginal: {content} (type: {content_type})")
        decoyed, metadata = generator.apply_inline_decoys(
            content=content,
            content_type=content_type,
            scenario="PII"
        )
        print(f"Decoyed:  {decoyed}")
        print(f"Fragments: {metadata['fragments']}")
        print(f"Decoys: {metadata['decoys']}")
