# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check whether a MiniLLM student/teacher pair uses the same token-id space.

MiniLLM compares student and teacher token probabilities by token id. Equal
vocabulary sizes are therefore not enough: every shared id must map to the same
token, and special token ids must agree.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "sentencepiece.bpe.model",
)

SPECIAL_TOKEN_ID_ATTRS = (
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "unk_token_id",
    "sep_token_id",
    "cls_token_id",
    "mask_token_id",
    "additional_special_tokens_ids",
)


def sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [make_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def load_tokenizer(
    model_name_or_path: str,
    *,
    revision: str | None,
    local_files_only: bool,
    trust_remote_code: bool,
) -> PreTrainedTokenizerBase:
    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if revision is not None:
        kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)


def collect_special_token_ids(tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    return {attr: make_jsonable(getattr(tokenizer, attr, None)) for attr in SPECIAL_TOKEN_ID_ATTRS}


def collect_tokenizer_summary(tokenizer: PreTrainedTokenizerBase, model_name_or_path: str) -> dict[str, Any]:
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    backend_fingerprint = sha256_text(backend_tokenizer.to_str()) if backend_tokenizer is not None else None
    chat_template = getattr(tokenizer, "chat_template", None)

    model_path = Path(model_name_or_path)
    file_fingerprints = {}
    if model_path.exists():
        file_fingerprints = {name: sha256_file(model_path / name) for name in TOKENIZER_FILES}

    return {
        "name_or_path": model_name_or_path,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_length": len(tokenizer),
        "added_vocab_size": len(tokenizer.get_added_vocab()),
        "special_tokens_map": make_jsonable(tokenizer.special_tokens_map),
        "special_token_ids": collect_special_token_ids(tokenizer),
        "chat_template_sha256": sha256_text(chat_template),
        "backend_tokenizer_sha256": backend_fingerprint,
        "file_sha256": file_fingerprints,
    }


def token_at_id(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str | None:
    token = tokenizer.convert_ids_to_tokens(token_id)
    if isinstance(token, list):
        return token[0] if token else None
    return token


def collect_examples(items: list[Any], max_examples: int) -> list[Any]:
    return items[:max_examples]


def compare_vocabularies(
    student_vocab: dict[str, int],
    teacher_vocab: dict[str, int],
    *,
    max_examples: int,
) -> dict[str, Any]:
    student_tokens = set(student_vocab)
    teacher_tokens = set(teacher_vocab)
    shared_tokens = student_tokens & teacher_tokens

    shared_id_mismatches = [
        {
            "token": token,
            "student_id": student_vocab[token],
            "teacher_id": teacher_vocab[token],
        }
        for token in sorted(shared_tokens)
        if student_vocab[token] != teacher_vocab[token]
    ]

    return {
        "match": student_vocab == teacher_vocab,
        "student_vocab_entries": len(student_vocab),
        "teacher_vocab_entries": len(teacher_vocab),
        "only_in_student_count": len(student_tokens - teacher_tokens),
        "only_in_teacher_count": len(teacher_tokens - student_tokens),
        "shared_token_id_mismatch_count": len(shared_id_mismatches),
        "only_in_student_examples": collect_examples(sorted(student_tokens - teacher_tokens), max_examples),
        "only_in_teacher_examples": collect_examples(sorted(teacher_tokens - student_tokens), max_examples),
        "shared_token_id_mismatch_examples": collect_examples(shared_id_mismatches, max_examples),
    }


def compare_id_to_token(
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase,
    student_vocab: dict[str, int],
    teacher_vocab: dict[str, int],
    *,
    max_examples: int,
) -> dict[str, Any]:
    token_ids = set(student_vocab.values()) | set(teacher_vocab.values())
    token_ids.update(range(max(len(student_tokenizer), len(teacher_tokenizer))))

    mismatches = []
    for token_id in sorted(token_ids):
        student_token = token_at_id(student_tokenizer, token_id)
        teacher_token = token_at_id(teacher_tokenizer, token_id)
        if student_token != teacher_token:
            mismatches.append(
                {
                    "token_id": token_id,
                    "student_token": student_token,
                    "teacher_token": teacher_token,
                }
            )

    return {
        "match": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "mismatch_examples": collect_examples(mismatches, max_examples),
    }


def compare_file_fingerprints(
    student_summary: dict[str, Any],
    teacher_summary: dict[str, Any],
    *,
    max_examples: int,
) -> dict[str, Any]:
    student_files = student_summary["file_sha256"]
    teacher_files = teacher_summary["file_sha256"]
    file_names = sorted(set(student_files) | set(teacher_files))
    mismatches = [
        {
            "file": file_name,
            "student_sha256": student_files.get(file_name),
            "teacher_sha256": teacher_files.get(file_name),
        }
        for file_name in file_names
        if student_files.get(file_name) != teacher_files.get(file_name)
    ]
    return {
        "match": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "mismatch_examples": collect_examples(mismatches, max_examples),
    }


def compare_tokenizers(
    student_tokenizer: PreTrainedTokenizerBase,
    teacher_tokenizer: PreTrainedTokenizerBase,
    *,
    student_model: str,
    teacher_model: str,
    max_examples: int,
) -> dict[str, Any]:
    student_summary = collect_tokenizer_summary(student_tokenizer, student_model)
    teacher_summary = collect_tokenizer_summary(teacher_tokenizer, teacher_model)

    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    vocab_report = compare_vocabularies(student_vocab, teacher_vocab, max_examples=max_examples)
    id_to_token_report = compare_id_to_token(
        student_tokenizer,
        teacher_tokenizer,
        student_vocab,
        teacher_vocab,
        max_examples=max_examples,
    )

    failures = []
    warnings = []

    if student_summary["tokenizer_length"] != teacher_summary["tokenizer_length"]:
        failures.append(
            "tokenizer lengths differ: "
            f"student={student_summary['tokenizer_length']}, teacher={teacher_summary['tokenizer_length']}"
        )
    if student_summary["vocab_size"] != teacher_summary["vocab_size"]:
        failures.append(
            f"vocab sizes differ: student={student_summary['vocab_size']}, teacher={teacher_summary['vocab_size']}"
        )
    if not vocab_report["match"]:
        failures.append("token-to-id vocabulary mappings differ")
    if not id_to_token_report["match"]:
        failures.append("id-to-token mappings differ")
    if student_summary["special_token_ids"] != teacher_summary["special_token_ids"]:
        failures.append("special token ids differ")

    if student_summary["tokenizer_class"] != teacher_summary["tokenizer_class"]:
        warnings.append(
            "tokenizer classes differ: "
            f"student={student_summary['tokenizer_class']}, teacher={teacher_summary['tokenizer_class']}"
        )
    if student_summary["special_tokens_map"] != teacher_summary["special_tokens_map"]:
        warnings.append("special token maps differ")
    if student_summary["chat_template_sha256"] != teacher_summary["chat_template_sha256"]:
        warnings.append(
            "chat templates differ; MiniLLM usually formats prompts with the trainer processing_class, "
            "but this can still matter for data preparation"
        )
    if student_summary["backend_tokenizer_sha256"] != teacher_summary["backend_tokenizer_sha256"]:
        warnings.append("backend tokenizer fingerprints differ; text encoding may not be identical")

    file_report = compare_file_fingerprints(student_summary, teacher_summary, max_examples=max_examples)
    if not file_report["match"]:
        warnings.append("tokenizer file fingerprints differ")

    return {
        "mini_llm_compatible": len(failures) == 0,
        "strict_match": len(failures) == 0 and len(warnings) == 0,
        "failures": failures,
        "warnings": warnings,
        "student": student_summary,
        "teacher": teacher_summary,
        "vocab": vocab_report,
        "id_to_token": id_to_token_report,
        "files": file_report,
    }


def print_human_report(report: dict[str, Any], *, strict: bool) -> None:
    status_key = "strict_match" if strict else "mini_llm_compatible"
    status = "PASS" if report[status_key] else "FAIL"
    requirement = "strict tokenizer equality" if strict else "MiniLLM token-id compatibility"
    print(f"{status}: {requirement}")
    print()
    print(
        "Student: "
        f"{report['student']['name_or_path']} "
        f"({report['student']['tokenizer_class']}, len={report['student']['tokenizer_length']}, "
        f"vocab_size={report['student']['vocab_size']})"
    )
    print(
        "Teacher: "
        f"{report['teacher']['name_or_path']} "
        f"({report['teacher']['tokenizer_class']}, len={report['teacher']['tokenizer_length']}, "
        f"vocab_size={report['teacher']['vocab_size']})"
    )

    if report["failures"]:
        print("\nFailures:")
        for failure in report["failures"]:
            print(f"- {failure}")

    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")

    if report["vocab"]["shared_token_id_mismatch_examples"]:
        print("\nShared token id mismatch examples:")
        for example in report["vocab"]["shared_token_id_mismatch_examples"]:
            print(f"- {example}")

    if report["id_to_token"]["mismatch_examples"]:
        print("\nId-to-token mismatch examples:")
        for example in report["id_to_token"]["mismatch_examples"]:
            print(f"- {example}")

    if report["files"]["mismatch_examples"]:
        print("\nTokenizer file fingerprint mismatch examples:")
        for example in report["files"]["mismatch_examples"]:
            print(f"- {example}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that a MiniLLM student and teacher tokenizer expose the same token-id space. "
            "By default the exit status only depends on token-id compatibility; use --strict to also fail on "
            "diagnostic differences such as tokenizer class, chat template, backend tokenizer, or tokenizer files."
        )
    )
    parser.add_argument("student_model", help="Student model or tokenizer name/path.")
    parser.add_argument("teacher_model", help="Teacher model or tokenizer name/path.")
    parser.add_argument("--student-revision", default=None, help="Optional student revision.")
    parser.add_argument("--teacher-revision", default=None, help="Optional teacher revision.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download tokenizer files.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to transformers.")
    parser.add_argument("--max-examples", type=int, default=10, help="Maximum mismatch examples to print per section.")
    parser.add_argument("--json", action="store_true", help="Print the full machine-readable report.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warning-level differences in addition to MiniLLM token-id incompatibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    student_tokenizer = load_tokenizer(
        args.student_model,
        revision=args.student_revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    teacher_tokenizer = load_tokenizer(
        args.teacher_model,
        revision=args.teacher_revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    report = compare_tokenizers(
        student_tokenizer,
        teacher_tokenizer,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        max_examples=args.max_examples,
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_human_report(report, strict=args.strict)

    passed = report["strict_match"] if args.strict else report["mini_llm_compatible"]
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
