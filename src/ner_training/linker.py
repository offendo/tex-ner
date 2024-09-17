#!/usr/bin/env python3
from dataclasses import dataclass
from typing import IO, Any, Callable
import logging
from pathlib import Path
import json
import pandas as pd
import click


@dataclass
class Link:
    start: int
    end: int
    tag: str
    fileid: str
    color: str
    source: str  # annoid of source
    target: str  # annoid of target


@dataclass
class Annotation:
    text: str
    start: int
    end: int
    tag: str
    color: str
    fileid: str
    annoid: str
    links: list[Link]

    @classmethod
    def from_dict(cls, json: dict[str, Any]):
        links = [Link(**l) for l in json["links"]]
        return cls(**{k: v for k, v in json.items() if k != "links"}, links=links)

    @classmethod
    def from_records(cls, json: list[dict[str, Any]]):
        return [Annotation.from_dict(j) for j in json]


def toggle_link(src: Annotation, tgt: Annotation, force_enable: bool = False):
    link = Link(
        start=tgt.start,
        end=tgt.end,
        tag=tgt.tag,
        fileid=tgt.fileid,
        color=tgt.color,
        source=src.annoid,
        target=tgt.annoid,
    )

    # If we're forcing an addition and it's not already in there, add it
    if force_enable and link not in src.links:
        src.links.append(link)
        return src.links
    # If we're forcing an addition but it's already in there, just return
    elif force_enable:
        return src.links
    # Otherwise, remove it and return
    elif link not in src.links:
        return src.links
    else:
        src.links.remove(link)
        return src.links


def match_name(name: str, ref: str):
    # 1. Check exact match
    # 2. Check plurals
    if ref == name or ref == name + "s" or ref == name + "es":
        return True

    # Do same with name/ref reversed
    if name == ref or name == ref + "s" or name == ref + "es":
        return True

    # Otherwise no match
    return False


def is_inside(inside: Annotation, outside: Annotation):
    return inside.start >= outside.start and inside.end <= outside.end


def make_rule_decorator(rules):
    def inner(key):
        def decorator(fn):
            rules[key] = fn

            def _(*args, **kwargs):
                result = fn(*args, **kwargs)
                if result:
                    logging.info("Autolinked with {key}")
                return result

        return decorator

    return inner


class AutoLinker:
    rules: dict[str, Callable] = {}
    rule = make_rule_decorator(rules)

    def link(self, annotations: list[Annotation]):
        for idx, anno in enumerate(annotations):
            rest = annotations[:idx] + annotations[idx + 1 :]
            for rule in self.rules.values():
                did_link = rule(anno, rest)
                if did_link:
                    break

    @rule("link_name_to_outside")
    @staticmethod
    def link_name_to_outside(name: Annotation, annotations: list[Annotation]):
        if name.tag != "name":
            return False

        # Check to see if `name` is inside any annotation
        target = None
        for guess in annotations:
            if guess.tag not in {"definition", "theorem", "example"}:
                continue
            elif target == None or (target.end - target.start) > (
                guess.end - guess.start
            ):
                target = guess

        # If we found one (or if we found multiple, the shortest one), then link it
        if target is not None:
            toggle_link(name, guess, force_enable=True)
            return True

        # Otherwise give up
        return False

    @rule("link_ref_to_outside")
    @staticmethod
    def link_ref_to_name(ref: Annotation, annotations: list[Annotation]):
        if ref.tag != "reference":
            return False

        # For each token in the reference, check to see if there's a matching name
        total = 0
        for token in ref.text.split(" "):
            matches = filter(
                lambda name: match_name(name.text, token) and name.end < ref.start,
                annotations,
            )
            matches = sorted(matches, key=lambda anno: -anno.start)

            # if we find a match, link to the most recent one (reverse sorted by start index)
            if len(matches) > 0:
                toggle_link(ref, matches[0], force_enable=True)
                total += 1
        return total

    @rule("link_proof_to_theorem")
    @staticmethod
    def link_proof_to_theorem(proof: Annotation, annotations: list[Annotation]):
        candidates = filter(
            lambda anno: anno.tag == "theorem" and (anno.end - proof.start) <= 250,
            annotations,
        )
        candidates = sorted(candidates, key=lambda x: -x.start)
        if len(candidates) > 0:
            toggle_link(proof, candidates[0])
            return True
        return False


@click.command("link")
@click.option("--file", type=click.File("r"))
@click.option("--key", type=str, default=None)
@click.option("--output", type=click.Path(writable=True))
def cli(file: IO, key: str | None, output: Path):
    linker = AutoLinker()
    annos = json.load(file)
    if key is not None:
        annos = annos[key]

    linker.link(Annotation.from_records(annos))
    with open(output, "w") as f:
        json.dump(annos, f)


if __name__ == "__main__":
    cli()
