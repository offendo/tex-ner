#!/usr/bin/env python3
from dataclasses import dataclass
from typing import IO, Any, Callable
import logging
from pathlib import Path
import copy
import uuid
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

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        return cls(
            start=d["start"],
            end=d["end"],
            tag=d["tag"],
            fileid=d["fileid"],
            source=d["source"],
            target=d["target"],
            color=d.get("color", "#dedede"),
        )

    def to_json(self):
        return self.__dict__


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
    def from_dict(cls, jsonl: dict[str, Any] | pd.Series):
        links = [Link.from_dict(l) for l in jsonl["links"]]
        default = {}
        default["color"] = "#dedede"
        default["fileid"] = jsonl["fileid"]
        default["text"] = jsonl["text"]
        default["tag"] = jsonl["tag"]
        default["start"] = int(jsonl["start"])  # type:ignore
        default["end"] = int(jsonl["end"])  # type:ignore
        default["links"] = links
        default["annoid"] = str(uuid.uuid4())
        return cls(**default)

    @classmethod
    def from_file(cls, file: str | Path):
        df = pd.read_json(file)
        fileid = Path(file).with_suffix("").stem
        if "links" not in df.columns:
            df["links"] = [[] for _ in df.iterrows()]
        df["fileid"] = fileid
        return df.apply(cls.from_dict, axis=1).to_list()

    def to_json(self):
        links = [l.to_json() for l in self.links]
        res = self.__dict__
        res["links"] = links
        return res


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
        src.links = [*src.links, link]
        return src
    # If we're forcing an addition but it's already in there, just return
    elif force_enable:
        return src
    # Otherwise, remove it and return
    elif link not in src.links:
        src.links = [*src.links, link]
        return src
    else:
        src.links.remove(link)
        return src


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
            for rule in self.rules.values():
                did_link = rule(anno, idx, annotations)
                if int(did_link) > 0:
                    break
        return annotations

    @rule("link_name_to_outside")
    @staticmethod
    def link_name_to_outside(name: Annotation, idx: int, annotations: list[Annotation]):
        if name.tag != "name":
            return False

        # Check to see if `name` is inside any annotation
        target = None
        for guess in annotations:
            if guess == name:
                continue
            if guess.tag not in {"definition", "theorem", "example"}:
                continue
            elif target == None or (target.end - target.start) > (guess.end - guess.start):
                target = guess

        # If we found one (or if we found multiple, the shortest one), then link it
        if target is not None:
            annotations[idx] = toggle_link(name, target, force_enable=True)
            return True

        # Otherwise give up
        return False

    @rule("link_ref_to_outside")
    @staticmethod
    def link_ref_to_name(ref: Annotation, idx: int, annotations: list[Annotation]):
        if ref.tag != "reference":
            return False

        # For each token in the reference, check to see if there's a matching name
        total = 0
        for token in ref.text.split(" "):
            matches = filter(
                lambda name: match_name(name.text, token)
                and name.end < ref.start
                and name.tag == "name"
                and name != ref,
                annotations,
            )
            matches = sorted(matches, key=lambda anno: -anno.start)

            # if we find a match, link to the most recent one (reverse sorted by start index)
            if len(matches) > 0:
                annotations[idx] = toggle_link(ref, matches[0], force_enable=True)
                total += 1
        return total

    @rule("link_proof_to_theorem")
    @staticmethod
    def link_proof_to_theorem(proof: Annotation, idx: int, annotations: list[Annotation]):
        candidates = filter(
            lambda anno: anno.tag == "theorem" and abs(anno.end - proof.start) <= 250 and anno != proof,
            annotations,
        )
        candidates = sorted(candidates, key=lambda x: -x.start)
        if len(candidates) > 0:
            annotations[idx] = toggle_link(proof, candidates[0])
            return True
        return False


@click.command("link")
@click.option("--file", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--output", type=click.Path(writable=True), required=True)
def cli(file: str, output: Path):
    linker = AutoLinker()
    annos = linker.link(Annotation.from_file(file))
    with open(output, "w") as f:
        json.dump([a.to_json() for a in annos], f)


if __name__ == "__main__":
    cli()
