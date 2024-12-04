#!/usr/bin/env python3
from dataclasses import dataclass
from typing import IO, Any, Callable
from pathlib import Path
from numpy import who
from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat
import random
import logging
import copy
import uuid
import json
import pandas as pd
import click

logging.basicConfig(level=logging.INFO)


@dataclass
class Link:
    start: int
    end: int
    tag: str
    file_id: str
    color: str
    source: str  # annoid of source
    target: str  # annoid of target
    text: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        return cls(
            start=d["start"],
            end=d["end"],
            tag=d["tag"],
            file_id=d["file_id"],
            source=d["source"],
            target=d["target"],
            color=d.get("color", "#dedede"),
            text=d["text"],
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
    file_id: str
    annoid: str
    links: list[Link]

    @classmethod
    def from_dict(cls, jsonl: dict[str, Any] | pd.Series):
        links = [Link.from_dict(l) for l in jsonl["links"]]
        default = {}
        default["color"] = "#dedede"
        default["file_id"] = Path(jsonl["file_id"]).stem.split(".")[0]
        default["text"] = jsonl["text"]
        default["tag"] = jsonl["tag"]
        default["start"] = int(jsonl["start"])  # type:ignore
        default["end"] = int(jsonl["end"])  # type:ignore
        default["links"] = links
        default["annoid"] = str(uuid.uuid4())
        return cls(**default)

    @classmethod
    def from_file(cls, dtpe: str | Path, nr: str | Path):
        # Load the DTPE annotations
        dtpe_df = pd.read_json(dtpe)
        if "links" not in dtpe_df.columns:
            dtpe_df["links"] = [[] for _ in dtpe_df.iterrows()]

        # Load the NR annotations
        nr_df = pd.read_json(nr)
        if "links" not in nr_df.columns:
            nr_df["links"] = [[] for _ in nr_df.iterrows()]

        return dtpe_df.apply(cls.from_dict, axis=1).to_list() + nr_df.apply(cls.from_dict, axis=1).to_list()

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
        file_id=tgt.file_id,
        color=tgt.color,
        source=src.annoid,
        target=tgt.annoid,
        text=tgt.text,
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


class AutoLinker:
    def link(self, annotations: list[Annotation], tags: list[str], max_links: int = 5, num_procs: int = 1):
        self.max_links = max_links

        # Create filters in advance
        self.names = [anno for anno in annotations if anno.tag == "name"]
        self.references = [anno for anno in annotations if anno.tag == "reference"]
        self.definitions = [anno for anno in annotations if anno.tag == "definition"]
        self.theorems = [anno for anno in annotations if anno.tag == "theorem"]
        self.proofs = [anno for anno in annotations if anno.tag == "proof"]
        self.examples = [anno for anno in annotations if anno.tag == "example"]

        if "proof" in tags:
            with Pool(processes=num_procs) as pool:
                self.proofs = pool.starmap(
                    self.link_proof_to_theorem, tqdm(zip(self.proofs, repeat(self.theorems)), total=len(self.proofs))
                )

        if "name" in tags:
            with Pool(processes=num_procs) as pool:
                defs_and_thms = self.definitions + self.theorems
                self.names = pool.starmap(
                    self.link_name_to_entity, tqdm(zip(self.names, repeat(defs_and_thms)), total=len(self.names))
                )

        if "reference" in tags:
            assert "name" in tags, "Can't link references to names without any names"
            logging.warning("Only linking a random pool of 1000 references!! Gotta change the hardcode")
            logging.warning("Don't forget to change the number of returned candidates too!!!")
            with Pool(processes=num_procs) as pool:
                references = random.sample(self.references, 1000)
                self.references = pool.starmap(
                    self.link_ref_to_name, tqdm(zip(references, repeat(self.names)), total=len(references))
                )
                # self.references = pool.starmap(
                #     self.link_ref_to_name, tqdm(zip(self.references, repeat(self.names)), total=len(self.references))
                # )
        return self.definitions + self.theorems + self.examples + self.proofs + self.names + self.references

    @staticmethod
    def link_name_to_entity(name: Annotation, targets: list[Annotation]):
        if name.tag != "name":
            raise Exception(f"Cannot call 'link_name_to_entity' on annotation type {name.tag}")

        # Check to see if `name` is inside any annotation
        target = None
        for guess in filter(lambda t: is_inside(name, t) and (name.file_id == t.file_id), targets):
            if target is None or (target.end - target.start) > (guess.end - guess.start):
                target = guess

        # If we found one (or if we found multiple, the shortest one), then link it
        if target is not None:
            return toggle_link(name, target, force_enable=True)

        # Otherwise give up
        return name

    @staticmethod
    def link_ref_to_name(ref: Annotation, targets: list[Annotation]):
        if ref.tag != "reference":
            raise Exception(f"Cannot call 'link_ref_to_name' on annotation type {ref.tag}")

        # if len(ref.text) <= 4:
        #     return ref

        # First try to match the whole name to a target. Link to every exact match we can find.
        matches = filter(
            lambda name: (name.end < ref.start or name.file_id != ref.file_id) and match_name(name.text, ref.text),
            targets,
        )
        for m in matches:
            ref = toggle_link(ref, m, force_enable=True)

        # For each token in the reference, check to see if there's a matching name
        for token in ref.text.split(" "):
            matches = filter(
                lambda name: (name.end < ref.start or name.file_id != ref.file_id)
                and match_name(name.text, token)
                and len(name.links) > 0,
                targets,
            )

            # Sort the matches by number of shared words between the reference and the name's target entity
            def sortkey(name):
                shared_words = 0
                tgts = set(name.links[0].text.split())
                for token in ref.text.split(" "):
                    if token in tgts:
                        shared_words += 1
                if name.file_id == ref.file_id:
                    shared_words += (ref.start - name.start) / 100_000

                return -shared_words

            matches = sorted(matches, key=sortkey)

            # if we find a match, link to the best choices up to 5 total
            # remaining = 5 - len(ref.links)
            for match in matches:
                ref = toggle_link(ref, match, force_enable=True)
        return ref

    @staticmethod
    def link_proof_to_theorem(proof: Annotation, targets: list[Annotation]):
        candidates = filter(
            lambda anno: abs(anno.end - proof.start) <= 250 and anno != proof,
            targets,
        )
        candidates = sorted(candidates, key=lambda x: -x.start)
        if len(candidates) > 0:
            return toggle_link(proof, candidates[0])
        return proof


@click.command("link")
@click.option("--nr", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--dtpe", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--output", type=click.Path(writable=True, dir_okay=False), required=True)
@click.option(
    "--tags", type=str, multiple=True, default=("proof", "theorem", "definition", "example", "reference", "name")
)
@click.option("--max_links", type=int, default=5)
@click.option("--nprocs", type=int, default=1)
def cli(nr: str, dtpe: str, output: Path, tags: list[str], max_links: int, nprocs: int):
    annotations = Annotation.from_file(dtpe=dtpe, nr=nr)
    logging.info(f"Loaded {len(annotations)} from {dtpe} and {nr}")

    linker = AutoLinker()
    linked_annos = linker.link(annotations, tags=tags, num_procs=nprocs, max_links=max_links)
    num_links = sum([len(l.links) for l in linked_annos])
    logging.info(f"Created {num_links} links.")

    pd.DataFrame.from_records([a.to_json() for a in linked_annos]).to_json(output)
    logging.info(f"Saved output to {output}.")


if __name__ == "__main__":
    cli()
