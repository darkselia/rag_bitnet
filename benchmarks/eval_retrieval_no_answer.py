"""Evaluate retrieval on a small, isolated corpus with answerable and unanswerable questions."""

import argparse
import json
import os
import tempfile

DOCUMENTS = [
    ("observatory", "The Aurora Observatory has a blue dome. It opens at 9:00 on Tuesdays."),
    ("garden", "The Linden Garden has white lilies. Visitors enter through its east gate."),
    ("train", "The Silver train leaves Redford station at 14:30 and arrives at Bellport at 16:00."),
    ("library", "The Willow Library lends books for 14 days and closes on Mondays."),
    ("museum", "Музей Севера находится на улице Лесной. Его эмблема — белый медведь."),
    ("festival", "Фестиваль Ручей проводится в июне. Главный концерт проходит в парке."),
]

# A missing expected ID means none of the documents contains the requested answer.
CASES = [
    ("What color is the Aurora Observatory dome?", "observatory"),
    ("When does the Aurora Observatory open on Tuesdays?", "observatory"),
    ("Where do visitors enter the Linden Garden?", "garden"),
    ("When does the Silver train leave Redford?", "train"),
    ("How long does the Willow Library lend books?", "library"),
    ("Где находится Музей Севера?", "museum"),
    ("Когда проводится фестиваль Ручей?", "festival"),
    ("Какого цвета купол обсерватории Аврора?", "observatory"),
    ("How long may I borrow a book?", "library"),
    ("Does the Willow Library open on Mondays?", "library"),
    ("How much is admission to the Aurora Observatory?", None),
    ("Who designed the Aurora Observatory?", None),
    ("When does the Aurora Observatory close on Tuesdays?", None),
    ("What time does the Linden Garden close?", None),
    ("What is the Silver train ticket price?", None),
    ("What time does the Silver train leave Bellport?", None),
    ("Who founded the Willow Library?", None),
    ("Сколько стоит билет в Музей Севера?", None),
    ("Кто организовал фестиваль Ручей?", None),
    ("What is the population of Mars?", None),
    ("Who wrote Hamlet?", None),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-threshold", type=float, default=0.0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="rag-retrieval-eval-") as data_dir:
        os.environ["DATA_DIR"] = data_dir
        os.environ["RERANKER_MIN_SCORE"] = ""
        from common import config
        from rag import data_store
        from rag.retrieval import retrieve

        data_store.add_chunks([
            {"id": doc_id, "text": text, "source": doc_id, "kind": "docs", "meta": {}}
            for doc_id, text in DOCUMENTS
        ])
        positive = negative = hits = rejected = 0
        scored_results = []
        print(json.dumps({
            "embed_model": config.EMBED_MODEL,
            "reranker": config.RERANKER_MODEL if config.USE_CROSS_RERANKER else None,
            "sim_threshold": args.sim_threshold,
        }, ensure_ascii=False))
        for question, expected in CASES:
            results = retrieve(question, k_vec=6, k_bm25=6, k_final=1, sim_threshold=args.sim_threshold)
            found = results[0] if results else None
            scored_results.append((expected, found))
            if expected is None:
                negative += 1
                rejected += int(found is None)
            else:
                positive += 1
                hits += int(found is not None and found["id"] == expected)
            print(json.dumps({
                "question": question,
                "expected": expected,
                "found": found["id"] if found else None,
                "score": round(found["score"], 4) if found else None,
            }, ensure_ascii=False))
        print(json.dumps({
            "answerable_top1": f"{hits}/{positive}",
            "unanswerable_without_context": f"{rejected}/{negative}",
        }))
        if config.USE_CROSS_RERANKER:
            for threshold in (0.05, 0.1, 0.5):
                kept = sum(1 for expected, found in scored_results if expected is not None and found
                           and found["id"] == expected and found["score"] >= threshold)
                rejected = sum(1 for expected, found in scored_results if expected is None
                               and (not found or found["score"] < threshold))
                print(json.dumps({"reranker_threshold": threshold, "answerable_top1": f"{kept}/{positive}",
                                  "unanswerable_without_context": f"{rejected}/{negative}"}))


if __name__ == "__main__":
    main()
